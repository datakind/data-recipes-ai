import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import requests
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores.pgvector import PGVector

from utils.utils import call_llm, get_models

db = None

warnings.filterwarnings("ignore")

# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)

load_dotenv()

# Lower numbers are more similar
similarity_cutoff = {"memory": 0.1, "recipe": 0.1, "helper_function": 0.1}

conn_params = {
    "RECIPES_OPENAI_API_TYPE": os.getenv("RECIPES_OPENAI_API_TYPE"),
    "RECIPES_OPENAI_API_KEY": os.getenv("RECIPES_OPENAI_API_KEY"),
    "RECIPES_OPENAI_API_ENDPOINT": os.getenv("RECIPES_OPENAI_API_ENDPOINT"),
    "RECIPES_OPENAI_API_VERSION": os.getenv("RECIPES_OPENAI_API_VERSION"),
    "RECIPES_BASE_URL": os.getenv("RECIPES_BASE_URL"),
    "RECIPES_MODEL": os.getenv("RECIPES_MODEL"),
    "RECIPES_OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME": os.getenv(
        "RECIPES_OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME"
    ),
    "POSTGRES_DB": os.getenv("POSTGRES_RECIPE_DB"),
    "POSTGRES_USER": os.getenv("POSTGRES_RECIPE_USER"),
    "POSTGRES_HOST": os.getenv("POSTGRES_RECIPE_HOST"),
    "POSTGRES_PORT": os.getenv("POSTGRES_RECIPE_PORT"),
    "OPENAI_API_KEY": os.getenv("AZURE_API_KEY"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_RECIPE_PASSWORD"),
}


# Stored in langchain_pg_collection and langchain_pg_embedding as this
def initialize_vector_db():
    """
    Initializes the database by creating store tables if they don't exist and returns the initialized database.

    Returns:
        dict: The initialized database with store tables for each memory type.
    """

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
        host=os.environ.get("POSTGRES_RECIPE_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_RECIPE_PORT", "5432")),
        database=os.environ.get("POSTGRES_RECIPE_DB", "postgres"),
        user=os.environ.get("POSTGRES_RECIPE_USER", "postgres"),
        password=os.environ.get("POSTGRES_RECIPE_PASSWORD", "postgres"),
    )

    db = {}

    embedding_model, chat = get_models()

    # This will create store tables if they don't exist
    for mem_type in similarity_cutoff.keys():
        COLLECTION_NAME = f"{mem_type}_embedding"
        db[mem_type] = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embedding_model,
        )

    return db


response_formats = [
    "csv",
    "dataframe",
    "json",
    "plot_image_file_location",
    "shape_file_location",
    "integer",
    "float",
    "string",
]

prompt_map = {
    "memory": """
        You judge matches of user intent with those stored in a database to decide if they are true matches of intent.
        When asked to compare two intents, check they are the same, have the same entities and would result in the same outcome.
        Be very strict in your judgement. If you are not sure, say no.
        A plotting intent is different from a request for just the data.
        A plotting intent will should have output format of plot_image_file_location.


        Answer with a JSON record ...

        {
            "answer": <yes or no>,
            "reason": <your reasoning>"
        }
    """,
    "recipe": """
        You judge matches of user intent with generic DB intents in the database to see if a DB intent can be used to solve the user's intent.
        The requested output format of the user's intent MUST be the same as the output format of the generic DB intent.
        For exmaple, if the user's intent is to get a plot, the generic DB intent MUST also be to get a plot.
        The level of data aggregation is important, if the user's request differs from the DB intent, then reject it.

        Answer with a JSON record ...

        {
            "answer": <yes or no>,
            "reason": <your reasoning>"
        }
    """,
    "helper_function": """
        You judge matches of user input helper functions and those already in the database
        If the user help function matches the database helper function, then it's a match

        Answer with a JSON record ...

        {
            "answer": <yes or no>,
            "reason": <your reasoning>"
        }
    """,
}


def add_recipe_memory(intent, metadata, mem_type="recipe", force=False):
    """
    Add a new memory document to the memory store.

    Parameters:
    - intent (str): The content of the memory document.
    - metadata (dict): Additional metadata for the memory document.
    - mem_type (str): The type of memory store to add the document to.
    - db (Database): The database object representing the memory store.
    - force (bool, optional): If True, force the addition of the memory document even if a similar document already exists. Default is False.

    Returns:
    - id (str): The ID of the added memory document.
    """

    global db
    if db is None:
        db = initialize_vector_db()

    # First see if we already have something in our memory
    if force is False:
        result_found, result = check_recipe_memory(
            intent, mem_type=mem_type, debug=False
        )
        if result_found is True:
            if (
                result["score"] is not None
                and result["score"] < similarity_cutoff[mem_type]
            ):
                message = f"{mem_type} already exists: {result['content']}"
                response = {"already_exists": "true", "message": message}
                return response

    print(f"Adding new document to {mem_type} store ...")
    data = {}
    data["page_content"] = intent

    uuid_str = str(uuid.uuid4())
    metadata["custom_id"] = uuid_str

    metadata["mem_type"] = mem_type

    # print(metadata)

    new_doc = Document(page_content=intent, metadata=metadata)
    print(metadata)
    id = db[mem_type].add_documents([new_doc], ids=[uuid_str])
    return id


def check_recipe_memory(intent, mem_type, debug=True):
    """
    Check the memory for a given intent.

    Args:
        intent (str): The intent to search for in the memory.
        mem_type (str): The type of memory to search in. Can be 'memory', 'recipe', or 'helper_function'.
        db (Database): The database object to perform the search on.

    Returns:
        dict: A dictionary containing the score, content, and metadata of the best match found in the memory.
            If no match is found, the dictionary values will be None.
    """

    global db
    if db is None:
        db = initialize_vector_db()

    if mem_type not in ["memory", "recipe", "helper_function"]:
        print(f"Memory type {mem_type} not recognised")
        sys.exit()
        return
    r = {"score": None, "content": None, "metadata": None}
    result_found = False
    if debug:
        print(f"======= Checking {mem_type} for intent: {intent}")
    docs = db[mem_type].similarity_search_with_score(intent, k=10)
    for d in docs:
        score = d[1]
        content = d[0].page_content
        metadata = d[0].metadata
        if metadata["mem_type"] != mem_type:
            continue
        if debug:
            print("\n", f"Score: {score} ===> {content}")
        if d[1] < similarity_cutoff[mem_type]:

            # Here ask LLM to confirm our match
            prompt = f"""
                User Intent:

                {intent}

                DB Intent:

                {content}

            """

            response = call_llm(prompt_map[mem_type], prompt)

            if debug:
                print("AI Judge of match: ", response)
            print(response)
            if response["answer"].lower() == "yes":
                r["score"] = score
                r["content"] = content
                r["metadata"] = metadata
                result_found = True
                return r
    return result_found, r


def generate_intent_from_history(chat_history: list, remove_code: bool = True) -> dict:
    """
    Generate the intent from the user query and chat history.

    Args:
        chat_history (str): The chat history.

    Returns:
        dict: The generated intent.

    """
    chat = get_models()
    # Only use last few interactions
    buffer = 4
    if len(chat_history) > buffer:
        chat_history = chat_history[-buffer:]

    # Remove any code nodes
    if remove_code is True:
        chat_history2 = []
        for c in chat_history:
            chat_history2.append(c)
            if "code" in c:
                # remove 'code' from dictionary c
                c.pop("code")
        chat_history = chat_history2

    prompt = f"""
        Given the chat history below, what is the user's intent?

        {chat_history}

    """
    intent = call_llm(
        instructions=prompt_map["instructions_intent_from_history"],
        prompt=prompt,
        chat=chat,
    )
    if not isinstance(intent, dict):
        intent = {"intent": intent}
    print(f"Generated intent: {intent}")
    return intent["intent"]
