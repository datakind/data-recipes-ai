import ast
import base64
import io
import json
import logging
import os
import re
import subprocess
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
from jinja2 import Environment, FileSystemLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores.pgvector import PGVector
from PIL import Image

from utils.db import execute_query
from utils.llm import call_llm, get_models

import time

environment = Environment(loader=FileSystemLoader("./templates/"))

db = None

warnings.filterwarnings("ignore")

# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)

load_dotenv()

recipes_work_dir = "./recipes"

# Lower numbers are more similar
similarity_cutoff = {
    "memory": os.getenv("MEMORY_SIMILARITY_CUTOFF", 0.2),
    "recipe": os.getenv("RECIPE_SIMILARITY_CUTOFF", 0.3),
    "helper_function": os.getenv("HELPER_FUNCTION_SIMILARITY_CUTOFF", 0.1),
}

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

#time tracking wrapper
import time
import pandas as pd

# Define the time_tracker decorator
def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' ran for {duration:.4f} seconds")
        return result, duration
    return wrapper

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
        result_found, result = check_recipe_memory(intent, debug=False)
        if result_found is True:
            if (
                result["score"] is not None
                and result["score"] < similarity_cutoff[mem_type]
            ):
                message = f"{mem_type} already exists: {result['content']}"
                response = {
                    "already_exists": "true",
                    "message": message,
                    "custom_id": result["metadata"]["custom_id"],
                }
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

@time_tracker
def check_recipe_memory(intent, debug=True):
    """
    Check the memory for a given intent.

    Args:
        intent (str): The intent to search for in the memory.
        debug (bool, optional): If True, print debug information. Default is True.

    Returns:
        dict: A dictionary containing the score, content, and metadata of the best match found in the memory.
            If no match is found, the dictionary values will be None.
    """

    global db
    if db is None:
        db = initialize_vector_db()

    # First do semantic search across memories and recipies
    matches = []
    for mem_type in ["memory", "recipe"]:
        if debug:
            print(f"======= Checking {mem_type} for intent: {intent}")
        docs = db[mem_type].similarity_search_with_score(intent, k=3)
        for d in docs:
            score = d[1]
            content = d[0].page_content
            metadata = d[0].metadata
            if debug:
                print("\n", f"Score: {score} ===> {content}")

            if d[1] < similarity_cutoff[mem_type]:
                matches.append(d)

    r = {"score": None, "content": None, "metadata": None}
    result_found = False

    # No matches, no point calling the AI
    if len(matches) == 0:
        return result_found, r

    # Build a list for the AI judge to review
    match_list = ""
    for i, d in enumerate(matches):
        match_list += f"{i+1}. {d[0].page_content}\n"

    ai_memory_judge_prompt = environment.get_template("ai_memory_judge_prompt.jinja2")
    prompt = ai_memory_judge_prompt.render(
        user_input=intent, possible_matches=match_list
    )
    print(prompt)
    response = call_llm("", prompt)
    print(response)
    if "content" in response:
        response = response["content"]
    if isinstance(response, str):
        response = json.loads(response)
    if debug:
        print("AI Judge of match: ", response, "\n")
    if response["answer"].lower() == "yes":
        print("    MATCH!")
        match_id = response["match_id"]
        d = matches[int(match_id) - 1]
        score = d[1]
        content = d[0].page_content
        metadata = d[0].metadata
        r["score"] = score
        r["content"] = content
        r["metadata"] = metadata
        result_found = True
    print(r)
    return result_found, r


def get_memory_recipe_metadata(custom_id, mem_type):
    """
    Get the metadata for a memory or recipe by Querying the database with the given custom ID.

    Args:
        custom_id (str): The custom ID of the memory or recipe document to get the metadata for.
        mem_type (str): The type of memory store to search in. Can be 'memory', 'recipe', or 'helper_function'.

    Returns:
        dict: The table data of the memory or recipe document with the given custom ID.
    """

    # Execute a SQL query to get the table data for the given custom ID
    query = f"""
        SELECT
            *
        FROM
            {mem_type}
        WHERE
            custom_id = '{custom_id}'
    """
    print(query)
    result = execute_query(query, "recipe")

    if result.shape[0] > 0:
        result = result.iloc[0]
        result = json.loads(result.to_json())
        return result
    else:
        raise ValueError(
            f"No table data (memory/recipe) found for custom ID {custom_id}"
        )


def generate_intent_from_history(chat_history: str) -> dict:
    """
    Generate the intent from the user query and chat history.

    Args:
        chat_history (str): The chat history.

    Returns:
        dict: The generated intent.

    """

    # Load ninja template
    generate_intent_from_history_prompt = environment.get_template(
        "generate_intent_from_history_prompt.jinja2"
    )

    # Generate the intent from the chat history
    prompt = generate_intent_from_history_prompt.render(chat_history=chat_history)

    intent = call_llm(instructions="", prompt=prompt)
    # if not isinstance(intent, dict):
    #    intent = {"intent": intent}
    print(f"Generated intent: {intent}")
    return intent


def process_image(encoded_string, recipe_id):
    """
    Takes a base64 encoded string of a picture, decodes it, and saves it as a PNG file.

    Args:
    encoded_string (str): Base64 encoded string of the image.
    recipe_id (str): The recipe ID to use in the image file name.

    Returns:
    str: Full path to the saved image file.
    """

    print("A visual memory was found. Processing image...")

    # Decode the base64 string
    image_data = base64.b64decode(encoded_string)

    # Convert binary data to image
    image = Image.open(io.BytesIO(image_data))

    # Create the full path for saving the image
    full_path = os.path.join("./recipes/public/", f"memory_image_{recipe_id}.png")

    # Save the image
    image.save(full_path, "PNG")

    print("Image processed and saved successfully.")

    return full_path


# TODO recfactor all these functions into simpler single function. Tech debt left
# over from Robocorp rapid prototype
def process_memory_recipe_results(result: dict, table_data: dict) -> str:
    """
    Processes the results of a memory recipe search and returns the response text and metadata.

    Args:
        result (dict): The result of the memory recipe search.
        table_data (dict): The data from the memory or recipe tables.

    Returns:
    """

    mem_type = result["metadata"]["mem_type"]
    custom_id = result["metadata"]["custom_id"]
    print(result)
    content = result["content"]
    table_data = get_memory_recipe_metadata(custom_id, mem_type)
    recipe_id = table_data["custom_id"]
    if "result_metadata" in table_data:
        metadata = table_data["result_metadata"]
    else:
        metadata = table_data["sample_result_metadata"]
    if metadata is None:
        metadata = ""

    print(f"====> Found {mem_type}")
    if table_data["result_type"] == "image":
        image = table_data["result"]
        process_image(image.replace("data:image/png;base64,", ""), recipe_id)
        file = f"{os.getenv('IMAGE_HOST')}/memory_image_{recipe_id}.png"
        result = {"type": "image", "file": file, "value": ""}
    else:
        # TODO 'result' in memory should be a JSON, just like sample_result
        # in recipe, then this goes away
        if mem_type == "recipe":
            result = json.loads(table_data["result"])
            result = json.loads(result)
        else:
            print("Memory, skipping result json extraction")
            result = json.loads(table_data["result"])
            result = result["result"]
            print(result)

    print("Recipe ID: ", recipe_id, "Intent: ", content)

    return {"result": result, "metadata": metadata}


# TODO Absolutely needs to be converted to registered functions
def run_recipe(custom_id: str, recipe: dict, user_input, chat_history):
    """
    Runs a recipe based on the result of a memory recipe search.

    Args:
        custom_id (str): The custom ID of the recipe.
        recipe(dict): The recipe details
        user_input (str): The user input.
        chat_history (str): The chat history.

    """

    print("Attempting to run recipe...")
    print(f"Recipe Custom ID: {custom_id}")

    function_code = recipe["function_code"]

    # TODO this should really use the new openapi_json field,
    # but that's blank currently, will come back to it.
    calling_code = recipe["sample_call"]

    recipe_run_python_prompt = environment.get_template(
        "recipe_run_python_prompt.jinja2"
    )
    prompt = recipe_run_python_prompt.render(
        recipe_code=function_code,
        calling_code=calling_code,
        user_input=user_input,
        chat_history=chat_history,
    )
    print("Calling LLM to generate new run code ...")
    new_code = call_llm("", prompt, None)
    print(new_code)

    result = {
        "output": "",
        "errors": "",
        "metadata": "",
    }

    if "new_calling_code" in new_code:
        calling_code = new_code["new_calling_code"]
        print("New calling code generated ...")

        # Combine function and calling code into a string
        code = function_code + "\n\n" + "if __name__ == '__main__':\n\n"

        # Write calling code to code, indented by 4
        code += "    " + calling_code.replace("\n", "\n    ") + "\n"

        # Make recipes folder if it doesn't exist
        if not os.path.exists(recipes_work_dir):
            print(f"Creating recipes directory ... {recipes_work_dir}")
            os.makedirs(recipes_work_dir)
            # Copy skills.py into directory use shutil
            print("Copying skills.py into directory ...")

        # Adjust .env location in code
        code = code.replace("load_dotenv()", "load_dotenv('../.env')")

        # Adjust any images saved in code
        code = re.sub(r"./work/(.*?\.png)", r"/app/recipes/public/\1", code)

        # Adjust path
        code = (
            "import os\nimport sys\n# Add parent folder to path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
            + code
        )

        recipe_path = f"{recipes_work_dir}/{custom_id}.py"
        with open(recipe_path, "w") as f:
            print(f"Writing recipe to file ... {recipe_path}")
            f.write(code)

        os.chdir(recipes_work_dir)
        cmd = f"python {custom_id}.py"
        run_output = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        result["output"] = run_output.stdout
        result["errors"] = run_output.stderr

        # TODO this exists here as well as in cli recipes_sync.py, merge
        if "OUTPUT:" in result["output"]:
            result["output"] = result["output"].split("OUTPUT:")[1]
            print(result["output"])
            result = json.loads(result["output"])
            print("Recipe executed successfully.")
        else:
            print(result["output"])
            print(result["errors"])
            result["result"] = {
                "type": "text",
                "file": "",
                "value": "Recipe produced no output",
            }
            result["metadata"] = {
                "params": {},
                "attribution": "",
                "data_url": "",
                "time_period": {"start": "", "end": ""},
            }

    print(result)
    return result


def get_memory_recipe(user_input, chat_history, generate_intent="true") -> str:
    """
    Performs a search in the memory for a given intent and returns the best match found.

    Args:
        user_input (str): The user input to search for in the memory.
        chat_history (str): The chat history.
        generate_intent (str): A flag to indicate whether to generate the intent from the chat history.

    Returns:
        str: Matched value
        str: metadata
    """

    logging.info("Python HTTP trigger function processed a request.")
    # Retrieve the CSV file from the request

    # Generate intent from chat history if generate_intent is true
    if generate_intent is not None and generate_intent == "true":
        print("********* Generating intent from chat history ...")
        print("Chat history: ", chat_history)
        user_input = generate_intent_from_history(chat_history)
        print("Generated intent: ", user_input)
        user_input = user_input["intent"]

    print("Checking my memories ...")
    memory_found, result = check_recipe_memory(user_input, debug=True)
    if memory_found is True:
        custom_id = result["metadata"]["custom_id"]
        mem_type = result["metadata"]["mem_type"]
        matched_doc = result["content"]
        # Get data from memory or recipe tables
        table_data = get_memory_recipe_metadata(custom_id, mem_type)
        if mem_type == "recipe":
            result = run_recipe(custom_id, table_data, user_input, chat_history)
        else:
            # Take the result directly from memory
            result = process_memory_recipe_results(result, table_data)

        print(result)
        result["memory_type"] = mem_type
        result["memory"] = matched_doc
        result["memory_found"] = "true"

        result_string = json.dumps(result, indent=4)

        print(result_string)
        return result_string

    result = {"result": "Sorry, no recipe or memory found", "memory_found": "false"}
    result = json.dumps(result, indent=4)
    print(result)

    return result
