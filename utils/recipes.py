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


prompt_map = {
    "memory": """
        You judge matches of user intent with those stored in a database to decide if they are true matches of intent.
        When asked to compare two intents, check they are the same, have the same entities and would result in the same outcome.
        If the user doesn't specify data source, but everything else matches, it's a match

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
        If the user doesn't specify data source, but everything else matches, it's a match

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
    result_found = False
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

    # Now run them through an AI check
    r = {"score": None, "content": None, "metadata": None}
    for d in matches:
        result_found = False
        score = d[1]
        content = d[0].page_content
        metadata = d[0].metadata
        mem_type = metadata["mem_type"]
        if debug:
            print(
                f"======= AI Checking {mem_type} for intent: {intent} \nContent: {content}"
            )
        prompt = f"""
            User Intent:

            {intent}

            DB Intent:

            {content}

        """
        response = call_llm(prompt_map[mem_type], prompt)
        if "content" in response:
            response = response["content"]
        if isinstance(response, str):
            response = json.loads(response)
        if debug:
            print("AI Judge of match: ", response, "\n")

        if response["answer"].lower() == "yes":
            print("    MATCH!")
            r["score"] = score
            r["content"] = content
            r["metadata"] = metadata
            result_found = True
            return result_found, r

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


def generate_intent_from_history(chat_history: list, remove_code: bool = True) -> dict:
    """
    Generate the intent from the user query and chat history.

    Args:
        chat_history (str): The chat history.

    Returns:
        dict: The generated intent.

    """

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

    prompt = (
        """
        You are an intelligent assistant tasked with converting a user input into a standard intent phrase.

        Here is the user input:

        """
        + str(chat_history)
        + """

        The standard format has the following fields and possible values:

        action: The specific action the user wants to perform (e.g., "plot", "generate", "export", "provide").
        visualization_type: The type of visualization, if applicable (e.g., "bar chart", "line chart", "pie chart", "text summary"). Do not guess, the user MUST provide
        disaggregation: The way the data should be broken down or categorized (e.g., "by state", "by company", "by year").
        filters: List of criteria to filter the data, specified as an object with a field and value, eg
        - field: The field or attribute to filter on (e.g., "region", "year").
        - value: The specific value to filter by (e.g., "North America", "2023"). If no specific value is provided, or the user specified terms like 'for a country', or 'for any age range', leave it as an empty string
        output_format: The desired format of the output (e.g., "image", "csv", "text").
        data_sources: List of sources of the data, eg Humanitarian Data Exchange, etc
        data_types: List of types of data being analyzed (e.g., "population", "GDP", "sales").
        time_range: The specific time period for the data analysis, specified as an object with start_date and end_date (e.g., {"start_date": "2022-01-01", "end_date": "2022-12-31"}).
        granularity: The time level of detail in the data (e.g., "daily", "monthly", "yearly").
        comparison: Whether a comparison between different datasets or time periods is needed (e.g., true, or "").
        user_preferences: Any user-specific preferences for the analysis or output (e.g., "include outliers", "show trend lines").

        Where the output should use the above fields in the order they are listed.


        Example output for the above input:

        plot a bar chart of population by state for 2023 using HDX(HAPI) data, highlighting top 5 states as an image

        Task:

        Produce the user's intent in the standard format, answering with a JSON record ...

        {
            "intent": <user intent sentence>
        }

    """
    )

    intent = call_llm(instructions="", prompt=prompt)
    if not isinstance(intent, dict):
        intent = {"intent": intent}
    print(f"Generated intent: {intent}")
    return intent["intent"]


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
    metadata = table_data["metadata"]
    if metadata is None:
        metadata = ""
    print(f"====> Found {mem_type}")
    if table_data["result_type"] == "image":
        response_image = table_data["result"]
        response_text = ""
    else:
        response_text = table_data["result"]
        response_image = ""
    recipe_id = table_data["custom_id"]
    print("Recipe ID: ", recipe_id, "Intent: ", content)
    if response_image is not None and response_image != "":
        process_image(response_image.replace("data:image/png;base64,", ""), recipe_id)
        # result = "http://localhost:9999/memory_image.png"
        result = f"{os.getenv('IMAGE_HOST')}/memory_image_{recipe_id}.png"
    else:
        result = response_text

    return {"result": result, "metadata": metadata}


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
        # result["errors"] = run_output.stderr

        # TODO - this is terrible, just for the demo, extract JSON between "{" and "}""
        # Match { }
        metadata = ""
        if result["output"].find("{") != -1:
            result["output"] = result["output"][result["output"].find("{") :]
            result["output"] = result["output"][: result["output"].rfind("}") + 1]
            print("Output: ", result["output"])
            j = json.loads(result["output"].replace("'", '"'))
            if "metadata" in j:
                metadata = j["metadata"]
        else:
            metadata = "Data was sourced from HDX"

        result["metadata"] = metadata

    print("Recipe executed successfully.")
    print(result)
    return result["output"] + " >> metadata: " + metadata


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

    generate_intent = "false"

    if generate_intent is not None and generate_intent == "true":
        # chat history is passed from promptflow as a string representation of a list and this has to be converted back to a list for the intent generation to work!
        history_list = ast.literal_eval(chat_history)
        history_list.append({"inputs": {"question": user_input}})
        user_input = generate_intent_from_history(history_list)
        # turn user_input into a proper json record
        user_input = json.dumps(user_input)

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

        result = re.escape(str(result))
        print(result)

        result = f"match_type: {mem_type};\n matched_doc: {matched_doc};\n result: {str(result)}"

        return result

    result = "Sorry, no recipe or found"
    print(result)

    return str(result)
