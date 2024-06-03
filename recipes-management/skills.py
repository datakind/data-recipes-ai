import json
import os
import uuid
from pathlib import Path
from typing import List
import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain_community.vectorstores.pgvector import PGVector
from openai import OpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage

# TODO, temporary while we do user testing
import warnings
warnings.filterwarnings("ignore")

# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)

load_dotenv()

# Lower numbers are more similar
similarity_cutoff = {"memory": 0.1, "recipe": 0.1, "helper_function": 0.1}

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
    host=os.environ.get("POSTGRES_RECIPE_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_RECIPE_PORT", "5432")),
    database=os.environ.get("POSTGRES_RECIPE_DB", "postgres"),
    user=os.environ.get("POSTGRES_RECIPE_USER", "postgres"),
    password=os.environ.get("POSTGRES_RECIPE_PASSWORD", "postgres"),
)
embedding_model = AzureOpenAIEmbeddings(
    deployment=os.getenv("RECIPES_OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("RECIPES_BASE_URL"),
    chunk_size=16,
)

chat = AzureChatOpenAI(
    #model_name="gpt-35-turbo",
    model_name = "gpt-4-turbo",
    azure_endpoint=os.getenv("RECIPES_BASE_URL"),
    api_version=os.getenv("RECIPES_OPENAI_API_VERSION"),
    temperature=1,
    max_tokens=1000,
    response_format={"type": "json_object"}
)

# Stored in langchain_pg_collection and langchain_pg_embedding as this
def initialize_vector_db():
    """
    Initializes the database by creating store tables if they don't exist and returns the initialized database.

    Returns:
        dict: The initialized database with store tables for each memory type.
    """
    db = {}

    # This will create store tables if they don't exist
    for mem_type in similarity_cutoff.keys():
        COLLECTION_NAME = f"{mem_type}_embedding"
        db[mem_type] = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embedding_model,
        )
    
    return db


db = initialize_vector_db()

response_formats = ['csv', 'dataframe', 'json', 'plot_image_file_location', 'shape_file_location','integer', 'float', 'string']

prompt_map ={
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
    """

}

def generate_and_save_images(query: str, image_size: str = "1024x1024") -> List[str]:
    """
    Function to paint, draw or illustrate images based on the users query or request. Generates images from a given query using OpenAI's DALL-E model and saves them to disk.  Use the code below anytime there is a request to create an image.

    :param query: A natural language description of the image to be generated.
    :param image_size: The size of the image to be generated. (default is "1024x1024")
    :return: A list of filenames for the saved images.
    """

    client = OpenAI()  # Initialize the OpenAI client
    response = client.images.generate(
        model="dall-e-3", prompt=query, n=1, size=image_size
    )  # Generate images

    # List to store the file names of saved images
    saved_files = []

    # Check if the response is successful
    if response.data:
        for image_data in response.data:
            # Generate a random UUID as the file name
            file_name = str(uuid.uuid4()) + ".png"  # Assuming the image is a PNG
            file_path = Path(file_name)

            img_url = image_data.url
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                # Write the binary content to a file
                with open(file_path, "wb") as img_file:
                    img_file.write(img_response.content)
                    print(f"Image saved to {file_path}")
                    saved_files.append(str(file_path))
            else:
                print(f"Failed to download the image from {img_url}")
    else:
        print("No image data found in the response!")

    # Return the list of saved files
    return saved_files


def get_connection():
    """
    This function gets a connection to the database
    """
    host = os.getenv("POSTGRES_DATA_HOST")
    port = os.getenv("POSTGRES_DATA_PORT")
    database = os.getenv("POSTGRES_DATA_DB")
    user = os.getenv("POSTGRES_DATA_USER")
    password = os.getenv("POSTGRES_DATA_PASSWORD")

    conn = psycopg2.connect(
        dbname=database, user=user, password=password, host=host, port=port
    )
    return conn


def execute_query(query):
    """
    This skill executes a query in the data database.

    To find out what tables and columns are available, you can run "select table_name, api_name, summary, columns from table_metadata"

    """
    conn = get_connection()
    cur = conn.cursor()

    # Execute the query
    cur.execute(query)

    # Fetch all the returned rows
    rows = cur.fetchall()

    # Close the cursor and connection
    cur.close()
    conn.close()

    return rows

def add_memory(intent, metadata, mem_type="recipe", force=False):
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

    # First see if we already have something in our memory
    if force == False:
        result = check_memory(intent, db=db, mem_type=mem_type, debug=False)
        if result != None:
            if result['score'] != None and result['score'] < similarity_cutoff[mem_type]:
                message = f"{mem_type} already exists: {result['content']}"
                response = {
                    "already_exists": "true",
                    "message": message
                }
                return response
 
    print(f"Adding new document to {mem_type} store ...")
    data = {}
    data['page_content'] = intent

    uuid_str = str(uuid.uuid4())
    metadata['custom_id'] = uuid_str

    metadata['mem_type'] = mem_type

    #print(metadata)

    new_doc =  Document(
        page_content=intent,
        metadata=metadata
    )
    print(metadata)
    id = db[mem_type].add_documents(
        [new_doc],
        ids=[uuid_str]
    )
    return id

def check_memory(intent, mem_type, db, debug=True):
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
    if mem_type not in ['memory','recipe','helper_function']:
        print(f"Memory type {mem_type} not recognised")
        sys.exit()
        return
    r = {
        "score": None,
        "content": None,
        "metadata": None
    }
    if debug:
        print(f"======= Checking {mem_type} for intent: {intent}")
    docs = db[mem_type].similarity_search_with_score(intent, k=10)
    for d in docs:
        score = d[1]
        content = d[0].page_content
        metadata = d[0].metadata
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

            if 'user_intent_output_format' in response:
                if response['user_intent_output_format'] != response['generic_db_output_format']:
                    response['answer'] = 'no'
                    response['reason'] = 'output formats do not match'
            if debug:
                print("AI Judge of match: ", response)
            if response['answer'].lower() == 'yes':
                r["score"] = score
                r["content"] = content
                r["metadata"] = metadata
                return r
    return r


# Stored in langchain_pg_collection and langchain_pg_embedding as this
def initialize_vector_db():
    """
    Initializes the database by creating store tables if they don't exist and returns the initialized database.
    The output of this function is needed as the db argument in the add_memory function

    Returns:
        dict: The initialized database with store tables for each memory type.
    """
    db = {}

    # This will create store tables if they don't exist
    for mem_type in similarity_cutoff.keys():
        COLLECTION_NAME = f"{mem_type}_embedding"
        db[mem_type] = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embedding_model,
        )
    return db

def call_llm(instructions, prompt):
    """
    Call the LLM (Language Learning Model) API with the given instructions and prompt.

    Args:
        instructions (str): The instructions to provide to the LLM API.
        prompt (str): The prompt to provide to the LLM API.
        chat (Langchain Open AI model): Chat model used for AI judging

    Returns:
        dict or None: The response from the LLM API as a dictionary, or None if an error occurred.
    """

    try:
        messages = [
            SystemMessage(content=instructions),
            HumanMessage(content=prompt),
        ]
        response = chat(messages)
        try:
            response = json.loads(response.content)
        except Exception as e:
            print(f"Error creating json from response from the LLM {e}")
            print("Aborting further processing. GPT-3.5 can be silly, just try again usally works")
            sys.exit()
        return response
    except Exception as e:
        print(f"Error calling LLM {e}")
        print("Aborting further processing")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage of the function:
    # generate_and_save_images("A cute baby sea otter")
    pass
