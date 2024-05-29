##### Begin of generate_images #####

import json
import os
import uuid
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import requests
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from openai import OpenAI

load_dotenv()

# Lower numbers are more similar
similarity_cutoff = {"memory": 0.2, "recipe": 0.3, "helper_function": 0.1}

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


# Example usage of the function:
# generate_and_save_images("A cute baby sea otter")


#### End of generate_images ####


##### Begin of query_data_db #####


## This is a skill to execute database queires in the data databse,
## For answering questions about humanitarian response.


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


def add_memory(intent, metadata, db, mem_type="recipe", force=False):
    """
    Add a data recipe to the data recipe db.

    Parameters:
    - intent (str): The content of the memory document.
    - metadata (dict): Additional metadata for the memory document.
    - mem_type (str): The type of memory store to add the document to.
    - db (Database): The database object representing the memory store. This is created by the initialize_db function.
    - force (bool, optional): If True, force the addition of the memory document even if a similar document already exists. Default is False.

    Returns:
    - id (str): The ID of the added memory document.
    """
    print(f"Adding new document to {mem_type} store ...")
    data = {}
    data["page_content"] = intent

    uuid_str = str(uuid.uuid4())
    metadata["custom_id"] = uuid_str

    metadata["mem_type"] = mem_type

    new_doc = Document(page_content=intent, metadata=metadata)
    id = db[mem_type].add_documents([new_doc], ids=[uuid_str])
    return id


# Stored in langchain_pg_collection and langchain_pg_embedding as this
def initialize_db():
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


if __name__ == "__main__":
    # Example usage of the function:
    # generate_and_save_images("A cute baby sea otter")
    pass
