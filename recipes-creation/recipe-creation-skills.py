import os
import psycopg2
import json
import uuid
from langchain_openai import OpenAIEmbeddings

def create_vector_from_string(input_string):
    model = OpenAIEmbeddings(model="text-embedding-ada-002",
                                        chunk_size=16)

    # Generate the embedding vector from the input string
    vector = model.embed_query(input_string)
    return vector

def get_recipe_db_connection():
    """
    This function gets a connection to the database
    """
    host = os.getenv("POSTGRES_RECIPE_HOST")
    port = os.getenv("POSTGRES_RECIPE_PORT")
    database = os.getenv("POSTGRES_RECIPE_DB")
    user = os.getenv("POSTGRES_RECIPE_USER")
    password = os.getenv("POSTGRES_RECIPE_PASSWORD")

    conn = psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn

def form_recipe_record():
    #load the data_recipe.json file
    with open('data_recipe.json') as json_file:
        cmetadata = json.load(json_file)
    #form the recipe record
    recipe_record = { 
        "embedding": create_vector_from_string(cmetadata['intent']),
        "uuid": uuid.uuid4(cmetadata['intent']),
        "document": cmetadata['intent'],
        "custom_id": uuid.uuid4(cmetadata['intent']),
        "cmetadata": cmetadata,
        "approval_status": "pending"
    }
    return recipe_record

def insert_recipe_record(recipe_record):
    """
    This function inserts a recipe record into the database
    """
    conn = get_recipe_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO langchain_pg_embedding (uuid, document, custom_id, cmetadata, approval_status)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            recipe_record["uuid"],
            recipe_record["embedding"],
            recipe_record["document"],
            recipe_record["custom_id"],
            json.dumps(recipe_record["cmetadata"]),
            recipe_record["approval_status"]
        )
    )
    conn.commit()
    conn.close()


