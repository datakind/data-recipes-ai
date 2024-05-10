from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import pandas as pd
import logging
import json
import subprocess

logging.basicConfig(level=logging.INFO)


load_dotenv()


# ToDo: This function is taken from ingest.py. perhaps it should be moved to something like a utils.py file
def connect_to_db():
    """
    Connects to the PostgreSQL database using the environment variables for host, port, database, user, and password.

    Returns:
        sqlalchemy.engine.base.Engine: The database connection engine.
    """

    # if is_running_in_docker():
    #    print("Running in Docker ...")
    #    host = os.getenv("POSTGRES_RECIPE_HOST")
    # else:
    #    host = "localhost"
    host = "recipedb"
    port = os.getenv("POSTGRES_RECIPE_PORT")
    database = os.getenv("POSTGRES_RECIPE_DB")
    user = os.getenv("POSTGRES_RECIPE_USER")
    password = os.getenv("POSTGRES_RECIPE_PASSWORD")
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    try:
        conn = create_engine(conn_str)
        return conn
    except Exception as error:
        print("--------------- Error while connecting to PostgreSQL", error)


def get_memories():
    conn = connect_to_db()
    query = text(
        "SELECT custom_id, document, cmetadata FROM public.langchain_pg_embedding"
    )
    result = conn.execute(query)
    result = result.fetchall()
    memories = pd.DataFrame(result)
    logging.info(memories.head())
    logging.info(memories.info())
    return memories


def save_data(df):
    base_path = "./checked_out"

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Folder path for this row
        folder_path = os.path.join(base_path, str(row["custom_id"]))

        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define file paths
        document_path = os.path.join(folder_path, "document.txt")
        metadata_path = os.path.join(folder_path, "metadata.json")
        recipe_code_path = os.path.join(folder_path, "recipe.py")
        output_path = os.path.join(folder_path, "output.txt")

        try:
            # read relevant keys from metadata
            metadata = row["cmetadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            document = row["document"]
            calling_code = metadata["calling_code"]
            functions_code = metadata["functions_code"]
            # Concatenate functions_code and calling_code into recipe code
            recipe_code = (
                f"# Functions code:\n{functions_code}\n\n"
                f"# Calling code:\n{calling_code}\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
            output = metadata["response_text"]

            # Save files
            with open(document_path, "w", encoding="utf-8") as file:
                file.write(document)

            # Save the metadata; assuming it's a JSON string, we write it directly
            with open(metadata_path, "w", encoding="utf-8") as file:
                json_data = json.dumps(metadata)
                file.write(json_data)

            # Save the recipe code
            with open(recipe_code_path, "w", encoding="utf-8") as file:
                file.write(recipe_code)

            # Save the output
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(output)
        except Exception as e:
            logging.error(f"Error while saving data for row {index}: {e}")


def format_code_with_black():
    # Define the command to run black on the current directory with --force-exclude ''
    command = ["black", ".", "--force-exclude", ""]

    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)


def main():
    memories = get_memories()
    save_data(memories)
    format_code_with_black()


if __name__ == "__main__":
    main()
