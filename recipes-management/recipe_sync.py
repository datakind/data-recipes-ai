import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)

load_dotenv()

# read the imports.txt file into a variable incl. linebreaks
with open("imports.txt", "r") as file:
    imports = file.read()


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
    conn = create_engine(conn_str)
    return conn

def get_recipes(force_checkout=False):
    """
    Retrieves recipes from the database.

    Returns:
        pandas.DataFrame: A DataFrame containing the retrieved recipes.
    """
    conn = connect_to_db()
    with conn.connect() as connection:
        query = """
            SELECT 
                lc.uuid, 
                lc.document,
                row_to_json(lr.*) as cmetadata
            FROM
                public.langchain_pg_embedding lc,
                public.recipe lr
            WHERE 
                lc.uuid=lr.uuid
        """

        if force_checkout is False:
            query += "AND cmetadata->>'locked_at' = ''"

        query = text(query)
        result = connection.execute(query)
        result = result.fetchall()
        recipes = pd.DataFrame(result)
        # If the dataset is empty, stop everything and return error
        if recipes.empty:
            logging.error(
                "No recipes found in the database - this might be due to the fact that all recipes are locked (because another recipe checker has them checked out)! You can overwrite this by using the --force_checkout flag, but this is not recommended."
            )
            sys.exit(1)
        return recipes


def save_data(df):
    """
    Save data from a DataFrame to the file system.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be saved.

    Raises:
        Exception: If there is an error while saving the data.

    Returns:
        None
    """
    base_path = "./checked_out"

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        folder_name = row["document"].replace(" ", "_").lower()[:100]
        # Folder path for this row
        folder_path = os.path.join(base_path, folder_name)

        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define file paths
        record_info_path = os.path.join(folder_path, "record_info.json")
        metadata_path = os.path.join(folder_path, "metadata.json")
        recipe_code_path = os.path.join(folder_path, "recipe.py")

        # read relevant keys from metadata
        metadata = row["cmetadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        uuid= row["uuid"]
        document = row["document"]
        output = metadata["sample_result"]
        calling_code = metadata["sample_call"]
        function_code = metadata["function_code"]
        # if import it not already in the function_code, add it with a linebreak
        if imports not in function_code:
            function_code = imports + "\n\n" + function_code
        # Concatenate function_code and calling_code into recipe code
        recipe_code = (
            f"{function_code}\n\n" f"# Calling code:\n{calling_code}\n\n"
        )

        # Save the recipe code
        with open(recipe_code_path, "w", encoding="utf-8") as file:
            file.write(recipe_code)

        # Create a dictionary with the variables
        record = {"uuid": str(uuid), "document": document, "output": output}

        # Convert the dictionary to a JSON string
        json_record = json.dumps(record, indent=4)

        # Save files
        with open(record_info_path, "w", encoding="utf-8") as file:
            file.write(json_record)

        # Save the metadata; assuming it's a JSON string, we write it directly
        with open(metadata_path, "w", encoding="utf-8") as file:
            json_data = json.dumps(metadata, indent=4)
            file.write(json_data)


def format_code_with_black():
    """
    Formats the code in the current directory using the Black code formatter.

    This function runs the Black command-line tool on the current directory to automatically format the code.
    It uses the `subprocess` module to execute the Black command and captures the output.

    Returns:
        None

    Raises:
        None
    """
    # Define the command to run black on the current directory with --force-exclude ''
    command = ["black", ".", "--force-exclude", ""]

    # Run the command
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )


def lock_records(df, locker_name):
    """
    Locks records in the database by setting the 'locked_by' and 'locked_at' fields
    within the 'cmetadata' JSON column for all 'uuid' values in the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the 'uuid' values to lock.
        locker_name (str): Name of the user locking the records.

    Returns:
        None
    """
    # Connect to the database
    conn = connect_to_db()

    # Get the current timestamp
    current_time = (
        datetime.now().isoformat()
    )  # Use 'now' to get the current timestamp in SQL

    # Prepare the SQL query
    query = f"""
    UPDATE public.langchain_pg_embedding
    SET cmetadata = jsonb_set(
        jsonb_set(
            cmetadata::jsonb,
            '{{locked_by}}',
            '\"{locker_name}\"'
        ),
        '{{locked_at}}',
        '\"{current_time}\"'
    );
    """

    query = text(query)
    with conn.connect() as connection:
        with connection.begin():
            connection.execute(query)

def clone_file(source_path, dest_path):
    """
    Clones a JSON file from source_path to dest_path.

    Args:
        source_path (str): Path to the source JSON file.
        dest_path (str): Path to the destination JSON file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        IOError: If the file cannot be read or written.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file {source_path} does not exist.")

    with open(source_path, "r", encoding="utf-8") as src_file:
        data = src_file.read()
    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(data)

def extract_code_sections(recipe_path):
    """
    Extracts the code sections from the recipe.py file.

    Args:
        recipe_path (str): Path to the recipe.py file.

    Returns:
        dict: A dictionary with 'function_code' and 'calling_code' as keys.

    Raises:
        FileNotFoundError: If the recipe file does not exist.
        ValueError: If the required sections are not found in the recipe file.
    """
    if not os.path.exists(recipe_path):
        raise FileNotFoundError(f"Recipe file {recipe_path} does not exist.")

    with open(recipe_path, "r", encoding="utf-8") as file:
        content = file.read()

    function_code_match = re.search(
        r"^(.*?)(?=# Calling code:)", content, re.DOTALL
    )
    calling_code_match = re.search(r"# Calling code:\s*(.*)", content, re.DOTALL)

    if not function_code_match or not calling_code_match:
        raise ValueError("Required sections not found in the recipe file.")

    return {
        "function_code": function_code_match.group(1).strip(),
        "calling_code": calling_code_match.group(1).strip(),
    }

def update_metadata_file(metadata_path, code_sections):
    """
    Updates the function_code and calling_code fields in the metadata file.

    Args:
        metadata_path (str): Path to the metadata file.
        code_sections (dict): A dictionary with 'function_code' and 'calling_code' as keys.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        IOError: If the file cannot be read or written.
        json.JSONDecodeError: If the metadata file is not a valid JSON.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")

    # if code sections are empty, do not update the metadata file
    if not code_sections["function_code"] and not code_sections["calling_code"]:
        logging.info(
            f"No code sections found in the recipe file. Skipping metadata update for record {metadata_path}."
        )
        return
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metadata["function_code"] = code_sections["function_code"]
    metadata["calling_code"] = code_sections["calling_code"]

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def merge_metadata_with_record(new_metadata_path, new_record_path):
    """
    Merges the metadata content into the record content as a 'metadata' key.

    Args:
        new_metadata_path (str): Path to the new metadata file.
        new_record_path (str): Path to the new record file.

    Raises:
        FileNotFoundError: If the new metadata file or new record file does not exist.
        IOError: If the file cannot be read or written.
        json.JSONDecodeError: If the metadata or record file is not a valid JSON.
    """
    if not os.path.exists(new_metadata_path):
        raise FileNotFoundError(
            f"New metadata file {new_metadata_path} does not exist."
        )
    if not os.path.exists(new_record_path):
        raise FileNotFoundError(f"New record file {new_record_path} does not exist.")

    with open(new_metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    with open(new_record_path, "r", encoding="utf-8") as record_file:
        record = json.load(record_file)

    # Include the entire metadata content as a 'metadata' key in the record
    record["metadata"] = metadata

    with open(new_record_path, "w", encoding="utf-8") as record_file:
        json.dump(record, record_file, indent=4)

def add_updated_files(directory):
    """
    Process a single directory to clone and update the metadata file.

    Args:
        directory (str): Path to the directory to process.
    """
    source_metadata_path = os.path.join(directory, "metadata.json")
    new_metadata_path = os.path.join(directory, "metadata_new.json")
    recipe_path = os.path.join(directory, "recipe.py")
    source_record_path = os.path.join(directory, "record_info.json")
    new_record_path = os.path.join(directory, "record_info_new.json")

    # Clone the metadata file
    clone_file(source_metadata_path, new_metadata_path)

    # Clone the record file
    clone_file(source_record_path, new_record_path)

    # Extract code sections from recipe.py
    code_sections = extract_code_sections(recipe_path)

    # Update the new metadata file with the extracted code sections
    update_metadata_file(new_metadata_path, code_sections)

    # Merge the updated metadata into the new record file
    merge_metadata_with_record(new_metadata_path, new_record_path)


def update_database(df: pd.DataFrame, approver: str):
    """
    Updates the database with the values from the DataFrame and sets additional columns.

    Args:
        df (pandas.DataFrame): DataFrame containing the values to update.
        approver (str): Name of the approver.

    Returns:
        None
    """
    engine = connect_to_db()

    query_template = text(
        """
        UPDATE 
            recipe
        SET
            function_code = :function_code,
            description = :description,
            openapi_json = :openapi_json,
            datasets = :datasets,
            python_packages = :python_packages,
            used_recipes_list = :used_recipes_list,
            sample_call = :sample_call,
            sample_result = :sample_result,
            sample_result_type = :sample_result_type,
            source = :source,
            updated_by = :updated_by,
            last_updated = NOW()
        WHERE
            uuid = :uuid
        """
    )
    with engine.connect() as conn:
        trans = conn.begin()
        for index, row in df.iterrows():
            metadata = row["metadata"]

            params = {
                "function_code": metadata["function_code"],
                "description": metadata["description"],
                "openapi_json": str(metadata["openapi_json"]),
                "datasets": metadata["datasets"],
                "python_packages": metadata["python_packages"],
                "used_recipes_list": metadata["used_recipes_list"],
                "sample_call": metadata["sample_call"],
                "sample_result": metadata["sample_result"],
                "sample_result_type": metadata["sample_result_type"],
                "source": metadata["source"],
                "updated_by": approver,
                "uuid": row["uuid"],
            }
            print(params)

            # TO DO Might need to update document on embedding table too if intent changed

            conn.execute(query_template, params)
        print("Committing changes to the database")
        trans.commit()


def check_out(recipe_checker="Mysterious Recipe Checker", force_checkout=False):
    """
    This is the check out function that executes the check out process.
    It retrieves recipes, saves data, and formats the code using black.
    """
    recipes = get_recipes(force_checkout=force_checkout)
    lock_records(df=recipes, locker_name=recipe_checker)
    save_data(recipes)
    format_code_with_black()


def check_in(recipe_checker="Mysterious Recipe Checker"):
    """
    Check in function to process each subdirectory in the checked_out directory.
    """
    base_directory = "checked_out"

    # delete pycache if it exists
    pycache_path = os.path.join(base_directory, "__pycache__")
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)

    if not os.path.exists(base_directory):
        print(f"Base directory {base_directory} does not exist.")
        return

    records = []

    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            add_updated_files(subdir_path)
            new_record_path = os.path.join(subdir_path, "record_info_new.json")

            if os.path.exists(new_record_path):
                with open(new_record_path, "r", encoding="utf-8") as file:
                    record = json.load(file)
                    records.append(record)
                # delete the subdirectory and all its contents
                shutil.rmtree(subdir_path)

    # Create a DataFrame from the list of records
    records_to_check_in = pd.DataFrame(records)

    # Update database
    update_database(df=records_to_check_in, approver=recipe_checker)


def main():
    """
    Main function to parse command-line arguments and call the appropriate function.
    """
    parser = argparse.ArgumentParser(
        description="Process check in and check out operations (i.e. extracting recipes and recipes from the database for quality checks and edits)."
    )

    # Add mutually exclusive group for checkout and checkin
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check_out", action="store_true", help="Perform check out operation"
    )
    group.add_argument(
        "--check_in", action="store_true", help="Perform check in operation"
    )

    # Add recipe_checker argument
    parser.add_argument("recipe_checker", type=str, help="Name of the recipe checker")

    # Add force_checkout argument
    parser.add_argument(
        "--force_checkout", action="store_true", help="Force check out operation"
    )

    args = parser.parse_args()

    if args.check_out:
        check_out(args.recipe_checker, force_checkout=args.force_checkout)
    elif args.check_in:
        check_in(args.recipe_checker)


if __name__ == "__main__":
    main()
