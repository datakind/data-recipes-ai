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
    try:
        conn = create_engine(conn_str)
        return conn
    except Exception as error:
        print("--------------- Error while connecting to PostgreSQL", error)


def get_memories(force_checkout=False):
    """
    Retrieves memories from the database.

    Returns:
        pandas.DataFrame: A DataFrame containing the retrieved memories.
    """
    conn = connect_to_db()
    with conn.connect() as connection:
        if force_checkout is False:
            query = text(
                "SELECT custom_id, document, cmetadata FROM public.langchain_pg_embedding WHERE cmetadata->>'locked_at' = ''"
            )
        else:
            query = text(
                "SELECT custom_id, document, cmetadata FROM public.langchain_pg_embedding"
            )
        result = connection.execute(query)
        result = result.fetchall()
        memories = pd.DataFrame(result)
        # If the dataset is empty, stop everything and return error
        if memories.empty:
            logging.error(
                "No memories found in the database - this might be due to the fact that all memories are locked (because another recipe checker has them checked out)! You can overwrite this by using the --force_checkout flag, but this is not recommended."
            )
            sys.exit(1)
        return memories


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

        try:
            # read relevant keys from metadata
            metadata = row["cmetadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            custom_id = row["custom_id"]
            document = row["document"]
            # if the metadata contains a response_text, save it as output
            if "response_text" in metadata:
                output = metadata["response_text"]
            elif "response_image" in metadata:
                output = metadata["response_image"]
            try:
                calling_code = metadata["calling_code"]
                functions_code = metadata["functions_code"]
                # if import it not already in the functions_code, add it with a linebreak
                if imports not in functions_code:
                    functions_code = imports + "\n\n" + functions_code
                # Concatenate functions_code and calling_code into recipe code
                recipe_code = (
                    f"{functions_code}\n\n" f"# Calling code:\n{calling_code}\n\n"
                )

                # Save the recipe code
                with open(recipe_code_path, "w", encoding="utf-8") as file:
                    file.write(recipe_code)

            except KeyError:
                logging.info(f"Record '{folder_name}' doesn't contain any code!")

            # Create a dictionary with the variables
            record = {"custom_id": custom_id, "document": document, "output": output}

            # Convert the dictionary to a JSON string
            json_record = json.dumps(record, indent=4)

            # Save files
            with open(record_info_path, "w", encoding="utf-8") as file:
                file.write(json_record)

            # Save the metadata; assuming it's a JSON string, we write it directly
            with open(metadata_path, "w", encoding="utf-8") as file:
                json_data = json.dumps(metadata, indent=4)
                file.write(json_data)

        except Exception as e:
            logging.error(f"Error while saving data for row {index}: {e}")


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
    within the 'cmetadata' JSON column for all 'custom_id' values in the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the 'custom_id' values to lock.
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
    # Execute the query within a transaction context
    try:
        with conn.connect() as connection:
            with connection.begin():
                connection.execute(query)
    except Exception as e:
        print(f"Error occurred: {e}")


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

    try:
        with open(source_path, "r", encoding="utf-8") as src_file:
            data = src_file.read()
        with open(dest_path, "w", encoding="utf-8") as dest_file:
            dest_file.write(data)
    except IOError as e:
        raise IOError(f"Error copying file: {e}")


def extract_code_sections(recipe_path):
    """
    Extracts the code sections from the recipe.py file.

    Args:
        recipe_path (str): Path to the recipe.py file.

    Returns:
        dict: A dictionary with 'functions_code' and 'calling_code' as keys.

    Raises:
        FileNotFoundError: If the recipe file does not exist.
        ValueError: If the required sections are not found in the recipe file.
    """
    if not os.path.exists(recipe_path):
        raise FileNotFoundError(f"Recipe file {recipe_path} does not exist.")

    try:
        with open(recipe_path, "r", encoding="utf-8") as file:
            content = file.read()

        functions_code_match = re.search(
            r"^(.*?)(?=# Calling code:)", content, re.DOTALL
        )
        calling_code_match = re.search(r"# Calling code:\s*(.*)", content, re.DOTALL)

        if not functions_code_match or not calling_code_match:
            raise ValueError("Required sections not found in the recipe file.")

        return {
            "functions_code": functions_code_match.group(1).strip(),
            "calling_code": calling_code_match.group(1).strip(),
        }
    except IOError as e:
        raise IOError(f"Error reading recipe file: {e}")


def update_metadata_file(metadata_path, code_sections):
    """
    Updates the functions_code and calling_code fields in the metadata file.

    Args:
        metadata_path (str): Path to the metadata file.
        code_sections (dict): A dictionary with 'functions_code' and 'calling_code' as keys.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        IOError: If the file cannot be read or written.
        json.JSONDecodeError: If the metadata file is not a valid JSON.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")

    # if code sections are empty, do not update the metadata file
    if not code_sections["functions_code"] and not code_sections["calling_code"]:
        logging.info(
            f"No code sections found in the recipe file. Skipping metadata update for record {metadata_path}."
        )
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        metadata["functions_code"] = code_sections["functions_code"]
        metadata["calling_code"] = code_sections["calling_code"]

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)
    except IOError as e:
        raise IOError(f"Error reading or writing metadata file: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in metadata file: {e}")


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

    try:
        with open(new_metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        with open(new_record_path, "r", encoding="utf-8") as record_file:
            record = json.load(record_file)

        # Include the entire metadata content as a 'metadata' key in the record
        record["metadata"] = metadata

        with open(new_record_path, "w", encoding="utf-8") as record_file:
            json.dump(record, record_file, indent=4)
    except IOError as e:
        raise IOError(f"Error reading or writing file: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file: {e}")


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

    try:
        # Extract code sections from recipe.py
        code_sections = extract_code_sections(recipe_path)

        # Update the new metadata file with the extracted code sections
        update_metadata_file(new_metadata_path, code_sections)

    except (FileNotFoundError, IOError, ValueError, json.JSONDecodeError):
        logging.info(f"Record '{directory}' doesn't contain any code!")

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
        UPDATE langchain_pg_embedding
        SET document = :document,
            cmetadata = :metadata
        WHERE custom_id = :custom_id
        """
    )
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            for index, row in df.iterrows():
                try:
                    print(row)
                    metadata_json = (
                        json.dumps(row["metadata"])
                        if isinstance(row["metadata"], dict)
                        else row["metadata"]
                    )

                    params = {
                        "document": row["document"],
                        "metadata": metadata_json,
                        "custom_id": row["custom_id"],
                    }

                    conn.execute(query_template, params)
                except KeyError as ke:
                    logging.error(
                        f"Skipping record at index {index} due to missing field: {ke}"
                    )
                except Exception as e:
                    logging.error(f"Error updating record at index {index}: {e}")
            trans.commit()
    except Exception as e:
        logging.error(f"Error updating records: {e}")


def check_out(recipe_checker="Mysterious Recipe Checker", force_checkout=False):
    """
    This is the check out function that executes the check out process.
    It retrieves memories, saves data, and formats the code using black.
    """
    memories = get_memories(force_checkout=force_checkout)
    lock_records(df=memories, locker_name=recipe_checker)
    save_data(memories)
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
                try:
                    with open(new_record_path, "r", encoding="utf-8") as file:
                        record = json.load(file)
                        records.append(record)
                    # delete the subdirectory and all its contents
                    shutil.rmtree(subdir_path)
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error reading {new_record_path}: {e}")

    # Create a DataFrame from the list of records
    records_to_check_in = pd.DataFrame(records)
    # Update database
    update_database(df=records_to_check_in, approver=recipe_checker)


def main():
    """
    Main function to parse command-line arguments and call the appropriate function.
    """
    parser = argparse.ArgumentParser(
        description="Process check in and check out operations (i.e. extracting recipes and memories from the database for quality checks and edits)."
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
