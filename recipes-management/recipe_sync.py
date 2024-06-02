import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_community.vectorstores.pgvector import PGVector
from skills import add_memory, call_llm

from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Where recipes will be checkout out or created
checked_out_folder_name = "./work/checked_out"
new_recipe_folder_name = checked_out_folder_name

# Lower numbers are more similar
similarity_cutoff = {"memory": 0.2, "recipe": 0.3, "helper_function": 0.1}

environment = Environment(loader=FileSystemLoader("templates/"))

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
    host=os.environ.get("POSTGRES_RECIPE_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_RECIPE_PORT", "5432")),
    database=os.environ.get("POSTGRES_RECIPE_DB", "postgres"),
    user=os.environ.get("POSTGRES_RECIPE_USER", "postgres"),
    password=os.environ.get("POSTGRES_RECIPE_PASSWORD", "postgres"),
)

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
    # add an echo=True to see the SQL queries
    conn = create_engine(conn_str)
    return conn

def get_recipes(force_checkout=False):
    """
    Retrieves recipes from the database.

    Args:
        force_checkout (bool, optional): If True, includes locked recipes in the result. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the retrieved recipes.

    Raises:
        SystemExit: If no recipes are found in the database and force_checkout is False.
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
            query += "AND lr.locked_at = '' OR lr.locked_at IS NULL"

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

def get_folder_cksum(folder):
    """
    Get the checksum of the files in the folder.

    Args:
        folder (str): The path to the folder.

    Returns:
        str: The checksum of the files in the folder.
    """

    files = ["metadata.json", "recipe.py"]
    files = [os.path.join(folder, file) for file in files]
    result = subprocess.run(
        ["cksum"] + files, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result



def save_cksum(folder):
    """
    Save the checksum of the files in the folder.

    Args:
        folder (str): The path to the folder.

    Returns:
        None
    """

    # Calculate the checksum of the files in the folder
    result = get_folder_cksum(folder)

    # Save the checksum to a file
    with open(os.path.join(folder, "cksum.txt"), "w", encoding="utf-8") as file:
        file.write(result.stdout.replace("./", ""))

def save_data(df):
    """
    Save data from a DataFrame to files.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be saved.

    Returns:
        None
    """

    import_template = environment.get_template("imports_template.jinja2")
    imports_content = import_template.render()

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        folder_name = row["document"].replace(" ", "_").lower()[:100]
        # Folder path for this row
        folder_path = os.path.join(checked_out_folder_name, folder_name)

        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define file paths
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
        if 'from skills import *' not in function_code:
            function_code = imports_content + "\n\n" + function_code
        # Concatenate function_code and calling_code into recipe code
        recipe_code = (
            f"{function_code}\n\n" f"# Calling code:\n{calling_code}\n\n"
        )

        # Save the recipe code
        with open(recipe_code_path, "w", encoding="utf-8") as file:
            file.write(recipe_code)

        # Add intent to metadata, useful later on
        metadata["intent"] = document

        # Save the metadata; assuming it's a JSON string, we write it directly
        with open(metadata_path, "w", encoding="utf-8") as file:
            json_data = json.dumps(metadata, indent=4)
            file.write(json_data)

        # Save the checksum of the files
        save_cksum(folder_path)


def format_code_with_black():
    """
    Formats the code in the current directory using the Black code formatter.

    This function runs the Black command-line tool on the current directory
    with the `--force-exclude` option set to an empty string. The `--force-exclude`
    option is used to exclude any files or directories from being formatted by Black.

    Note: Make sure you have Black installed before running this function.

    Example usage:
        format_code_with_black()
    """
    # Define the command to run black on the current directory with --force-exclude ''
    command = ["black", ".", "--force-exclude", ""]

    # Run the command
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )


def lock_records(df, locker_name):
    """
    Locks the records in the 'public.langchain_pg_embedding' table by updating the 'cmetadata' column.

    Args:
        df (DataFrame): The DataFrame containing the records to be locked.
        locker_name (str): The name of the locker.

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
    UPDATE public.recipe
    SET locked_by = '{locker_name}', locked_at = '{current_time}'
    """

    query = text(query)
    with conn.connect() as connection:
        with connection.begin():
            connection.execute(query)

def clone_file(source_path, dest_path):
    """
    Clones a file from the source path to the destination path.

    Args:
        source_path (str): The path of the source file to be cloned.
        dest_path (str): The path where the cloned file will be saved.

    Raises:
        FileNotFoundError: If the source file does not exist.

    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file {source_path} does not exist.")

    with open(source_path, "r", encoding="utf-8") as src_file:
        data = src_file.read()
    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(data)

def extract_code_sections(recipe_path):
    """
    Extracts the function code and calling code sections from a recipe file.

    Args:
        recipe_path (str): The path to the recipe file.

    Returns:
        dict: A dictionary containing the extracted function code and calling code sections.
            The function code section is stored under the key "function_code",
            and the calling code section is stored under the key "calling_code".

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

def add_code_to_metadata(metadata_path, code_sections):
    """
    Update the metadata file with the provided code sections.

    Args:
        metadata_path (str): The path to the metadata file.
        code_sections (dict): A dictionary containing the code sections to update.

    Raises:
        FileNotFoundError: If the metadata file does not exist.

    Returns:
        None
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


def update_metadata_file(directory):
    """
    Adds updated files to the specified directory.

    Parameters:
    - directory (str): The directory where the files will be added.

    Returns:
    - None
    """
    metadata_path = os.path.join(directory, "metadata.json")
    recipe_path = os.path.join(directory, "recipe.py")

    # Extract code sections from recipe.py
    code_sections = extract_code_sections(recipe_path)

    # Update the new metadata file with the extracted code sections
    add_code_to_metadata(metadata_path, code_sections)

    # Here update metadata outputs

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    return metadata

def insert_records_in_db(df, approver):
    """
    Inserts the recipe records in the database with the provided metadata.

    Args:
        metadata (dict): The metadata to insert.

    Returns:
        None
    """
    engine = connect_to_db()

    query_template = text(
        """
        INSERT INTO
            recipe (
                uuid,
                function_code,
                description,
                openapi_json,
                datasets,
                python_packages,
                used_recipes_list,
                sample_call,
                sample_result,
                sample_result_type,
                source,
                created_by,
                updated_by,
                last_updated,
                approval_status,
                approver,
                approval_latest_update
        )
        VALUES (
            :uuid,
            :function_code,
            :description,
            :openapi_json,
            :datasets,
            :python_packages,
            :used_recipes_list,
            :sample_call,
            :sample_result,
            :sample_result_type,
            :source,
            :created_by,
            :updated_by,
            NOW(),
            'approved',
            :created_by,
            NOW()
        )
        """
    )
 
    with engine.connect() as conn:
        trans = conn.begin()
        for index, row in df.iterrows():
            metadata = row   
            response = add_memory(
                intent=metadata["intent"],
                metadata=metadata,
                mem_type=metadata["mem_type"],
            )
            if 'already_exists' in response:
                print(response)
                print("\nCannot add this recipe, a very similar one already exists. Aborting operation")
                sys.exit()

            uuid = response[0]

            # Now insert into recipe table
            params = {
                "uuid": uuid,
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
                "created_by": approver,
                "updated_by": approver,
                "intent": metadata["intent"],
            }
            conn.execute(query_template, params)
        
        print("Committing changes to the database")
        trans.commit()


def update_records_in_db(df, approver):
    """
    Updates the recipe record in the database with the provided metadata.

    Args:
        metadata (dict): The metadata to update.
        uuid (str): The UUID of the record to update.

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
            last_updated = NOW(),
            approval_status = 'approved',
            approver = :updated_by,
            approval_latest_update = NOW()
        WHERE
            uuid = :uuid
        """
    )

    with engine.connect() as conn:

        trans = conn.begin()
        for index, row in df.iterrows():
            metadata = row   
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
            conn.execute(query_template, params)

        print("Committing changes to the database")
        trans.commit()
    


def update_database(df: pd.DataFrame, approver: str):
    """
    Updates the recipe records in the database with the provided DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the recipe data to update.
        approver (str): The name of the user who is approving the update.

    Returns:
        None
    """
    engine = connect_to_db()

    # Get list of uuids in table recipe
    query = text(
        """
        SELECT
            uuid
        FROM
            recipe
        """
    )
    conn = connect_to_db()
    with conn.connect() as connection:
        result = connection.execute(query)
        result = result.fetchall()
        result = pd.DataFrame(result)
        uuids = result["uuid"].tolist()
        uuids = [str(uuid) for uuid in uuids]

    update_ids = []
    insert_ids = []
    for index, row in df.iterrows():
        uuid = str(row["uuid"])
        if uuid in uuids:  
           update_ids.append(uuid) 
        else:
           insert_ids.append(uuid)   

    if len(update_ids) > 0:
        print(f"Proceeding with update of {len(update_ids)} records in the database")
        update_df = df[df["uuid"].isin(update_ids)]
        update_records_in_db(update_df, approver)
    
    if len(insert_ids) > 0:
        print(f"Proceeding with insert of {len(insert_ids)} records in the database")
        insert_df = df[df["uuid"].isin(insert_ids)]
        insert_records_in_db(insert_df, approver)


def check_out(recipe_author="Mysterious Recipe Checker", force_checkout=False):
    """
    Checks out recipes for editing.

    Args:
        recipe_author (str): The name of the recipe checker.
        force_checkout (bool): Whether to force checkout the recipes.

    Returns:
        None
    """
    recipes = get_recipes(force_checkout=force_checkout)
    lock_records(df=recipes, locker_name=recipe_author)
    save_data(recipes)
    format_code_with_black()

def generate_openapi_from_function_code(function_code):
    """
    Generate OpenAPI JSON from function code.

    Args:
        function_code (str): The function code.

    Returns:
        dict: The OpenAPI JSON generated from the function code.
    """
 
    prompt = f"""
        Generate openapi.json JSON for the code below. 

        ```{function_code}```
    """

    _, chat = get_models()
    openapi_json = call_llm("", prompt, chat)
    openapi_json = json.dumps(openapi_json, indent=4)
    return openapi_json

# TODO this is same code as used in recipe manager action, need to refactor so there is only one instance
def get_models():
    api_key = os.getenv("RECIPES_OPENAI_API_KEY")
    base_url = os.getenv("RECIPES_BASE_URL")
    api_version = os.getenv("RECIPES_OPENAI_API_VERSION")
    api_type = os.getenv("RECIPES_OPENAI_API_TYPE")
    completion_model = os.getenv("RECIPES_OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME")

    if api_type == "openai":
        #print("Using OpenAI API in memory.py")
        embedding_model = OpenAIEmbeddings(
            api_key=api_key,
            # model=completion_model
        )
        chat = ChatOpenAI(
            # model_name="gpt-3.5-turbo",
            model_name="gpt-3.5-turbo-16k",
            api_key=api_key,
            temperature=1,
            max_tokens=1000,
        )
    elif api_type == "azure":
        #print("Using Azure OpenAI API in memory.py")
        embedding_model = AzureOpenAIEmbeddings(
            api_key=api_key,
            deployment=completion_model,
            azure_endpoint=base_url,
            chunk_size=16,
        )
        chat = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=base_url,
            model_name="gpt-35-turbo",
            # model_name="gpt-4-turbo",
            # model="gpt-3-turbo-1106", # Model = should match the deployment name you chose for your 1106-preview model deployment
            # response_format={ "type": "json_object" },
            temperature=1,
            max_tokens=1000,
        )
    else:
        print("OPENAI API type not supported")
        sys.exit(1)
    return embedding_model, chat


def generate_openapi_json(df):
    """
    Generate OpenAPI JSON from function_code.

    Args:
        df (pd.DataFrame): The DataFrame containing the recipe data.

    Returns:
        pd.DataFrame: The DataFrame with the OpenAPI JSON added.
    """
    for index, row in df.iterrows():
        function_code = row["function_code"]
        openapi_json = generate_openapi_from_function_code(function_code)
        df.at[index, "openapi_json"] = openapi_json        

    return df

def compare_cksums(folder):
    """
    Compare the old and new checksums.

    Args:
        folder (str): Folder where files and old cksum are

    Returns:
        bool: True if the checksums match, False otherwise.
    """
    # get cksums of files in the directory
    new_cksum = get_folder_cksum(folder)
    new_cksum = new_cksum.stdout

    # read the cksum.txt file into a variable
    cksum_path = os.path.join(folder, "cksum.txt")

    if os.path.exists(cksum_path):
        with open(cksum_path, "r") as file:
            old_cksum = file.read()

        new_cksum = new_cksum.replace('./', '')

        if old_cksum == new_cksum:
            #print(f"No changes detected in {folder}")
            return True
        else:
            print(f"Changes detected in {folder}")
            return False

def delete_recipe(recipe_uuid):
    """
    Delete a recipe by UUID

    Args:
        recipe_uuid (str): The UUID of the recipe to delete

    Returns:
        None
    """
    engine = connect_to_db()
    with engine.connect() as conn:
        trans = conn.begin()

        for table in ["langchain_pg_embedding", "recipe"]:

            query = f"""
                DELETE FROM
                    public.{table}
                WHERE
                    uuid = '{recipe_uuid}'
            """
            query = text(query)
            conn.execute(query)
        trans.commit()
    print(f"Recipe with UUID {recipe_uuid} deleted from the database.")

def unlock_records(recipe_author):
    """
    Clear locked records in the 'public.langchain_pg_embedding' table.

    Args:
        recipe_author (str): The name of the recipe checker.

    Returns:
        None
    """
    conn = connect_to_db()
    query = f"""
        UPDATE public.recipe
        SET locked_by = '', locked_at = ''
        WHERE locked_by = '{recipe_author}'
    """
    query = text(query)
    with conn.connect() as connection:
        with connection.begin():
            connection.execute(query)
            print(f"Locked records cleared for recipe checker {recipe_author}.")


def check_in(recipe_author="Mysterious Recipe Checker"):
    """
    Check in function to process each subdirectory in the checked_out directory.
    """

    # delete pycache if it exists
    pycache_path = os.path.join(checked_out_folder_name, "__pycache__")
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)

    if not os.path.exists(checked_out_folder_name):
        print(f"Base directory {checked_out_folder_name} does not exist.")
        return

    records = []
    cksums_to_update = []

    for subdir in os.listdir(checked_out_folder_name):
        subdir_path = os.path.join(checked_out_folder_name, subdir)
        if os.path.isdir(subdir_path):

            # Skip if the checksums match
            if compare_cksums(subdir_path):
                continue

            record = update_metadata_file(subdir_path)
            records.append(record)
            cksums_to_update.append(subdir_path)

    # Create a DataFrame from the list of records
    records_to_check_in = pd.DataFrame(records)

    # Generate openapi_json from function_code
    records_to_check_in = generate_openapi_json(records_to_check_in)

    # Update database
    if records_to_check_in.empty:
        print("Nothing changed, no records to check in.")
    else:
        update_database(df=records_to_check_in, approver=recipe_author)
        for subdir in cksums_to_update:
            save_cksum(subdir)
        unlock_records(recipe_author)

def create_new_recipe(recipe_intent, recipe_author):
    """
    Create a new recipe folder with necessary metadata and template files.

    Parameters:
    - recipe_intent (str): The name of the recipe.

    This function performs the following steps:
    1. Defines the folder name and path for the new recipe.
    2. Creates the folder if it does not already exist.
    3. Defines the metadata structure with placeholder values.
    4. Writes the metadata to a `metadata.json` file in the recipe folder.
    5. Reads an `imports.txt` file if it exists, or uses a default import template.
    6. Writes a `recipe.py` file in the recipe folder with the imports and a template for the recipe code.
    """

    # Create new folder
    recipe_folder = os.path.join(new_recipe_folder_name, recipe_intent)
    os.makedirs(recipe_folder, exist_ok=True)

    # Render jinja templates
    import_template = environment.get_template("imports_template.jinja2")
    imports_content = import_template.render()

    new_recipe_code_template = environment.get_template("new_recipe_code_template.jinja2")
    code_content = new_recipe_code_template.render(
        imports=imports_content,
        recipe_intent=recipe_intent
    )
    new_recipe_metadata_template = environment.get_template("new_recipe_metadata_template.jinja2")
    metadata_content = new_recipe_metadata_template.render(
        uuid= uuid4(),
        recipe_intent=recipe_intent,
        recipe_author=recipe_author
    )

    # Write content to recipe.py file
    recipe_path = os.path.join(recipe_folder, "recipe.py")
    with open(recipe_path, "w", encoding="utf-8") as recipe_file:
        recipe_file.write(code_content)
    metadata_path = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        metadata_file.write(metadata_content)

    # Save an empty cksum file
    with open(os.path.join(recipe_folder, "cksum.txt"), "w", encoding="utf-8") as file:
        file.write("")

def main():
    """
    Main function to parse command-line arguments and call the appropriate function.

    This function parses the command-line arguments using the `argparse` module and calls the appropriate function based on the provided arguments. It supports two operations: check out and check in.

    Usage:
        python recipe_sync.py --check_out <recipe_author> [--force_checkout]
        python recipe_sync.py --check_in <recipe_author>
        python recipe_sync.py --create_recipe

    Arguments:
        --check_out --recipe_author <recipe author>: Perform check out operation.
        --check_in --recipe_author <recipe author>: Perform check in operation.
        --force_checkout: Force check out operation
        --create_recipe <recipe_intent>: Create a new blank recipe

    <recipe author>: The name of the recipe author, used for locking recipes for editing.

    Example:
        python recipe_sync.py --check_out my_recipe_author --force_checkout

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
    group.add_argument(
        "--create_recipe", action="store_true", help="Create a new blank recipe"
    )
    group.add_argument(
        "--delete_recipe", action="store_true", help="Delete a recipe by UUID"
    )

    parser.add_argument("--recipe_author", type=str, help="Name of the recipe checker")
    parser.add_argument("--recipe_intent", type=str, help="Intent of the new recipe")
    parser.add_argument("--recipe_uuid", type=str, help="UUID of recipe")

    # Add force_checkout argument
    parser.add_argument(
        "--force_checkout", action="store_true", help="Force check out operation"
    )

    args = parser.parse_args()

    if (args.check_out or args.check_in or args.create_recipe) and not args.recipe_author:
        parser.error("--recipe_author is required for this action")

    if args.check_out:
        check_out(args.recipe_author, force_checkout=args.force_checkout)
    elif args.check_in:
        check_in(args.recipe_author)
        # Check out to refresh metadata file. TODO, do this as part fo check in
        check_out(args.recipe_author, force_checkout=True)
    elif args.create_recipe:
        recipe_intent = args.recipe_intent.lower().replace(" ","_")
        check_out(args.recipe_author, force_checkout=True)
        create_new_recipe(recipe_intent, args.recipe_author)
    elif args.delete_recipe:
        delete_recipe(args.recipe_uuid)



if __name__ == "__main__":
    main()
