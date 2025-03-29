import argparse
import base64
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from uuid import UUID, uuid4

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import create_engine, text

from utils.db import connect_to_db
from utils.llm import call_llm
from utils.recipes import add_recipe_memory

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Where recipes will be checkout out or created
checked_out_folder_name = "./work/checked_out"
new_recipe_folder_name = checked_out_folder_name

# String separating sample calling code and recipe functions
code_separator = 'if __name__ == "__main__":'

# Lower numbers are more similar
similarity_cutoff = {"memory": 0.2, "recipe": 0.3, "helper_function": 0.1}

# Fields an intent must have to be useful
required_intent_fields = [
    "action",
    "visualization_type",
    "output_format",
    "data_types",
    "filters",
    "data_sources",
]

# String to indicate start of output
output_start_string = "OUTPUT:"

# Non-optional Fields that must exist in recipe json output
required_output_json_fields = ["result"]

environment = Environment(loader=FileSystemLoader("templates/"))


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
    conn = connect_to_db(instance="recipe")
    with conn.connect() as connection:
        query = """
            SELECT
                lc.custom_id,
                lc.document,
                row_to_json(lr.*) as cmetadata
            FROM
                public.langchain_pg_embedding lc,
                public.recipe lr
            WHERE
                lc.custom_id=lr.custom_id
            ORDER BY
                lr.created_by
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
            # sys.exit(1)
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
        folder_name = row["document"].replace(" ", "_").lower()
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
        document = row["document"]
        calling_code = metadata["sample_call"]
        function_code = metadata["function_code"]
        # if import it not already in the function_code, add it with a linebreak
        if "from skills import *" not in function_code:
            function_code = imports_content + "\n\n" + function_code
        # Concatenate function_code and calling_code into recipe code
        recipe_code = f"{function_code}\n\n" f"{code_separator}\n{calling_code}\n\n"

        # Make max number blank lines in a row as 3
        recipe_code = re.sub(r"\n{4,}", "\n\n\n", recipe_code)

        # Save the recipe code
        with open(recipe_code_path, "w", encoding="utf-8") as file:
            file.write(recipe_code)

        # Add intent to metadata, useful later on
        metadata["intent"] = document

        # Update metadata file
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)

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
    conn = connect_to_db(instance="recipe")

    # Get the current timestamp
    current_time = (
        datetime.datetime.now().isoformat()
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

    if "__name__" not in content:
        raise ValueError(
            f"Code separator '{code_separator}' not found in the recipe file '{recipe_path}'."
        )

    content = content.split("\n")

    # Find the line containing '__name__'
    split_index = next(
        (i for i, line in enumerate(content) if "__name__" in line), None
    )

    # If the line is found, split the content into two parts
    if split_index is not None:
        before_name = content[:split_index]
        after_name = content[split_index + 1 :]
    else:
        before_name = content
        after_name = []

    # Convert lists back to strings
    function_code = "\n".join(before_name)
    calling_code = "\n".join(after_name)

    if function_code is None or calling_code is None:
        raise ValueError(
            f"Function code or calling code not found in the recipe file '{recipe_path}'."
        )

    return {
        "function_code": function_code,
        "calling_code": calling_code,
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
    metadata["sample_call"] = code_sections["calling_code"]

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
    engine = connect_to_db(instance="recipe")

    query_template = text(
        """
        INSERT INTO
            recipe (
                custom_id,
                function_code,
                description,
                openapi_json,
                datasets,
                python_packages,
                used_recipes_list,
                sample_call,
                sample_result,
                sample_result_type,
                sample_metadata,
                source,
                created_by,
                updated_by,
                last_updated,
                approval_status,
                approver,
                approval_latest_update
        )
        VALUES (
            :custom_id,
            :function_code,
            :description,
            :openapi_json,
            :datasets,
            :python_packages,
            :used_recipes_list,
            :sample_call,
            :sample_result,
            :sample_result_type,
            :sample_metadata,
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
            response = add_recipe_memory(
                intent=metadata["intent"],
                metadata={"mem_type": "recipe"},
                mem_type="recipe",
                force=True,
            )
            if "already_exists" in response:
                print(response)
                print(
                    "\nCannot add this recipe, a very similar one already exists. Aborting operation"
                )
                continue

            custom_id = response[0]

            metadata["sample_result"] = json.dumps(metadata["sample_result"])
            metadata["sample_metadata"] = json.dumps(metadata["sample_metadata"])

            # Now insert into recipe table
            params = {
                "custom_id": custom_id,
                "function_code": metadata["function_code"],
                "description": metadata["description"],
                "openapi_json": str(metadata["openapi_json"]),
                "datasets": metadata["datasets"],
                "python_packages": metadata["python_packages"],
                "used_recipes_list": metadata["used_recipes_list"],
                "sample_call": metadata["sample_call"],
                "sample_result": str(metadata["sample_result"]),
                "sample_result_type": metadata["sample_result_type"],
                "sample_metadata": metadata["sample_metadata"],
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
        custom_id (str): The custom_id of the record to update.

    Returns:
        None
    """
    engine = connect_to_db(instance="recipe")

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
            sample_metadata = :sample_metadata,
            source = :source,
            updated_by = :updated_by,
            last_updated = NOW(),
            approval_status = 'approved',
            approver = :updated_by,
            approval_latest_update = NOW()
        WHERE
            custom_id = :custom_id
        """
    )

    with engine.connect() as conn:

        trans = conn.begin()
        for index, row in df.iterrows():
            metadata = row
            metadata["sample_result"] = json.dumps(metadata["sample_result"])
            metadata["sample_metadata"] = json.dumps(metadata["sample_metadata"])
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
                "sample_metadata": metadata["sample_metadata"],
                "source": metadata["source"],
                "updated_by": approver,
                "custom_id": row["custom_id"],
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

    # Get list of custom_ids in table recipe
    query = text(
        """
        SELECT
            custom_id
        FROM
            recipe
        """
    )
    conn = connect_to_db(instance="recipe")
    with conn.connect() as connection:
        result = connection.execute(query)
        result = result.fetchall()
        result = pd.DataFrame(result)
        if not result.empty:
            existing_custom_ids = result["custom_id"].tolist()
            existing_custom_ids = [str(custom_id) for custom_id in existing_custom_ids]
        else:
            existing_custom_ids = []

    incoming_custom_ids = df["custom_id"].tolist()

    update_ids = []
    insert_ids = []
    for custom_id in incoming_custom_ids:
        if custom_id in existing_custom_ids:
            update_ids.append(custom_id)
        else:
            insert_ids.append(custom_id)

    if len(update_ids) > 0:
        print(f"Proceeding with update of {len(update_ids)} records in the database")
        update_df = df[df["custom_id"].isin(update_ids)]
        update_records_in_db(update_df, approver)

    if len(insert_ids) > 0:
        print(f"Proceeding with insert of {len(insert_ids)} records in the database")
        insert_df = df[df["custom_id"].isin(insert_ids)]
        insert_records_in_db(insert_df, approver)


def check_out(recipe_author, force_checkout=False):
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

    openapi_json = call_llm("", prompt)
    openapi_json = json.dumps(openapi_json, indent=4)
    return openapi_json


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

        new_cksum = new_cksum.replace("./", "")

        if old_cksum == new_cksum:
            # print(f"No changes detected in {folder}")
            return True
        else:
            print(f"Changes detected in {folder}")
            return False


def delete_recipe(recipe_custom_id):
    """
    Delete a recipe by custom_id

    Args:
        recipe_custom_id (str): The custom_id of the recipe to delete

    Returns:
        None
    """
    engine = connect_to_db(instance="recipe")
    with engine.connect() as conn:
        trans = conn.begin()

        for table in ["langchain_pg_embedding", "recipe"]:

            query = f"""
                DELETE FROM
                    public.{table}
                WHERE
                    custom_id = '{recipe_custom_id}'
            """
            query = text(query)
            conn.execute(query)
        trans.commit()
    print(f"Recipe with custom_id {recipe_custom_id} deleted from the database.")


def unlock_records(recipe_author):
    """
    Clear locked records in the 'public.langchain_pg_embedding' table.

    Args:
        recipe_author (str): The name of the recipe checker.

    Returns:
        None
    """
    conn = connect_to_db(instance="recipe")
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


def check_in(recipe_author="Mysterious Recipe Checker", force=False):
    """
    Checks in the modified recipe files to the repository.

    Args:
        recipe_author (str, optional): The author of the recipe. Defaults to "Mysterious Recipe Checker".
        force (bool, optional): If True, forces the check-in even if the checksums match. Defaults to False.
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
            if force is not True:
                if compare_cksums(subdir_path):
                    continue

            record = update_metadata_file(subdir_path)
            records.append(record)
            cksums_to_update.append(subdir_path)

    # Create a DataFrame from the list of records
    records_to_check_in = pd.DataFrame(records)

    # Generate openapi_json from function_code. TODO, commenting out for now
    # records_to_check_in = generate_openapi_json(records_to_check_in)
    records_to_check_in["openapi_json"] = "{}"

    # Update database
    if records_to_check_in.empty:
        print("Nothing changed, no records to check in.")
    else:
        update_database(df=records_to_check_in, approver=recipe_author)
        for subdir in cksums_to_update:
            save_cksum(subdir)
        unlock_records(recipe_author)


def get_data_info():
    """
    Get data info from the database.

    Returns:
        str: The data info.
    """
    db = connect_to_db(instance="data")

    # run this query: select table_name, summary, columns from table_metadata

    query = text(
        """
        SELECT
            table_name,
            summary,
            columns
        FROM
            table_metadata
        --WHERE
        --    countries is not null
        """
    )

    with db.connect() as connection:
        result = connection.execute(query)
        result = result.fetchall()
        result = pd.DataFrame(result)
        data_info = result.to_json(orient="records")

    data_info = json.dumps(json.loads(data_info), indent=4)

    return data_info


def llm_generate_new_recipe_code(recipe_intent, imports_content):
    """
    Generate new recipe code using LLM.

    Args:
        recipe_intent (str): The intent of the recipe.
        imports_content (str): The content of the imports

    Returns:
        str: The generated recipe code.
    """

    data_info = get_data_info()

    coding_standards_template = environment.get_template("coding_standards.jinja2")
    coding_standards = coding_standards_template.render()

    new_recipe_code_template = environment.get_template("new_recipe_code_prompt.jinja2")
    prompt = new_recipe_code_template.render(
        imports=imports_content,
        recipe_intent=recipe_intent,
        data_info=data_info,
        coding_standards=coding_standards,
    )

    print("Calling LLM to generate recipe starting code ...")
    response = call_llm("", prompt)

    code = response["code"]
    comment = response["message"]

    return code, comment, prompt


def generate_intent_short_form(intent_long_form):
    """
    Generate short form and check if recipe intent has all required fields.

    Will abort if required fields missing.

    Aim of this function is to try and stardize somewhat inptent layout.

    Args:
        recipe_intent (dict): The long-form intent of the recipe.

    Returns:
        intent_short_form (str): The intent in short form.


    """
    intent_template = environment.get_template("intent_short_form_prompt.jinja2")
    prompt = intent_template.render(intent_long_form=intent_long_form)

    print("Calling LLM to generate short_form intent ...")
    intent = call_llm("", prompt)
    intent = intent["content"]

    return intent


def check_for_missing_intent_entities(recipe_intent):
    """
    Check if recipe intent has all required fields.

    Will abort if required fields missing.

    Args:
        recipe_intent (dict): The long-form intent of the recipe.

    Returns:
        None

    """
    populated_fields = []
    for f in recipe_intent:
        val = recipe_intent[f]
        if f == "filters":
            has_filter = False
            for f2 in val:
                if f2["field"] != "":
                    has_filter = True
                    break
            if has_filter is False:
                val = ""
        if f in ["data_sources", "data_types"]:
            if str(val) == "['']":
                val = ""

        if val != "":
            populated_fields.append(f)

    for f in required_intent_fields:
        if f not in populated_fields:
            print(recipe_intent)
            raise ValueError(
                f"\n\n     !!!!!!Required intent field {f} is empty, be more specific in your intent"
            )


def generate_intent_long_format(user_input):
    """
    Generate an intent from the user's input in standard intent format.

    Args:
        user_input (str): The intent of the recipe.

    Returns:
        str: The generated intent in standard format.
    """

    intent_template = environment.get_template("intent_long_form_prompt.jinja2")
    prompt = intent_template.render(user_input=user_input)

    print("Calling LLM to generate long form intent ...")
    recipe_intent = call_llm("", prompt)

    # Skipping entity check
    if "/nochecks" in user_input:
        return recipe_intent
    else:
        check_for_missing_intent_entities(recipe_intent)

    return recipe_intent


def create_new_recipe(recipe_intent, recipe_author):

    # Render jinja templates
    import_template = environment.get_template("imports_template.jinja2")
    imports_content = import_template.render()

    # Generate an intent from the user's input in standard intent format
    intent_long_format = generate_intent_long_format(recipe_intent)

    recipe_intent = generate_intent_short_form(intent_long_format)

    # Create new folder
    recipe_folder = os.path.join(
        new_recipe_folder_name,
        recipe_intent.replace(" ", "_").lower().replace("(", "").replace(")", ""),
    )

    # Use a fixed template for the recipe code
    # new_recipe_code_template = environment.get_template("new_recipe_code_template.jinja2")
    # code_content = new_recipe_code_template.render(
    #    imports=imports_content,
    #    recipe_intent=recipe_intent
    # )

    # Generate recipe code using LLM single-shot. Later this can go to AI team
    code_content, comment, prompt = llm_generate_new_recipe_code(
        recipe_intent, imports_content
    )

    print(f"LLM generated code with this comment: {comment}")

    new_recipe_metadata_template = environment.get_template(
        "new_recipe_metadata_template.jinja2"
    )
    metadata_content = new_recipe_metadata_template.render(
        custom_id=uuid4(),
        recipe_intent=recipe_intent,
        recipe_author=recipe_author,
        intent_long_format=intent_long_format,
    )

    os.makedirs(recipe_folder, exist_ok=True)

    # Write content to recipe.py file
    recipe_path = os.path.join(recipe_folder, "recipe.py")
    with open(recipe_path, "w", encoding="utf-8") as recipe_file:
        recipe_file.write(code_content)
    metadata_path = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        metadata_file.write(metadata_content)

    # Save the prompt to a recipe folder
    with open(
        os.path.join(recipe_folder, "prompt_generation.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(prompt)

    print("Running recipe to capture errors for LLM ...")
    result = run_recipe(recipe_path)

    # If there was an error, call edit recipe to try and fix it one round
    if result.returncode != 0:
        print("My code didn't work, trying to fix it ...")
        llm_edit_recipe(recipe_path, result.stderr, recipe_author)

    # Save an empty cksum file
    with open(os.path.join(recipe_folder, "cksum.txt"), "w", encoding="utf-8") as file:
        file.write("")


def llm_edit_recipe(recipe_path, llm_prompt, recipe_author):

    recipe_folder = os.path.dirname(recipe_path)

    with open(recipe_path, "r") as file:
        recipe_code = file.read()

    # Automatically run recipe to get errors and output
    print("Running recipe to capture errors for LLM ...")
    result = run_recipe(recipe_path)
    stderr_output = result.stderr
    stdout_output = result.stdout

    data_info = get_data_info()

    edit_recipe_code_template = environment.get_template(
        "edit_recipe_code_prompt.jinja2"
    )

    coding_standards_template = environment.get_template("coding_standards.jinja2")
    coding_standards = coding_standards_template.render()

    prompt = edit_recipe_code_template.render(
        llm_prompt=llm_prompt,
        recipe_code=recipe_code,
        stderr_output=stderr_output,
        stdout_output=stdout_output,
        data_info=data_info,
        coding_standards=coding_standards,
    )
    with open(
        os.path.join(recipe_folder, "prompt_modify.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(prompt)

    print("Calling LLM to generate recipe code ...")
    response = call_llm("", prompt)
    code = response["code"]
    comment = response["message"]

    print(f"\n\nLLM Gave this comment when generating code: \n\n{comment}\n\n")

    # Copy recip.py to recipe.bak.py
    recipe_bak_path = os.path.join(recipe_folder, "recipe.bak.py")
    clone_file(recipe_path, recipe_bak_path)

    # Write content to recipe.py file
    with open(recipe_path, "w") as recipe_file:
        recipe_file.write(code)

    # Update metadata file
    metadata_path = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    metadata["updated_by"] = recipe_author
    metadata["last_updated"] = datetime.datetime.now().isoformat()

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print("Running recipe with the new code ...")
    result = run_recipe(recipe_path)
    stderr_output = result.stderr
    stdout_output = result.stdout

    print("\n\nRecipe editing done")


def llm_validate_recipe(user_input, recipe_path):

    recipe_folder = os.path.dirname(recipe_path)

    with open(recipe_path, "r") as file:
        recipe_code = file.read()

    metadata_path = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    result_type = metadata["sample_result_type"]
    result = metadata["sample_result"]

    validation_prompt = environment.get_template("validate_recipe_prompt.jinja2")
    prompt = validation_prompt.render(
        user_input=user_input, recipe_code=recipe_code, recipe_result=result
    )

    if len(prompt.split(" ")) > 8000:
        return {
            "answer": "error",
            "user_input": user_input,
            "reason": "Prompt too long, please shorten recipe code or result",
        }

    if result_type == "image":
        llm_result = call_llm("", prompt, image=result)
    else:
        llm_result = call_llm("", prompt)

    print(llm_result)
    return llm_result


def update_metadata_file_results(recipe_folder, output):
    """
    Update the metadata file for a given recipe folder with the provided result.

    Args:
        recipe_folder (str): The path to the recipe folder.
        output JSON: The result of the recipe call.

    Returns:
        None
    """
    metadata_path = os.path.join(recipe_folder, "metadata.json")
    print(f"Updating metadata file with recipe sample call results: {metadata_path}")

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    print(output)

    if output["result"]["type"] == "image":

        png_file = output["result"]["file"]

        # does png exist?
        if not os.path.exists(png_file):

            # Is it in working directory?
            if os.path.exists(os.path.join("./work", png_file)):
                png_file = os.path.join("./work", png_file)
            else:
                print(f"PNG file {png_file} does not exist, skipping metadata update")
                return

        # Move png file to recipe folder
        png_file_basename = os.path.basename(png_file)
        png_file_path = os.path.join(recipe_folder, png_file_basename)
        print(f"Moving {png_file} to {png_file_path}")
        shutil.move(png_file, png_file_path)

        with open(png_file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            metadata["sample_result"] = encoded_string
            metadata["sample_result_type"] = "image"

            # Valid with GPT-4o
            if os.getenv("RECIPES_MODEL") == "gpt-4o":
                image_validation_prompt = environment.get_template(
                    "image_validation_prompt.jinja2"
                )
                prompt = image_validation_prompt.render(user_input=metadata["intent"])
                llm_result = call_llm("", prompt, image=png_file_path)
                if "answer" in llm_result:
                    if llm_result["answer"] == "yes":
                        print("Image validation passed")
                    else:
                        print(
                            "\n\n     !!!!! Image validation failed, skipping metadata update\n"
                        )
                        print(f"     {llm_result['message']}\n\n")

    else:
        metadata["sample_result"] = output
        metadata["sample_result_type"] = "text"

    # Is there metadata. TODO: Tamle call_llm to generate proper JSON for all models
    if "metadata" in output:
        metadata["sample_metadata"] = output["metadata"]
    else:
        metadata["sample_metadata"] = ""

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)


def delete_all_db_records():
    """
    Delete all records in the recipe table. We only need delete from langchain_pg_embedding,
    DB contraints will remove from ther tabels (memory/recipe).

    Returns:
        None
    """
    engine = connect_to_db(instance="recipe")
    with engine.connect() as conn:
        trans = conn.begin()
        query = """
        DELETE FROM
            public.langchain_pg_embedding
        """
        print(query)
        query = text(query)
        conn.execute(query)
        trans.commit()


def dump_db():
    """
    Dump all tables in the recipe database as inserts.

    Returns:
        None
    """

    tables_to_dump = [
        {
            "table": "langchain_pg_embedding",
            "file": "./db/3-demo-data-langchain-embedding.sql",
        },
        {"table": "recipe", "file": "./db/4-demo-data-recipes.sql"},
        {"table": "memory", "file": "./db/5-demo-data-memories.sql"},
    ]

    engine = connect_to_db(instance="recipe")
    conn = engine.connect()

    for t in tables_to_dump:
        table_name = t["table"]
        file_name = t["file"]
        query = f"SELECT * FROM public.{table_name}"
        result = conn.execute(text(query))
        result = result.fetchall()
        df = pd.DataFrame(result)

        with open(file_name, "w") as f:
            print(f"Dumping {table_name} to {file_name}")
            for index, row in df.iterrows():

                # Replace ' with '' in any cols with 'code' in their name
                for col in df.columns:
                    if "code" in col or "_call" in col or "result" in col:
                        row[col] = str(row[col]).replace("'", "''")

                cols = ", ".join([f'"{col}"' for col in df.columns])
                vals = ", ".join(
                    [
                        (
                            "'%s'" % json.dumps(val)
                            if isinstance(val, dict)
                            else (
                                "'%s'" % val
                                if isinstance(
                                    val, (str, UUID, datetime.date, datetime.datetime)
                                )
                                else ("NULL" if val is None else str(val))
                            )
                        )
                        for val in row
                    ]
                )
                f.write(f"INSERT INTO {table_name} ({cols}) VALUES ({vals});\n")


def rebuild(recipe_author):
    """
    Rebuild the recipes in the checked_out folder.

    Returns:
        None
    """

    recipes = os.listdir(checked_out_folder_name)

    if ".DS_Store" in recipes:
        recipes.remove(".DS_Store")

    delete_all_db_records()

    for r in recipes:
        recipe_folder = os.path.join(checked_out_folder_name, r)
        recipe_path = os.path.join(recipe_folder, "recipe.py")
        metadata_path = os.path.join(recipe_folder, "metadata.json")
        # Skip any recipe recipes at the end
        if "recipes" in recipe_path:
            continue
        with open(metadata_path, "r") as file:
            metadata = json.load(file)
        custom_id = metadata["custom_id"]
        print(f"Running recipe {recipe_folder} : {custom_id}")
        run_recipe(recipe_path)

    check_in(recipe_author, force=True)

    for r in recipes:
        recipe_folder = os.path.join(checked_out_folder_name, r)
        print("   Saving memory ...")
        save_as_memory(recipe_folder)

    # Now do checkout to align all ids
    # check_out(recipe_author, force_checkout=True)


def validate_output(output):
    """
    Validate output from recipe.

    Args:
        output (str): The output from the recipe.

    Returns:
        None
    """

    # Remove any lines with DEBUG in them
    output = re.sub(r"DEBUG.*\n", "", output)

    error = None

    try:
        output = json.loads(output)
        print("JSON output parsed successfully")
        # Now check for required fields
        for f in required_output_json_fields:
            if f not in output:
                error = f"Output of recipe must contain field {f}"
                print(error)
        if "type" not in output["result"]:
            error = 'Output of recipe must contain field "type" in output["result"]'
            print(error)
    except json.JSONDecodeError:
        print("Output: \n\n")
        print(output)
        error = "Output of recipe must be JSON"
        print(error)

    return error


def run_recipe(recipe_path):
    """
    Run recipe and update its metadata results

    Args:
        recipe_path (str): The path to the recipe.py file to run
        recipe_author (str): The name of the recipe checker.

    Returns:
        None
    """

    cmd = f"python {recipe_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Extract output
    if output_start_string in result.stdout:
        output = result.stdout.split(output_start_string)[1]
        # output is JSON
        error = validate_output(output)
        if error is None:
            output = json.loads(output)

            # Check for required fields
            required_output_json_fields = ["result"]
            for f in required_output_json_fields:
                if f not in output:
                    error = f"Output of recipe must contain field {f}"
                    print(error)
                    result.stderr += f"{error}"
                    result.returncode = 1
                    break

        else:
            result.stderr += f"{error}"
            result.returncode = 1
    else:
        error_str = "ERROR: Output of recipe must contain 'OUTPUT:'"
        print(error_str)
        result.stderr += f"\n\n{error_str}"
        output = {}

    recipe_folder = os.path.dirname(recipe_path)

    if result.returncode == 0 and len(output) > 0:
        update_metadata_file_results(recipe_folder, output)
    else:
        if len(result.stderr) > 0:
            print("Error running recipe, skipping metadata update")
        else:
            print("No output printed by recipe, skipping metadata update")

    return result


def generate_calling_params(functions_code, calling_code):
    """
    Generate calling parameters JSON for the recipe.

    Args:
        functions_code (str): The function code.
        calling_code (str): The calling code.

    Returns:
        dict: The calling parameters JSON for the recipe.
    """
    print("Generating calling parameters JSON for the recipe")
    prompt = f"""
        Using the following function code and sample call, generate a calling parameters JSON for the recipe.
        The JSON should have fields 'function' and 'params'

        ```{functions_code}```

        ```{calling_code}```
    """

    params = call_llm("", prompt)
    params = json.dumps(params)
    return params


def generate_memory_intent(recipe_intent, calling_code):
    """
    Generate memory intent for the recipe.

    Args:
        recipe_intent (str): The intent of the recipe.
        calling_code (str): The calling code.

    Returns:
        str: The memory intent for the recipe.
    """
    print("Generating memory intent for the recipe")
    prompt = f"""
        You are an AI agent that modifies the generic intent for a function, to make a more specific intent due to
        how the function is called.

        Examples:

        Generic intent: "plot a bar chart of humanitarian organizations by sector for a given region using Humanitarian Data Exchange data as an image"
        Calling Code: create_bar_chart_of_humanitarian_organizations_in_a_given_region_disaggregated_by_sector('Wadi Fira')

        Specific intent: "plot a bar chart of humanitarian organizations in Wadi Fira by sector using Humanitarian Data Exchange data as an image"


        Here is the generic intent:

        ```{recipe_intent}```

        Here is the calling code:

        ```{calling_code}```

        Your response must be a JSON record with the following fields:
        - intent: The specific intent

        What is the exact intent of this code?

        ```{calling_code}```

    """
    print(prompt)

    intent = call_llm("", prompt)
    intent = intent["intent"]
    print(intent)
    return intent


def get_data_info_summary(user_question=None):
    """
    Get data info from the database.

    Returns:
        str: The data info.
    """
    db = connect_to_db(instance="data")

    # run this query: select table_name, summary, columns from table_metadata

    query = text(
        """
        SELECT
            table_name,
            summary,
            columns, countries
        FROM
            table_metadata
        --WHERE
        --    countries is not null
        """
    )

    with db.connect() as connection:
        result = connection.execute(query)
        result = result.fetchall()
        result = pd.DataFrame(result)
        data_info = result.to_json(orient="records")

    data_info = json.dumps(json.loads(data_info), indent=4)

    if user_question is None:
        prompt = "Provide a summary of what datasets are available, table names"
    else:
        prompt = f"Answer this questions: {user_question}"

    prompt = f"""

        {user_question}

        DATA INFORMATION:

        ```{data_info}```
    """

    print("Calling LLM to get data info ...")
    data_summary = call_llm("", prompt)
    print(data_summary["content"])


def save_as_memory(recipe_folder):
    """
    Save a memory from recipe sample outputs

    Args:
        recipe_folder (str): The path to the recipe folder

    Returns:
        None
    """
    metadata_path = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    # Generate recipe params
    function_code = metadata["function_code"]
    sample_call = metadata["sample_call"]
    params = generate_calling_params(function_code, sample_call)

    # Generate memory intent
    recipe_intent = metadata["intent"]
    memory_intent = generate_memory_intent(recipe_intent, sample_call)

    response = add_recipe_memory(
        intent=memory_intent,
        metadata={"mem_type": "memory"},
        mem_type="memory",
        force=True,
    )
    print(response)

    if "already_exists" in response:
        print(response)
        print(
            "\nCannot add this memory, a very similar one already exists. Aborting operation"
        )
        sys.exit()

    custom_id = response[0]

    engine = connect_to_db(instance="recipe")
    with engine.connect() as conn:
        trans = conn.begin()

        query_template = text(
            """
            INSERT INTO memory (
                custom_id,
                recipe_custom_id,
                recipe_params,
                result,
                result_type,
                result_metadata,
                source,
                created_by,
                updated_by,
                last_updated
            )
            VALUES (
                :custom_id,
                :recipe_custom_id,
                :recipe_params,
                :result,
                :result_type,
                :result_metadata,
                :source,
                :created_by,
                :updated_by,
                NOW()
            )
            """
        )

        metadata["sample_result"] = json.dumps(metadata["sample_result"])
        metadata["sample_metadata"] = json.dumps(metadata["sample_metadata"])

        # Now insert into recipe table
        params = {
            "custom_id": custom_id,
            "recipe_custom_id": metadata["custom_id"],
            "recipe_params": params,
            "result": metadata["sample_result"],
            "result_type": metadata["sample_result_type"],
            "result_metadata": metadata["sample_metadata"],
            "source": "Recipe sample result",
            "created_by": metadata["created_by"],
            "updated_by": metadata["created_by"],
        }
        conn.execute(query_template, params)

        print("Committing changes to the database")
        trans.commit()


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
        --create_recipe --recipe_intent <recipe_intent>: Create a new blank recipe
        --delete_recipe --recipe_custom_id <recipe_custom_id>: Delete a recipe by custom_id
        --save_as_memory --recipe_path <recipe_path>: Save a memory from recipe sample outputs
        --rebuild --recipe_author <recipe_author>: Rebuild the recipes in the checked_out folder
        --dump_db: Dump embedding, recipe and memoty tables intoupgrade script folder as inserts
        --run_recipe --recipe_path <recipe_path>: Run recipe and update its metadata results

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
        "--delete_recipe", action="store_true", help="Delete a recipe by custom_id"
    )
    group.add_argument(
        "--run_recipe", action="store_true", help="Create a new blank recipe"
    )
    group.add_argument(
        "--save_as_memory",
        action="store_true",
        help="Create a memory from recipe sample outputs",
    )
    group.add_argument(
        "--edit_recipe", action="store_true", help="Create a new blank recipe"
    )
    group.add_argument(
        "--validate_recipe", action="store_true", help="Validate a recipe using LLM"
    )
    group.add_argument(
        "--info", action="store_true", help="Get information about the data available"
    )
    group.add_argument(
        "--rebuild",
        action="store_true",
        help="CAUTION: WIll remove database memories/recipes and push local data",
    )
    group.add_argument(
        "--dump_db",
        action="store_true",
        help="Dump embedding, recipe and memory tables to DB upgrade script folder",
    )

    parser.add_argument("--recipe_author", type=str, help="Name of the recipe checker")
    parser.add_argument("--recipe_intent", type=str, help="Intent of the new recipe")
    parser.add_argument("--recipe_custom_id", type=str, help="custom_id of recipe")
    parser.add_argument("--recipe_path", type=str, help="Path to recipe.py file to run")
    parser.add_argument("--llm_prompt", type=str, help="Prompt for the LLM")

    # Add force_checkout argument
    parser.add_argument(
        "--force_checkout", action="store_true", help="Force check out operation"
    )

    args = parser.parse_args()

    if (
        args.check_out or args.check_in or args.create_recipe
    ) and not args.recipe_author:
        parser.error("--recipe_author is required for this action")

    if args.check_out:
        check_out(args.recipe_author, force_checkout=args.force_checkout)
    elif args.check_in:
        check_in(args.recipe_author)
    elif args.create_recipe:
        recipe_intent = args.recipe_intent.replace(" ", "_").lower()
        create_new_recipe(recipe_intent, args.recipe_author)
    elif args.delete_recipe:
        delete_recipe(args.recipe_custom_id)
    elif args.run_recipe:
        run_recipe(args.recipe_path)
    elif args.save_as_memory:
        save_as_memory(args.recipe_path)
    elif args.edit_recipe:
        llm_edit_recipe(args.recipe_path, args.llm_prompt, args.recipe_author)
    elif args.validate_recipe:
        recipe_intent = args.recipe_intent
        llm_validate_recipe(recipe_intent, args.recipe_path)
    elif args.rebuild:
        rebuild(args.recipe_author)
    elif args.dump_db:
        dump_db()
    elif args.info:
        get_data_info_summary(args.recipe_author)


if __name__ == "__main__":
    main()
