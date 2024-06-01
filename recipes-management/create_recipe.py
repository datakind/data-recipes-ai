import json
import os
import shutil
import sys

from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from recipe_sync import extract_code_sections, update_metadata_file
from skills import add_memory

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

chat = AzureChatOpenAI(
    model_name="gpt-35-turbo",
    azure_endpoint=os.getenv("RECIPES_BASE_URL"),
    api_version=os.getenv("RECIPES_OPENAI_API_VERSION"),
    temperature=1,
    max_tokens=1000,
)


def create_recipe_folder(recipe_name):
    """
    Create a new recipe folder with necessary metadata and template files.

    Parameters:
    - recipe_name (str): The name of the recipe.

    This function performs the following steps:
    1. Defines the folder name and path for the new recipe.
    2. Creates the folder if it does not already exist.
    3. Defines the metadata structure with placeholder values.
    4. Writes the metadata to a `metadata.json` file in the recipe folder.
    5. Reads an `imports.txt` file if it exists, or uses a default import template.
    6. Writes a `recipe.py` file in the recipe folder with the imports and a template for the recipe code.
    """

    # Define the folder name
    folder_name = "new_recipe_staging"
    recipe_folder = os.path.join(folder_name, recipe_name)

    # Create the folder
    os.makedirs(recipe_folder, exist_ok=True)

    # Define the metadata structure
    metadata = {
        "recipe_name": recipe_name,
        "intent": "Please enter intent",
        "python_packages": ["Please enter the python packages"],
        "parameters": "Please enter the arguments the recipe expects",
        "response_text": "Please enter your response text",
        "data_attribution": "Please enter the ist of table names that the recipe code queries to determine the answer",
        "mem_type": "recipe",
        "locked_at": "",
        "locked_by": "",
    }

    # Write metadata.json file
    metadata_path = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    # Read imports.txt file
    imports_path = "imports.txt"
    if os.path.exists(imports_path):
        with open(imports_path, "r") as file:
            imports = file.read()
    else:
        imports = "# Add your imports here\n"

    # Write recipe.py file
    recipe_path = os.path.join(recipe_folder, "recipe.py")
    with open(recipe_path, "w", encoding="utf-8") as recipe_file:
        recipe_file.write(
            imports
            + "\n\n"
            + "# Add your functions code here\n\n\n"
            + "# Calling code:\n\n#Add your calling code here\n\n"
        )


def create_metadata_upload_file(folder_path, metadata_path, recipe_path):
    """
    Create and update a metadata upload file for a recipe.

    Parameters:
    - folder_path (str): The path to the folder where the metadata upload file will be created.
    - metadata_path (str): The path to the original metadata.json file.
    - recipe_path (str): The path to the recipe.py file containing the code sections.

    This function performs the following steps:
    1. Creates a new file path for the metadata upload file (`metadata_upload.json`) in the specified folder.
    2. Copies the original `metadata.json` file to the new file path.
    3. Updates the copied metadata file by adding code sections extracted from the `recipe.py` file.
    """

    # add metadata.json to the filepath, then create copy of it
    metadata_path_new = os.path.join(folder_path, "metadata_upload.json")
    shutil.copy(metadata_path, metadata_path_new)
    update_metadata_file(
        metadata_path=metadata_path_new,
        code_sections=extract_code_sections(recipe_path),
    )


# TODO: THIS SHOULD BE IMPORTED FROM SKILLS
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


def main():
    """
    Main function to handle command line arguments and execute corresponding actions.

    This function supports the following commands:
    1. `--create_recipe_template <recipe_name>`: Creates a new recipe folder with the specified recipe name.
    2. `--prepare_push_recipe_to_db`: Prepares all recipe folders in 'new_recipe_staging' by creating metadata upload files.
    3. `--push_recipe_to_db`: Pushes all prepared recipes in 'new_recipe_staging' to the database.

    Usage:
    - python create_recipe.py --create_recipe_template <recipe_name>
    - python create_recipe.py --prepare_push_recipe_to_db
    - python create_recipe.py --push_recipe_to_db

    The function will print usage instructions and exit if incorrect or insufficient arguments are provided.
    """

    if len(sys.argv) < 2:
        print(
            "Usage: python create_recipe.py --create_recipe_template <recipe_name> | --prepare_push_recipe_to_db | --push_recipe_to_db"
        )
        sys.exit(1)

    if sys.argv[1] == "--create_recipe_template":
        if len(sys.argv) != 3:
            print(
                "Usage: python create_recipe.py --create_recipe_template <recipe_name>"
            )
            sys.exit(1)
        recipe_name = sys.argv[2]
        create_recipe_folder(recipe_name)
        print(
            f"Recipe folder '{recipe_name}' created successfully in 'new_recipe_staging'."
        )
    elif sys.argv[1] == "--prepare_push_recipe_to_db":
        # for each recipe folder in new_recipe_staging, create a metadata_upload.json file
        for recipe_folder in os.listdir("new_recipe_staging"):
            if recipe_folder.startswith("."):
                continue
            full_path = os.path.join("new_recipe_staging", recipe_folder)
            metadata_path = os.path.join(full_path, "metadata.json")
            recipe_path = os.path.join(full_path, "recipe.py")
            create_metadata_upload_file(
                folder_path=full_path,
                metadata_path=metadata_path,
                recipe_path=recipe_path,
            )
    elif sys.argv[1] == "--push_recipe_to_db":
        db = initialize_db()
        for recipe_folder in os.listdir("new_recipe_staging"):
            if recipe_folder.startswith("."):
                continue
            full_path = os.path.join("new_recipe_staging", recipe_folder)
            metadata_path = os.path.join(full_path, "metadata_upload.json")
            metadata = json.load(open(metadata_path))
            add_memory(
                intent=metadata["intent"],
                metadata=metadata,
                db=db,
                mem_type=metadata["mem_type"],
            )
    else:
        print(
            "Invalid option. Usage: python create_recipe.py --create_recipe_template <recipe_name> | --push_recipe_to_db"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
