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
        "mem_type": "recipe",
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
            + "# Calling code:\n#Add your calling code here\n"
        )


def create_metadata_upload_file(folder_path, metadata_path, recipe_path):
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
