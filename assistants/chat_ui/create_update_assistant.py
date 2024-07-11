import asyncio
import datetime
import glob
import json
import os
import sys
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import AzureOpenAI, OpenAI

from utils.db import get_data_info

load_dotenv("../../.env")

SUPPORTED_ASSISTANT_FILE_TYPES = ["csv", "json", "xlsx", "txt", "pdf", "docx", "pptx"]

bot_name = os.environ.get("ASSISTANTS_BOT_NAME")
environment = Environment(loader=FileSystemLoader("templates/"))

# Needed to get common fields standard_names
INTEGRATION_CONFIG = "ingestion.config"
SYSTEM_PROMPT = "instructions.txt"


def setup_openai():
    """
    Setup the OpenAI client.

    Returns:
        tuple: A tuple containing the OpenAI client, assistant ID, and model.
    """
    api_key = os.environ.get("ASSISTANTS_API_KEY")
    assistant_id = os.environ.get("ASSISTANTS_ID")
    model = os.environ.get("ASSISTANTS_MODEL")
    api_type = os.environ.get("ASSISTANTS_API_TYPE")
    api_endpoint = os.environ.get("ASSISTANTS_BASE_URL")
    api_version = os.environ.get("ASSISTANTS_API_VERSION")

    if api_type == "openai":
        print("Using OpenAI API")
        client = OpenAI(api_key=api_key)
    elif api_type == "azure":
        print("Using Azure API")
        print(f"Endpoint: {api_endpoint}")
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
            default_headers={"OpenAI-Beta": "assistants=v2"},
        )
    else:
        print("API type not supported")
        sys.exit(1)

    return client, assistant_id, model


def get_common_field_standard_names():
    """
    Get the standard names of common fields from the integration configuration file.

    Returns:
        list: A list of standard names of common fields.
    """
    with open(INTEGRATION_CONFIG) as f:
        print(f"Reading {INTEGRATION_CONFIG}")
        config = json.load(f)
    return config["standard_names"]


def get_local_files(assistant_file_type):
    """
    Get local files of a specific type.

    Args:
        assistant_file_type (str): The type of file to get, "file_search" or "code_interpreter"

    Returns:
        list: A list of file paths.
    """

    if assistant_file_type not in ["file_search", "code_interpreter"]:
        print(f"Assistant file type {assistant_file_type} is not supported")
        sys.exit(1)

    file_paths = []
    for file_type in SUPPORTED_ASSISTANT_FILE_TYPES:
        dir = f"./files/{assistant_file_type}"
        files = glob.glob(f"{dir}/**/*.{file_type}", recursive=True)
        file_paths = file_paths + files

    print(
        f"Found {len(file_paths)} files - {file_paths} - of type {assistant_file_type}"
    )

    return file_paths


def upload_files_for_code_interpreter(client):
    """
    Uploads files for the code interpretor section of the assistant.

    Args:
        client (OpenAI): The OpenAI client.
        assistant_id (str): The assistant ID.
    """
    file_ids = []

    file_paths = get_local_files("code_interpreter")

    for file_path in file_paths:
        print(f"Uploading {file_path} to code interpreter ...")
        file = client.files.create(file=open(file_path, "rb"), purpose="assistants")
        file_ids.append(file.id)

    return file_ids


def upload_files_to_vector_store(vector_store_name, client):
    """
    Uploads files to the vector store.

    Args:
        vestor_store_name(str): The name of the vector store.
        client (OpenAI): The OpenAI client.
    """

    file_paths = get_local_files("file_search")

    # No files to upload
    if len(file_paths) == 0:
        print("No files found to upload to vector store")
        return None

    print(f"Uploading {file_paths} to vector store {vector_store_name} ...")

    # Create a vector store
    vector_store = client.beta.vector_stores.create(name=vector_store_name)

    # Ready the files for upload to OpenAI
    file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    return vector_store.id


def get_manually_defined_functions():
    """
    Get a list of manually defined functions.

    Returns:
        list: A list of dictionaries representing the manually defined functions in openai format
    """
    functions = [
        {
            "type": "function",
            "function": {
                "name": "call_execute_query_api",
                "description": "Execute Query",
                "parameters": {
                    "properties": {
                        "sql": {
                            "type": "string",
                            "title": "SQL",
                            "description": "The SQL query to be executed. Only read queries are allowed.",
                        }
                    },
                    "type": "object",
                    "required": ["sql"],
                },
            },
        }
    ]

    return functions


def create_update_assistant():
    """
    Creates or updates a humanitarian response assistant.

    To force creation of a new assistant, be sure that ASSITANT_ID is not set in the .env file.

    """

    # Client setup
    client, assistant_id, model = setup_openai()

    # Information on common names
    standard_names = get_common_field_standard_names()

    # Get database information for system prompt
    data_info = get_data_info()
    print(data_info)

    # Load code examples
    template = environment.get_template("openai_assistant_prompt.jinja2")
    instructions = template.render(
        data_info=data_info,
        country_code_field=standard_names["country_code_field"],
        admin1_code_field=standard_names["admin1_code_field"],
        admin2_code_field=standard_names["admin2_code_field"],
        admin3_code_field=standard_names["admin3_code_field"],
    )

    # Save for debugging
    with open(SYSTEM_PROMPT, "w") as f:
        f.write(instructions)

    # Upload any local files needed by assistant for file_search (RAG)
    vector_store_id = upload_files_to_vector_store("local_files_vectore_store", client)

    # Upload any files that will be used for code_interpretor
    code_interpreter_file_ids = upload_files_for_code_interpreter(client)

    params = {
        "name": bot_name,
        "instructions": instructions,
        "model": model,
        "temperature": 0.1,
    }

    tool_resources = {}

    # Set tools
    tools = [{"type": "code_interpreter"}]

    # Add file search if we have data
    if vector_store_id is not None:
        tools.append({"type": "file_search"})
        params["tool_resources"] = {
            "file_search": {"vector_store_ids": [vector_store_id]}
        }
        tool_resources["file_search"] = {"vector_store_ids": [vector_store_id]}

    if len(code_interpreter_file_ids) > 0:
        tool_resources["code_interpreter"] = {"file_ids": code_interpreter_file_ids}

    # Add manually defined functions
    tools = tools + get_manually_defined_functions()

    params["tools"] = tools

    # Add in tool files as needed
    if "code_interpreter" in tool_resources or "file_search" in tool_resources:
        params["tool_resources"] = tool_resources

    # If we were provided an ID in .env, pass it in to update existing assistant
    if assistant_id is not None:
        params["assistant_id"] = assistant_id

    print(json.dumps(params, indent=4))

    if assistant_id is None:
        print(
            f"Calling assistant API for ID: {assistant_id}, name: {bot_name} and model {model} ..."
        )
        assistant = client.beta.assistants.create(**params)
        print("Assistant created!! Here is the assistant ID:\n")
        print(assistant.id)
        print("\nNow update ASSISTANTS_ID in your .env file with this ID")
    else:
        print(
            f"Calling assistant API for ID: {assistant_id}, name: {bot_name} and model {model} ..."
        )
        assistant = client.beta.assistants.update(**params)
        print("Assistant updated!!")


if __name__ == "__main__":
    create_update_assistant()
