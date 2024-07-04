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

    client, assistant_id, model = setup_openai()

    standard_names = get_common_field_standard_names()

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

    tools = [{"type": "code_interpreter"}]

    # Add in our functions
    tools = tools + get_manually_defined_functions()

    # Find if agent exists. v1 needs a try/except for this, TODO upgrade to v2 API
    try:
        print(
            f"Updating existing assistant {assistant_id} {bot_name} and model {model} ..."
        )
        assistant = client.beta.assistants.update(
            assistant_id,
            name=bot_name,
            instructions=instructions,
            tools=tools,
            model=model,
            temperature=0.1,
            # file_ids=file_ids,
        )
    except Exception:
        print(f"Creating assistant with model {model} ...")
        assistant = client.beta.assistants.create(
            name=bot_name,
            instructions=instructions,
            tools=tools,
            model=model,
            temperature=0.1,
            # file_ids=file_ids,
        )
        print("Assistant created!! Here is the assistant ID:")
        print(assistant.id)
        print("Now save the ID in your .env file so next time it's updated")


if __name__ == "__main__":
    create_update_assistant()
