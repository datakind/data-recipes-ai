import base64
import json
import os
import re
import sys
import warnings

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Suppress all warnings
warnings.filterwarnings("ignore")

execute_query_url = f"{os.getenv('RECIPE_SERVER_API')}execute_query"
get_memory_recipe_url = f"{os.getenv('RECIPE_SERVER_API')}get_memory_recipe"

data_info = None


def replace_env_variables(value):
    """
    Recursively replaces environment variable placeholders in a given value.

    Args:
        value (dict, list, str): The value to process

    Returns:
        The processed value with environment variable placeholders replaced.

    """
    if isinstance(value, dict):
        return {k: replace_env_variables(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [replace_env_variables(v) for v in value]
    elif isinstance(value, str):
        matches = re.findall(r"\{\{ (.+?) \}\}", value)
        for match in matches:
            value = value.replace("{{ " + match + " }}", os.getenv(match))
        return value
    else:
        return value


def read_integration_config(integration_config_file):
    """
    Read the APIs configuration from the integration config file.

    Args:
        integration_config_file (str): The path to the integration config file.

    Returns:
        apis (dict): A dictionary containing the API configurations.
        field_map (dict): A dictionary containing the field mappings.
        standard_names (dict): A dictionary containing the standard names.
        data_node (str): The data node to use for the integration.
    """
    with open(integration_config_file) as f:
        print(f"Reading {integration_config_file}")
        config = json.load(f)
        config = replace_env_variables(config)
        apis = config["openapi_interfaces"]
        field_map = config["field_map"]
        standard_names = config["standard_names"]

    return apis, field_map, standard_names


def is_running_in_docker():
    """
    Check if the code is running inside a Docker container.

    Returns:
        bool: True if running inside a Docker container, False otherwise.
    """
    return os.path.exists("/.dockerenv")


def make_api_request(url, payload):
    """
    Makes an API request to the specified URL with the given payload.

    Args:
        url (str): The URL to make the API request to.
        payload (dict): The payload to send with the API request.

    Returns:
        dict: The response from the API as a dictionary.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the API request.
    """
    headers = {"Content-Type": "application/json"}
    print(f"API URL: {url}")
    print(f"API Payload: {payload}")
    response = requests.post(url, headers=headers, json=payload)
    print(f"API Response Status Code: {response.status_code}")
    response = response.content
    print(f"API Response {response}")
    return response


def call_execute_query_api(sql):
    """
    Calls the execute query action API endpoint with the given SQL query.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        dict: The response from the API.

    """
    data = {"query": f"{sql}"}
    print(f"Calling execute query API {execute_query_url} with {sql} ...")
    return make_api_request(execute_query_url, data)


def call_get_memory_recipe_api(user_input, history, generate_intent="true"):
    """
    Calls the API to get a memory recipe action.

    Args:
        user_input (str): The user input.
        history (str): The chat history.
        generate_intent (str): Whether to generate the intent.


    Returns:
        The API response from the make_api_request function.
    """

    data = {
        "user_input": f"{user_input}",
        "chat_history": history,
        "generate_intent": "true",
    }
    print(f"Calling execute query API {get_memory_recipe_url} with {data} ...")
    result = make_api_request(get_memory_recipe_url, data)

    if isinstance(result, bytes):
        result = result.decode("utf-8")

    print("IN API CALL", result)
    result = json.loads(result)

    return result
