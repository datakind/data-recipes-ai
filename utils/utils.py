import json
import os
import re
import sys


def replace_env_variables(value):
    """
    Recursively replaces environment variable placeholders in a given value.

    Args:
        value (dict, list, str): The value to process.

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
