import base64
import json
import os
import re
import sys
import warnings

import psycopg2
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from sqlalchemy import create_engine

# Suppress all warnings
warnings.filterwarnings("ignore")

chat = None
embedding_model = None


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


def call_llm(instructions, prompt, image=None):
    """
    Call the LLM (Language Learning Model) API with the given instructions and prompt.

    Args:
        instructions (str): The instructions to provide to the LLM API.
        prompt (str): The prompt to provide to the LLM API.
        chat (Langchain Open AI model): Chat model used for AI judging

    Returns:
        dict or None: The response from the LLM API as a dictionary, or None if an error occurred.
    """

    global chat, embedding_model
    if chat is None or embedding_model is None:
        embedding_model, chat = get_models()

    human_message = HumanMessage(content=prompt)

    # Multimodal
    if image:
        if os.getenv("RECIPES_MODEL") == "gpt-4o":
            print("Sending image to LLM ...")
            with open(image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()

            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        },
                    },
                ]
            )
        else:
            print("Multimodal not supported for this model")
            return None

    try:
        messages = [
            SystemMessage(content=instructions),
            human_message,
        ]
        response = chat(messages)

        if hasattr(response, "content"):
            response = response.content

        if "content" in response and not response.startswith("```"):
            response = json.loads(response)
            response = response["content"]

        # Some silly things that sometimes happen
        response = response.replace(",}", "}")

        # Different models do different things when prompted for JSON. Here we try and handle this
        try:
            # Is it already JSON?
            response = json.loads(response)
        except json.decoder.JSONDecodeError:
            # Did the LLM provide JSON in ```json```?
            if "```json" in response:
                # print("LLM responded with JSON in ```json```")
                response = response.split("```json")[1]
                response = response.replace("\n", "").split("```")[0]
                response = json.loads(response)
            elif "```python" in response:
                # print("LLM responded with Python in ```python```")
                all_sections = response.split("```python")[1]
                code = all_sections.replace("\n", "").split("```")[0]
                message = all_sections.split("```")[0]
                response = {}
                response["code"] = code
                response["message"] = message
            else:
                # Finally just send it back
                print("LLM response unparsable, using raw results")
                print(response)
                response = {"content": response}
        return response

    except Exception as e:
        print(response)
        print("Error calling LLM: ", e)
        response = None


def get_models():
    api_key = os.getenv("RECIPES_OPENAI_API_KEY")
    base_url = os.getenv("RECIPES_BASE_URL")
    api_version = os.getenv("RECIPES_OPENAI_API_VERSION")
    api_type = os.getenv("RECIPES_OPENAI_API_TYPE")
    completion_model = os.getenv("RECIPES_OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME")
    model = os.getenv("RECIPES_MODEL")

    # print("LLM Settings ...")
    # print(f"   API type: {api_type}")
    # print(f"   Model name: {model}")

    if api_type == "openai":
        # print("Using OpenAI API in memory.py")
        embedding_model = OpenAIEmbeddings(
            api_key=api_key,
            # model=completion_model
        )
        chat = ChatOpenAI(
            model_name=model,
            api_key=api_key,
            temperature=1,
            max_tokens=3000,
        )
    elif api_type == "azure":
        # print("Using Azure OpenAI API in memory.py")
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
            model_name=model,
            # response_format={ "type": "json_object" },
            temperature=1,
            max_tokens=1000,
        )
    else:
        print("OPENAI API type not supported")
        sys.exit(1)
    return embedding_model, chat


def get_connection(instance="data"):
    """
    This function gets a connection to the database

    Args:

        instance (str): The instance of the database to connect to, "recipe" or "data". Default is "data"
    """
    instance = instance.upper()

    host = os.getenv(f"POSTGRES_{instance}_HOST")
    port = os.getenv(f"POSTGRES_{instance}_PORT")
    database = os.getenv(f"POSTGRES_{instance}_DB")
    user = os.getenv(f"POSTGRES_{instance}_USER")
    password = os.getenv(f"POSTGRES_{instance}_PASSWORD")

    conn = psycopg2.connect(
        dbname=database, user=user, password=password, host=host, port=port
    )
    return conn


def execute_query(query, instance="data"):
    """
    This skill executes a query in the data database.

    To find out what tables and columns are available, you can run "select table_name, api_name, summary, columns from table_metadata"

    """
    conn = get_connection(instance)
    cur = conn.cursor()

    # Execute the query
    cur.execute(query)

    # Fetch all the returned rows
    rows = cur.fetchall()

    # Close the cursor and connection
    cur.close()
    conn.close()

    return rows


def connect_to_db(instance="recipe"):
    """
    Connects to the specified database instance (RECIPE or DATA) DB and returns a connection object.

    Args:
        instance (str): The name of the database instance to connect to. Defaults to "RECIPE".

    Returns:
        sqlalchemy.engine.base.Engine: The connection object for the specified database instance.
    """

    instance = instance.upper()

    # Fallback for CLI running outside of docker
    if not is_running_in_docker():
        os.environ[f"POSTGRES_{instance}_HOST"] = "localhost"
        os.environ[f"POSTGRES_{instance}_PORT"] = "5433"

    host = os.getenv(f"POSTGRES_{instance}_HOST")
    port = os.getenv(f"POSTGRES_{instance}_PORT")
    database = os.getenv(f"POSTGRES_{instance}_DB")
    user = os.getenv(f"POSTGRES_{instance}_USER")
    password = os.getenv(f"POSTGRES_{instance}_PASSWORD")
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    # add an echo=True to see the SQL queries
    conn = create_engine(conn_str)
    return conn
