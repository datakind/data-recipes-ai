import json
import logging
import os
import sys

import chainlit as cl
import pandas as pd
import requests
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from utils.utils import call_llm

logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger = logging.getLogger()


def print(*tup):
    logger.info(" ".join(str(x) for x in tup))


llm_prompt_cap = 5000
sql_rows_cap = 100

# URL for actions, assuming FastAPI instance
base_url = "http://actions:8080/"
execute_query_url = f"{base_url}execute_query"
read_memory_recipe_url = f"{base_url}get_memory_recipe"

environment = Environment(loader=FileSystemLoader("./templates/"))

load_dotenv("../../.env")

data_info = None


async def get_data_info():
    """
    Get data info from the database.

    Returns:
        str: The data info.
    """

    global data_info

    # run this query: select table_name, summary, columns from table_metadata

    query = """
        SELECT
            table_name,
            summary,
            columns
        FROM
            table_metadata
        --WHERE
        --    countries is not null
        """

    data_info = await call_execute_query_action(query)


async def gen_sql(input, chat_history, output):
    """
    Generate SQL query based on input, chat history, and output.

    Args:
        input (str): The input for generating the SQL query.
        chat_history (str): The chat history used for generating the SQL query.
        output (str): The output of the SQL query.

    Returns:
        str: The generated SQL query.

    Raises:
        None

    """
    global data_info

    if data_info is None:
        data_info = await get_data_info()

    gen_sql_template = environment.get_template("gen_sql_prompt.jinja2")
    prompt = gen_sql_template.render(
        input=input,
        stdout_output=output,
        stderr_output="",
        data_info=data_info,
        chat_history=chat_history,
    )
    print(prompt)

    response = call_llm("", prompt)

    query = response["code"]

    query = query.replace(";", "") + f" \nLIMIT {sql_rows_cap};"

    return query


async def gen_summarize_results(input, sql, output):
    """
    Summarizes the results of a query and answers the user's question.

    Args:
        input (str): The user's question.
        sql (str): The SQL query that was executed.
        output (str): The output of the executed query.

    Returns:
        str: The summarized results of the query.

    Important:
    - If you see 'attribution' in the response with a URL, display it as a foot note like this: "Reference: [HDX](<URL>)
    When showing results for these questions, always add a foot note: "✅ *A human approved this data recipe*"
    - Always display images inline, do not use links
    """
    cl.Message("    Summarizing results ...").send()

    if len(output) > llm_prompt_cap:
        output = output[:llm_prompt_cap] + "..."

    prompt = f"""
        The user asked this question:

        {input}

        Which resulted in this SQL query:

        {sql}

        The query was executed and the output was:

        {output}

        Important:

        - If you see 'attribution'  in the response with a URL, display it as a foot note like this: "Reference: [HDX](<URL>)
        When showing results for these questions, always add a foot note: "✅ *A human approved this data recipe*"
        - Always display images inline, do not use links

        Task:

        Summarize the results of the query and answer the user's question

    """

    response = call_llm("", prompt)
    if "content" in response:
        response = response["content"]

    return response


async def make_api_request(url, payload):
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
    response = response.json()
    print(f"API Response {response}")
    return response


async def call_execute_query_action(sql):
    """
    Calls the execute query action API endpoint with the given SQL query.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        dict: The response from the API.

    """
    data = {"query": f"{sql}"}
    return await make_api_request(execute_query_url, data)


async def call_get_memory_recipe_action(user_input):
    """
    Calls the API to get a memory recipe action.

    Args:
        user_input (str): The user input.

    Returns:
        The API response from the make_api_request function.
    """
    data = {
        "user_input": f"{user_input}",
        "chat_history": "[]",
        "generate_intent": "true",
    }
    return await make_api_request(read_memory_recipe_url, data)


async def ask_data(input, chat_history):
    """
    Asynchronously processes the input data and chat history to generate an output.

    Args:
        input: The input data.
        chat_history: The chat history.

    Returns:
        The generated output.

    Raises:
        Exception: If there is an error during the execution of the query.
    """

    output = ""

    # Loop 3 times to retry errors
    for i in range(5):
        # sql = await gen_sql(input, chat_history, output)
        try:
            output = await call_get_memory_recipe_action(input)
            # output = await call_execute_query_action(sql)
            sql = ""
            output = await gen_summarize_results(input, sql, output)
            print(output)
            break
        except Exception as e:
            print(e)
        if i == 2:
            print("Failed to execute query")
            break

    return output


@cl.step(type="tool")
async def tool(message: str):
    """
    This function represents a tool step in the data recipe chat chainlit.

    Parameters:
        message (str): The message to be passed to the ask_data function.

    Returns:
        The result obtained from the ask_data function.
    """
    result = await ask_data(message, [])
    return result


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    final_answer = await cl.Message(content="").send()

    # Call the tool
    final_answer.content = await tool(message.content)

    await final_answer.update()
