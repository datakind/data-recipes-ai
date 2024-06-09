import json
import logging
import os
import sys

import chainlit as cl
import pandas as pd
import requests
from dotenv import load_dotenv

from utils.general import call_execute_query_api, call_get_memory_recipe_api
from utils.llm import gen_summarize_results

logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger = logging.getLogger()

load_dotenv("../../.env")


def print(*tup):
    logger.info(" ".join(str(x) for x in tup))


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
            output = await call_get_memory_recipe_api(input)
            # output = await call_execute_query_api(sql)
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
