import json
import logging
import os
import sys

import chainlit as cl
import pandas as pd
import requests
from dotenv import load_dotenv

from utils.general import call_execute_query_api, call_get_memory_recipe_api
from utils.llm import gen_sql, gen_summarize_results

logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger = logging.getLogger()

load_dotenv("../../.env")


def print(*tup):
    logger.info(" ".join(str(x) for x in tup))


async def ask_data(input, chat_history, active_message):
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

    # Default to memory/recipe
    mode = "memory_recipe"

    chat_history = cl.user_session.get("chat_history")
    if len(chat_history) > 3:
        chat_history = chat_history[-3:]

    # Loop 3 times to retry errors
    for i in range(5):
        try:
            if mode == "memory_recipe":
                output = await call_get_memory_recipe_api(
                    input, history=str(chat_history), generate_intent="true"
                )
                # To do, make this a variable in recipes module
                if "Sorry, no recipe or found" in str(output):
                    mode = "execute_query"
                sql = ""

            if mode == "execute_query":
                # active_message.content = "Hmm. I didn't find any recipes, let me query the database  ..."
                # await active_message.update()
                # await active_message.send()

                sql = await gen_sql(input, str(chat_history), output)
                print(sql)
                output = await call_execute_query_api(sql)

            # Hack for the demo
            if "error" in str(output):
                print("Error in output, trying again ...")
            else:
                output = await gen_summarize_results(input, sql, output)

            # print(output)
            break
        except Exception as e:
            print(e)
        if i == 2:
            print("Failed to execute query")
            break

    return output


@cl.step(type="tool")
async def tool(message: str, active_message: cl.Message):
    """
    This function represents a tool step in the data recipe chat chainlit.

    Parameters:
        message (str): The message to be passed to the ask_data function.

    Returns:
        The result obtained from the ask_data function.
    """
    result = await ask_data(message, [], active_message)
    return result


@cl.on_chat_start
async def start_chat():
    """
    Starts a chat session by creating a new thread, setting the thread in the user session,
    and sending an introductory message from bot.
    """

    cl.user_session.set("messages", [])


async def add_message_to_history(message, role):
    """
    Adds a message to the chat history.

    Args:
        message: The message to be added to the chat history.
        role: The role of the message (bot/user)

    Returns:
        None.
    """

    if cl.user_session.get("chat_history") is None:
        cl.user_session.set("chat_history", [])

    chat_history = cl.user_session.get("chat_history") + [
        {"role": role, "content": message},
    ]
    cl.user_session.set("chat_history", chat_history)


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

    await add_message_to_history(message.content, "user")

    final_answer = await cl.Message(content="").send()

    # Call the tool
    final_answer.content = await tool(message.content, final_answer)

    # print(final_answer.content)

    # await add_message_to_history(final_answer.content, "bot")

    await final_answer.update()
