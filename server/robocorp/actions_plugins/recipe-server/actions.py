import logging
import sys
from functools import lru_cache

# This directory ../utils is copied or mounted into Docker image
from utils.recipes import get_memory_recipe as gmr


# @lru_cache(maxsize=100)
def get_memory_recipe(user_input, chat_history, generate_intent="true") -> str:
    """
    Performs a search in the memory for a given intent and returns the best match found.

    Args:
        user_input (str): The user input to search for in the memory.
        chat_history (str): The chat history.
        generate_intent (str): A flag to indicate whether to generate the intent from the chat history.

    Returns:
        str: Matched value + metadata
    """

    logging.info("Python HTTP trigger function processed a request.")

    result = gmr(user_input, chat_history, generate_intent)

    return str(result)


if __name__ == "__main__":
    # query = "Generate a population map for Haiti at the administrative level 1"
    # query = "What's the total population of AFG"
    # query = "what's the population of Mali"
    # query = "what recipes do you have"
    # query = "Create a chart that demonstrates the number of organizations working in Sila within each sector"
    query = "plot a bar chart of humanitarian organizations in Wadi Fira by sector"
    # history = str(
    # [
    #    {
    #        "inputs": {
    #            "question": "Generate a population map for Haiti at the administrative level 1"
    #        }
    #    },
    # ]
    # )
    get_memory_recipe(query, "[]", False)
