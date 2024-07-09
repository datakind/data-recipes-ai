import json
import os
import sys

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from utils.db import execute_query as db_execute_query
from utils.recipes import get_memory_recipe

MAX_RESULTS = 500


class MemoryRecipeInput(BaseModel):
    """
    Represents the input data for a memory recipe.

    Attributes:
        user_input (str): The user's input.
        chat_history (str): The chat history.
        generate_intent (str, optional): Whether to generate an intent. Defaults to "false".
    """

    user_input: str
    chat_history: str
    generate_intent: str = "false"


class ExecuteQueryInput(BaseModel):
    """
    Represents the input data for executing a query.

    Attributes:
        query (str): The query to be executed.
    """

    query: str


app = FastAPI()


@app.post("/get_memory_recipe")
def memory_recipe_route(data: MemoryRecipeInput):
    """
    Retrieves a memory recipe based on the provided input data.

    Args:
        data (MemoryRecipeInput): The input data containing user input, chat history, and generate intent.

    Returns:
        The memory recipe generated based on the input data.
    """
    result = get_memory_recipe(data.user_input, data.chat_history, data.generate_intent)

    result = json.loads(result)
    return result


@app.post("/execute_query")
def execute_query_route(data: ExecuteQueryInput):
    """
    Executes a query and returns the results.

    Args:
        data (ExecuteQueryInput): The input data containing the query.

    Returns:
        pandas.DataFrame: The results of the query.
    """

    try:

        trailer = ""

        results = db_execute_query(data.query)
        num_results = results.shape[0]

        # TODO: Add code to send back a link, if results are too large
        if num_results > MAX_RESULTS:
            print("Results are too large to send back")
            results = results[0:MAX_RESULTS]
            trailer = "... etc"
            trailer += f"\n\nToo many rows ({num_results}) in the SQL query results. Please try again with a different query."

        results = results.to_json(orient="records")
        results = json.dumps(json.loads(results), indent=4)
        results += trailer

    except Exception as e:
        print(f"Error executing query: {e}")
        results = f"Error executing query: {e}"
        return results

    return results
