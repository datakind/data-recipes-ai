import ast
import base64
import io
import json
import logging
import os
import subprocess
import sys
from functools import lru_cache
from typing import Tuple

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from robocorp.actions import action

# This directory ../utils is copied or mounted into Docker image
from utils.recipes import (
    check_recipe_memory,
    generate_intent_from_history,
    get_memory_recipe_metadata,
)
from utils.utils import call_llm

environment = Environment(loader=FileSystemLoader("templates/"))

recipes_work_dir = "./recipes"

# Load environment variables from .env file
load_dotenv()


def process_image(encoded_string, recipe_id):
    """
    Takes a base64 encoded string of a picture, decodes it, and saves it as a PNG file.

    Args:
    encoded_string (str): Base64 encoded string of the image.
    recipe_id (str): The recipe ID to use in the image file name.

    Returns:
    str: Full path to the saved image file.
    """

    print("A visual memory was found. Processing image...")

    # Decode the base64 string
    image_data = base64.b64decode(encoded_string)

    # Convert binary data to image
    image = Image.open(io.BytesIO(image_data))

    # Create the full path for saving the image
    full_path = os.path.join("./images", f"memory_image_{recipe_id}.png")

    # Save the image
    image.save(full_path, "PNG")

    print("Image processed and saved successfully.")

    return full_path


def process_memory_recipe_results(result: dict, table_data: dict) -> str:
    """
    Processes the results of a memory recipe search and returns the response text and attribution.

    Args:
        result (dict): The result of the memory recipe search.
        table_data (dict): The data from the memory or recipe tables.

    Returns:
    """

    mem_type = result["metadata"]["mem_type"]
    custom_id = result["metadata"]["custom_id"]
    table_data = get_memory_recipe_metadata(custom_id, mem_type)
    attribution = table_data["attribution"]
    if attribution is None:
        attribution = ""
    print(f"====> Found {mem_type}")
    if table_data["result_type"] == "image":
        response_image = table_data["result"]
        response_text = ""
    else:
        response_text = table_data["result"]
        response_image = ""
    recipe_id = table_data["custom_id"]
    print("Recipe ID: ", recipe_id)
    if response_image is not None and response_image != "":
        process_image(response_image.replace("data:image/png;base64,", ""), recipe_id)
        # result = "http://localhost:9999/memory_image.png"
        result = f"{os.getenv('IMAGE_HOST')}/memory_image_{recipe_id}.png"
    else:
        result = response_text

    return {"result": result, "attribution": attribution}


def run_recipe(custom_id: str, recipe: dict, user_input, chat_history):
    """
    Runs a recipe based on the result of a memory recipe search.

    Args:
        custom_id (str): The custom ID of the recipe.
        recipe(dict): The recipe details
        user_input (str): The user input.
        chat_history (str): The chat history.

    """

    print("Attempting to run recipe...")
    print(f"Recipe Custom ID: {custom_id}")

    function_code = recipe["function_code"]

    # TODO this should really use the new openapi_json field,
    # but that's blank currently, will come back to it.
    calling_code = recipe["sample_call"]

    recipe_run_python_prompt = environment.get_template(
        "recipe_run_python_prompt.jinja2"
    )
    prompt = recipe_run_python_prompt.render(
        recipe_code=function_code,
        calling_code=calling_code,
        user_input=user_input,
        chat_history=chat_history,
    )
    print("Calling LLM to generate new run code ...")
    new_code = call_llm("", prompt, None)

    if "new_calling_code" in new_code:
        calling_code = new_code["new_calling_code"]
        print("New calling code generated ...")

        # Combine function and calling code into a string
        code = function_code + "\n\n" + "if __name__ == '__main__':\n\n"

        # Write calling code to code, indented by 4
        code += "    " + calling_code.replace("\n", "\n    ") + "\n"

        # Make recipes folder if it doesn't exist
        if not os.path.exists(recipes_work_dir):
            print(f"Creating recipes directory ... {recipes_work_dir}")
            os.makedirs(recipes_work_dir)
            # Copy skills.py into directory use shutil
            print("Copying skills.py into directory ...")

        # Adjust .env location in code
        code = code.replace("load_dotenv()", "load_dotenv('../.env')")

        # Adjust path
        code = (
            "import os\nimport sys\n# Add parent folder to path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
            + code
        )

        recipe_path = f"{recipes_work_dir}/{custom_id}.py"
        with open(recipe_path, "w") as f:
            f.write(code)

        cmd = f"python {recipe_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result)
        print(result.stdout)
        print(result.stderr)

    # Run the recipe here
    # exec(recipe)

    print("Recipe executed successfully.")


@lru_cache(maxsize=100)
@action()
def get_memory_recipe(user_input, chat_history, generate_intent=True) -> str:
    """
    Performs a search in the memory for a given intent and returns the best match found.

    Args:
        user_input (str): The user input to search for in the memory.
        chat_history (str): The chat history.
        generate_intent (str): A flag to indicate whether to generate the intent from the chat history.

    Returns:
        str: Matched value
        str: Attribution
    """

    logging.info("Python HTTP trigger function processed a request.")
    # Retrieve the CSV file from the request

    generate_intent = False

    if generate_intent is not None and generate_intent is True:
        # chat history is passed from promptflow as a string representation of a list and this has to be converted back to a list for the intent generation to work!
        history_list = ast.literal_eval(chat_history)
        history_list.append({"inputs": {"question": user_input}})
        user_input = generate_intent_from_history(history_list)
        # turn user_input into a proper json record
        user_input = json.dumps(user_input)

    for mem_type in ["memory", "recipe"]:
        print(f"Checking {mem_type}")
        memory_found, result = check_recipe_memory(
            user_input, mem_type=mem_type, ai_check=True
        )
        if memory_found is True:
            custom_id = result["metadata"]["custom_id"]
            # Get data from memory or recipe tables
            table_data = get_memory_recipe_metadata(custom_id, mem_type)
            if mem_type == "recipe":
                # Run the recipe
                result = run_recipe(custom_id, table_data, user_input, chat_history)
            else:
                # Take the result directly from memory
                result = process_memory_recipe_results(result, table_data)
                print(result)
            return str(result)

    result = "Sorry, no recipe or found"
    print(result)

    return str(result)


if __name__ == "__main__":
    # query = "Generate a population map for Haiti at the administrative level 1"
    query = "What's the total population of AFG"
    # query = "what's the population of Mali"
    # query = "what recipes do you have"
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
