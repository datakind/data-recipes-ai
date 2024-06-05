import ast
import base64
import io
import json
import logging
import os
import sys

from dotenv import load_dotenv
from PIL import Image
from robocorp.actions import action

# This directory ../utils is copied or mounted into Docker image
from utils.recipes import check_recipe_memory, generate_intent_from_history

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


@action()
def get_memory(user_input, chat_history, generate_intent=True) -> str:
    """
    Performs a search in the memory for a given intent and returns the best match found.

    Args:
        user_input (str): The user input to search for in the memory.
        chat_history (str): The chat history.
        generate_intent (str): A flag to indicate whether to generate the intent from the chat history.

    Returns:
        str: The 3 best matches found in the memory.
    """

    logging.info("Python HTTP trigger function processed a request.")
    # Retrieve the CSV file from the request

    if generate_intent is not None and generate_intent is True:
        # chat history is passed from promptflow as a string representation of a list and this has to be converted back to a list for the intent generation to work!
        history_list = ast.literal_eval(chat_history)
        history_list.append({"inputs": {"question": user_input}})
        user_input = generate_intent_from_history(history_list)
        # turn user_input into a proper json record
        user_input = json.dumps(user_input)
    memory_found, result = check_recipe_memory(user_input, mem_type="memory")
    if memory_found is True:
        if result["metadata"]["result_type"] == "image":
            response_image = result["metadata"]["result"]
            response_text = ""
        else:
            response_text = result["metadata"]["result"]
            response_image = ""
        recipe_id = result["metadata"]["custom_id"]
        print("Recipe ID: ", recipe_id)
        if response_image is not None and response_image != "":
            process_image(
                response_image.replace("data:image/png;base64,", ""), recipe_id
            )
            # result = "http://localhost:9999/memory_image.png"
            result = f"{os.getenv('IMAGE_HOST')}/memory_image_{recipe_id}.png"

        else:
            result = response_text
    else:
        result = "No memory found"
    return result


if __name__ == "__main__":
    query = "What is the total population of Mali?"
    history = str(
        [
            {"inputs": {"question": "What is the total population of Mali?"}},
        ]
    )
    get_memory(query, history, False)
