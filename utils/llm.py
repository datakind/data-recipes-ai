import base64
import json
import os
import sys

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

from utils.db import get_data_info

load_dotenv()

# Caps for LLM summarization of SQL output and number of rows in the output
llm_prompt_cap = 5000
sql_rows_cap = 100

# Because CLI runs on host. TODO, make this more elegant.
template_dir = "../templates"
if not os.path.exists(template_dir):
    template_dir = "./templates"
environment = Environment(loader=FileSystemLoader(template_dir))
sql_prompt_template = environment.get_template("gen_sql_prompt.jinja2")

chat = None
embedding_model = None

data_info = None


def get_models():
    """
    Retrieves the embedding model and chat model based on the specified API type.

    Returns:
        embedding_model: The embedding model used for text embeddings.
        chat: The chat model used for generating responses.

    Raises:
        SystemExit: If the specified API type is not supported.
    """
    api_key = os.getenv("RECIPES_OPENAI_API_KEY")
    base_url = os.getenv("RECIPES_BASE_URL")
    api_version = os.getenv("RECIPES_OPENAI_API_VERSION")
    api_type = os.getenv("RECIPES_OPENAI_API_TYPE")
    completion_model = os.getenv("RECIPES_OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME")
    model = os.getenv("RECIPES_MODEL")
    temp = os.getenv("RECIPES_MODEL_TEMP")
    max_tokens = os.getenv("RECIPES_MODEL_MAX_TOKENS")

    if api_type == "openai":
        embedding_model = OpenAIEmbeddings(
            api_key=api_key,
        )
        chat = ChatOpenAI(
            model_name=model,
            api_key=api_key,
            temperature=temp,
            max_tokens=max_tokens,
        )
    elif api_type == "azure":
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
            temperature=temp,
            max_tokens=max_tokens,
        )
    else:
        print(f"OPENAI API type: {api_type} not supported")
        sys.exit(1)
    return embedding_model, chat


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
        # print(response)
        print("Error calling LLM: ", e)
        response = None


def gen_sql(input, chat_history, output):
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
        data_info = get_data_info()

    prompt = sql_prompt_template.render(
        input=input,
        stdout_output=output,
        stderr_output="",
        data_info=data_info,
        chat_history=chat_history,
    )

    response = call_llm("", prompt)

    query = response["code"]

    query = query.replace(";", "") + f" \nLIMIT {sql_rows_cap};"

    return query
