import ast
import json
import logging
import sys
import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from robocorp.actions import action
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Lower numbers are more similar
similarity_cutoff = {"memory": 0.2, "recipe": 0.3, "helper_function": 0.1}

response_formats = [
    "csv",
    "dataframe",
    "json",
    "file_location",
    "integer",
    "float",
    "string",
]

prompt_map = {
    "memory": """
        You judge matches of user intent with those stored in a database to decide if they are true matches of intent.
        When asked to compare two intents, check they are the same, have the same entities and would result in the same outcome.
        Be very strict in your judgement. If you are not sure, say no.
        A plotting intent is different from a request for just the data.
        Intents to generate plots must have the same plot type, data and data aggregation level.


        Answer with a JSON record ...

        {
            "answer": <yes or no>,
            "reason": <your reasoning>"
        }
    """,
    "recipe": """
        You judge matches of user intent with generic DB intents in the database to see if a DB intent can be used to solve the user's intent.
        The requested output format of the user's intent MUST be the same as the output format of the generic DB intent.
        For exmaple, if the user's intent is to get a plot, the generic DB intent MUST also be to get a plot.
        The level of data aggregation is important, if the user's request differs from the DB intent, then reject it.

        Answer with a JSON record ...

        {
            "answer": <yes or no>,
            "user_intent_output_format": <one of: """
    + str(response_formats)
    + """>,
            "generic_db_output_format": <one of: """
    + str(response_formats)
    + """>,
            "reason": <your reasoning>"
        }
    """,
    "helper_function": """
        You judge matches of user input helper functions and those already in the database
        If the user help function matches the database helper function, then it's a match

        Answer with a nested JSON record ...

        {
            "answer": <yes or no>,
            "reason": <your reasoning>"
        }
    """,
    "instructions_intent_from_history": """
    Given a user query, identify the intent of the user query.

    Determine the closest fit and rephrase the intent as one of the following standard sentence structures:
    - "What is the [data point] for [subject/topic] in [location/context]?"
    - "What are the recent trends in [topic] for [time period]?"
    - "Can you show a comparison of [data point] between [subject/topic] and [subject/topic] over [time period]?"
    - "Can you show a [type of visualization] of [data point] for [subject/topic] over [time period]?"
    - "What does the latest data say about [cause-effect relationship] in [context]?"
    - "What are the impacts of [event/condition] on [subject/topic] according to recent data?"
    - "Provide a visualization of the distribution of [resources/assistance] in [location]."
    - "How has the [data point] changed since [starting year]?"
    - "What are the latest figures for [subject/topic]?"

    """,
}

conn_params = {
    "OPENAI_API_TYPE": os.getenv("OPENAI_API_TYPE"),
    "OPENAI_API_ENDPOINT": os.getenv("OPENAI_API_ENDPOINT"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION_MEMORY"),
    "BASE_URL": os.getenv("BASE_URL_MEMORY"),
    "MODEL": os.getenv("MODEL"),
    "OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME": os.getenv("OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME"),
    "POSTGRES_DB": os.getenv("POSTGRES_RECIPE_DB"),
    "POSTGRES_USER": os.getenv("POSTGRES_RECIPE_USER"),
    "POSTGRES_HOST": os.getenv("POSTGRES_RECIPE_HOST"),
    "POSTGRES_PORT": os.getenv("POSTGRES_RECIPE_PORT"),
    "OPENAI_API_KEY": os.getenv("AZURE_API_KEY"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_RECIPE_PASSWORD"),
}

# Setting db to None so that we can initialize it in the first invocation
db = None


def call_llm(instructions, prompt, chat):
    """
    Call the LLM (Language Learning Model) API with the given instructions and prompt.

    Args:
        instructions (str): The instructions to provide to the LLM API.
        prompt (str): The prompt to provide to the LLM API.
        chat (Langchain Open AI model): Chat model used for AI judging

    Returns:
        dict or None: The response from the LLM API as a dictionary, or None if an error occurred.
    """
    try:
        messages = [
            SystemMessage(content=instructions),
            HumanMessage(content=prompt),
        ]
        response = chat(messages)
        print(response)
        try:
            response = json.loads(response.content)
        except Exception as e:
            print(f"Error creating json from response {e}")
            # Until gpt 3.5 has json output, we'll return just the string for now when prompting fails to create json
            response = response.content
        return response
    except Exception as e:
        print(f"Error calling LLM {e}")


def initialize_db(mem_type, connection_string, embedding_model):
    """
    Initialize the database by creating store tables if they don't exist and returns the initialized database.

    Returns:
        dict: The initialized database with store tables for each memory type.
    """
    db = {}

    # This will create store tables if they don't exist
    collection_name = f"{mem_type}_embedding"
    db[mem_type] = PGVector(
        collection_name=collection_name,
        connection_string=connection_string,
        embedding_function=embedding_model,
    )
    return db


def get_models(conn_params):
    """Write short summary.

    Args:
        conn_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    api_key = conn_params["OPENAI_API_KEY"]
    base_url = conn_params["BASE_URL"]
    api_version = conn_params["OPENAI_API_VERSION"]
    api_type = conn_params["OPENAI_API_TYPE"]
    completion_model = conn_params["OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME"]

    if api_type == "openai":
        print("Using OpenAI API in memory.py")
        embedding_model = OpenAIEmbeddings(
            api_key=api_key,
            # model=completion_model
        )
        chat = ChatOpenAI(
            # model_name="gpt-3.5-turbo",
            model_name="gpt-3.5-turbo-16k",
            api_key=api_key,
            temperature=1,
            max_tokens=1000,
        )
    elif api_type == "azure":
        print("Using Azure OpenAI API in memory.py")
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
            model_name="gpt-35-turbo",
            # model_name="gpt-4-turbo",
            # model="gpt-3-turbo-1106", # Model = should match the deployment name you chose for your 1106-preview model deployment
            # response_format={ "type": "json_object" },
            temperature=1,
            max_tokens=1000,
        )
    else:
        print("OPENAI API type not supported")
        sys.exit(1)
    return embedding_model, chat


def get_recipe_memory(intent) -> str:
    """Write short summary.

    Args:
        intent (_type_): _description_

    Returns:
        str: _description_
    """
    embedding_model, chat = get_models(conn_params)
    connection_string = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=conn_params["POSTGRES_HOST"],
        port=int(conn_params["POSTGRES_PORT"]),
        database=conn_params["POSTGRES_DB"],
        user=conn_params["POSTGRES_USER"],
        password=conn_params["POSTGRES_PASSWORD"],
    )
    mem_type = "memory"
    global db
    if db is None:
        db = initialize_db(mem_type, connection_string, embedding_model)
    memory_found, recipe_response = check_memory(intent, mem_type, db, chat)
    return memory_found, recipe_response


def check_memory(intent, mem_type, db, chat):
    """
    Check the memory for a given intent.

    Args:
        intent (str): The intent to search for in the memory.
        mem_type (str): The type of memory to search in. Can be 'memory', 'recipe', or 'helper_function'.
        db (Database): The database object to perform the search on.
        chat (Langchain OpenAI model): Chat model

    Returns:
        memory_found: Boolean, true if memeory found
        result: A dictionary containing the score, content, and metadata of the best match found in the memory.
            If no match is found, the dictionary values will be None.
    """
    if mem_type not in ["memory", "recipe", "helper_function"]:
        print("Memory type not recognised")
        sys.exit()
        return
    r = {"score": None, "content": None, "metadata": None}
    print(f"======= Checking {mem_type} for intent: {intent}")

    matches = get_matching_candidates(intent, mem_type, db)

    for m in matches:
        score = m["score"]
        content = m["content"]
        metadata = m["metadata"]
        if (
            metadata["calling_code_run_status"] != "ERROR"
            and metadata["mem_type"] == mem_type
        ):
            print(f"\n Reranking candidate: Score: {score} ===> {content} \n")
            # Here ask LLM to confirm our match
            prompt = f"""
                User Intent:

                {intent}

                DB Intent:

                {content}

            """

            response = call_llm(prompt_map[mem_type], prompt, chat)
            print(response)

            if "user_intent_output_format" in response:
                if (
                    response["user_intent_output_format"]
                    != response["generic_db_output_format"]
                ):
                    response["answer"] = "no"
                    response["reason"] = "output formats do not match"

            print("AI Judge of match: ", response)

            if response["answer"].lower() == "yes":
                print("We have a match!!!")
                r["score"] = score
                r["content"] = content
                r["metadata"] = metadata
                return True, r

    return False, r


def get_matching_candidates(intent, mem_type, db, cutoff=None):
    """
    Get the matching candidates for a given intent based on similarity search. No LLM judge.

    Args:
        intent (str): The intent to search for in the memory.
        mem_type (str): The type of memory to search in. Can be 'memory', 'recipe', or 'helper_function'.
        db (Database): The database object to perform the search on.
        cutoff (float, optional): The similarity cutoff to use for the search. Defaults to similarity_cutoff[mem_type]

    Returns:
        list: A list of matching candidates found in the memory.
    """
    if mem_type not in ["memory", "recipe", "helper_function"]:
        print("Memory type not recognised")
        sys.exit()
        return
    if cutoff is None:
        cutoff = similarity_cutoff[mem_type]
    print(f"\n\n======= Getting matches for {mem_type} and intent: {intent}\n\n")
    docs = db[mem_type].similarity_search_with_score(intent, k=3)
    matches = []
    for d in docs:
        r = {}
        score = d[1]
        content = d[0].page_content
        metadata = d[0].metadata
        print("\n", f"\n\nMatches: Score: {score} ===> {content}\n\n")
        if (
            metadata["calling_code_run_status"] != "ERROR"
            and metadata["mem_type"] == mem_type
        ):
            if d[1] < cutoff:
                print("\n", " << MATCHED >>")
                r["score"] = score
                r["content"] = content
                r["metadata"] = metadata
                matches.append(r)

    return matches


def generate_intent_from_history(chat_history: list, remove_code: bool = True) -> dict:
    """
    Generate the intent from the user query and chat history.

    Args:
        chat_history (str): The chat history.

    Returns:
        dict: The generated intent.

    """
    chat = get_models(conn_params)[1]
    # Only use last few interactions
    buffer = 4
    if len(chat_history) > buffer:
        chat_history = chat_history[-buffer:]

    # Remove any code nodes
    if remove_code is True:
        chat_history2 = []
        for c in chat_history:
            chat_history2.append(c)
            if "code" in c:
                # remove 'code' from dictionary c
                c.pop("code")
        chat_history = chat_history2

    prompt = f"""
        Given the chat history below, what is the user's intent?

        {chat_history}

    """
    intent = call_llm(
        instructions=prompt_map["instructions_intent_from_history"],
        prompt=prompt,
        chat=chat,
    )
    if not isinstance(intent, dict):
        intent = {"intent": intent}
    print(f"Generated intent: {intent}")
    return intent["intent"]


@action()
def get_memory(user_input, chat_history, generate_intent=True) -> str:
    """
    Performs a search in the memory for a given intent and returns the best match found.

    Args:
        conn_params (str): The connection parameters for the database.
        user_input (str): The user input to search for in the memory.
        chat_history (str): The chat history.
        generate_intent (str): A flag to indicate whether to generate the intent from the chat history.

    Returns:
        str: The 3 best matches found in the memory.
    """

    logging.info("Python HTTP trigger function processed a request.")
    # Retrieve the CSV file from the request

    if generate_intent is not None and generate_intent == True:
        # chat history is passed from promptflow as a string representation of a list and this has to be converted back to a list for the intent generation to work!
        history_list = ast.literal_eval(chat_history)
        history_list.append({"inputs": {"question": user_input}})
        user_input = generate_intent_from_history(history_list)
        # turn user_input into a proper json record
        user_input = json.dumps(user_input)
    print(f"\n\n\n\n+++++++++User intent: {user_input}\n\n\n\n")
    memory_found, result = get_recipe_memory(user_input)
    if memory_found is True:
        response_text = result["metadata"]["response_text"]
        response_image = result["metadata"]["response_image"]
        if response_image is not None and response_image != "":
            result = f"![Visualization](http://localhost:9999/{result['metadata']['custom_id']}.png"
        else:
            result = response_text
    else:
        result = "No memory found"
    return result
