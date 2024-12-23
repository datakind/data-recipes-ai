import ast
import asyncio
import json
import logging
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List

import chainlit as cl
from chainlit import make_async, run_sync
from chainlit.config import config
from chainlit.element import Element
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from literalai.helper import utc_now
from openai import (
    AssistantEventHandler,
    AsyncAssistantEventHandler,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AzureOpenAI,
    OpenAI,
)
from typing_extensions import override

from utils.general import call_execute_query_api, call_get_memory_recipe_api

logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger = logging.getLogger()

load_dotenv("../../.env")

footer = "\n***\n"
llm_footer = footer + "🤖 *Caution: LLM Analysis*"
human_footer = footer + "✅ *A human approved this data recipe*"
images_loc = "./public/images/"
user = os.environ.get("USER_LOGIN")
password = os.environ.get("USER_PWD")


def setup(cl):
    """
    Sets up the assistant and OpenAI API clients based on the environment variables.

    Args:
        cl: The ChatLabs instance.

    Returns:
        tuple: A tuple containing the assistant, async OpenAI API client, and sync OpenAI API client.
    """

    if os.environ.get("ASSISTANTS_API_TYPE") == "openai":
        async_openai_client = AsyncOpenAI(api_key=os.environ.get("ASSISTANTS_API_KEY"))
        sync_openai_client = OpenAI(api_key=os.environ.get("ASSISTANTS_API_KEY"))
    else:
        async_openai_client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("ASSISTANTS_BASE_URL"),
            api_key=os.getenv("ASSISTANTS_API_KEY"),
            api_version=os.getenv("ASSISTANTS_API_VERSION"),
        )
        sync_openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("ASSISTANTS_BASE_URL"),
            api_key=os.getenv("ASSISTANTS_API_KEY"),
            api_version=os.getenv("ASSISTANTS_API_VERSION"),
        )

    cl.instrument_openai()  # Instrument the OpenAI API client

    assistant = sync_openai_client.beta.assistants.retrieve(
        os.environ.get("ASSISTANTS_ID")
    )

    # config.ui.name = assistant.name
    bot_name = os.getenv("ASSISTANTS_BOT_NAME")
    config.ui.name = bot_name

    return assistant, async_openai_client, sync_openai_client


def get_event_handler(cl, assistant_name, sync_openai_client):  # noqa: C901
    """
    Returns an instance of the EventHandler class, which is responsible for handling events in the ChatChainlitAssistant.

    Args:
        cl: The ChatClient instance used for communication with the chat service.
        assistant_name (str): The name of the assistant.
        sync_openai_client: The synchronous OpenAI API client.

    Returns:
        EventHandler: An instance of the EventHandler class.
    """

    class EventHandler(AssistantEventHandler):

        def __init__(self, cl, assistant_name: str, sync_openai_client) -> None:
            """
            Initializes a new instance of the ChatChainlitAssistant class.

            Args:
                assistant_name (str): The name of the assistant.
                sync_openai_client: The synchronous OpenAI API client.


            Returns:
                None
            """
            super().__init__()
            self.current_message: cl.Message = None
            self.current_step: cl.Step = None
            self.current_tool_call = None
            self.current_message_text = ""
            self.assistant_name = assistant_name
            self.cl = cl
            self.sync_openai_client = sync_openai_client

        @override
        def on_message_done(self, message) -> None:
            self.handle_message_completed(message)

        @override
        def on_event(self, event):
            """
            Handles the incoming event and performs the necessary actions based on the event type.

            Args:
                event: The event object containing information about the event.

            Returns:
                None
            """
            print(event.event)
            run_id = event.data.id
            if event.event == "thread.message.created":
                self.current_message = self.cl.Message(content="")
                self.current_message = run_sync(self.current_message.send())
                self.current_message_text = ""
                print("Run started")
            # if event.event == "thread.message.completed":
            #    self.handle_message_completed(event.data, run_id)
            elif event.event == "thread.run.requires_action":
                self.handle_requires_action(event.data, run_id)
            elif event.event == "thread.message.delta":
                self.handle_message_delta(event.data)
            elif event.event == "thread.run.step.completed":
                print("Message done")
            elif event.event == "thread.run.step.delta":
                # TODO Here put code to stream code_interpreter output to the chat.
                # When chainlit openai async supports functions
                pass
            elif event.event == "thread.run.completed":
                print("Run completed")
            else:
                print(json.dumps(str(event.data), indent=4))
                print(f"Unhandled event: {event.event}")

        def handle_message_delta(self, data):
            """
            Handles the message delta data.

            Args:
                data: The message delta data.

            Returns:
                None
            """
            for content in data.delta.content:
                if content.type == "text":
                    content = content.text.value
                    if content is not None:
                        self.current_message_text += content
                        run_sync(self.current_message.stream_token(content))
                elif content.type == "image_file":
                    file_id = content.image_file.file_id
                    image_data = self.sync_openai_client.files.content(file_id)
                    image_data_bytes = image_data.read()
                    png_file = f"{images_loc}{file_id}.png"
                    print(f"Writing image to {png_file}")
                    with open(png_file, "wb") as file:
                        file.write(image_data_bytes)
                        image = cl.Image(path=png_file, display="inline", size="large")
                        print(f"Image: {png_file}")
                        if not self.current_message.elements:
                            self.current_message.elements = []
                            self.current_message.elements.append(image)
                            run_sync(self.current_message.update())
                else:
                    print(f"Unhandled delta type: {content.type}")

        def handle_message_completed(self, message):
            """
            Handles the completion of a message.

            Args:
                message: The message object.
                citations: The citations to be added to the message.

            Returns:
                None
            """

            citations = None

            # Check for citations
            if hasattr(message.content[0], "text"):
                message_content = message.content[0].text
                annotations = message_content.annotations
                citations = []
                if annotations:
                    message_content = message.content[0].text
                    annotations = message_content.annotations
                    for index, annotation in enumerate(annotations):
                        message_content.value = message_content.value.replace(
                            annotation.text, f"[{index}]"
                        )
                    if file_citation := getattr(annotation, "file_citation", None):
                        cited_file = self.sync_openai_client.files.retrieve(
                            file_citation.file_id
                        )
                        citations.append(f"[{index}] {cited_file.filename}")

                    print(message_content.value)
                    content = message_content.value
                    self.current_message.content = content

            # Add footer to self message. We have to start a new message so it's in right order
            # TODO combine streaming with image and footer
            run_sync(self.current_message.update())
            self.current_message = run_sync(
                cl.Message(content="", disable_feedback=True).send()
            )

            word_count = len(self.current_message_text.split())
            if word_count > 10:
                if citations is not None and len(citations) > 0:
                    citations = "; Sources: " + "; ".join(citations)
                else:
                    citations = ""
                run_sync(self.current_message.stream_token(llm_footer + citations))

            run_sync(self.current_message.update())

        def handle_requires_action(self, data, run_id):
            """
            Handles the required action by executing the specified tools and submitting the tool outputs.

            Args:
                data: The data containing the required action information.
                run_id: The ID of the current run.

            Returns:
                None
            """
            tool_outputs = []

            for tool in data.required_action.submit_tool_outputs.tool_calls:
                print(tool)

                function_name = tool.function.name
                function_args = tool.function.arguments

                function_output = run_function(function_name, function_args)

                tool_outputs.append(
                    {"tool_call_id": tool.id, "output": function_output}
                )

            print("TOOL OUTPUTS: ")

            print(tool_outputs)

            # Submit all tool_outputs at the same time
            self.submit_tool_outputs(tool_outputs, run_id)

        def submit_tool_outputs(self, tool_outputs, run_id):
            """
            Submits the tool outputs to the current run.

            Args:
                tool_outputs (list): A list of tool outputs to be submitted.
                run_id (str): The ID of the current run.

            Returns:
                None
            """
            event_handler = get_event_handler(
                cl, self.assistant_name, self.sync_openai_client
            )
            with self.sync_openai_client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.current_run.thread_id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=event_handler,
            ) as stream:
                # Needs this line, or it doesn't work! :)
                for text in stream.text_deltas:
                    print(text)

    event_handler = EventHandler(cl, assistant_name, sync_openai_client)

    return event_handler


def run_function(function_name, function_args):
    """
    Run a function with the given name and arguments.

    Args:
        function_name (str): The name of the function to run.
        function_args (dict): The arguments to pass to the function.

    Returns:
        Any: The output of the function
    """
    if not hasattr(sys.modules[__name__], function_name):
        raise Exception(f"Function {function_name} not found")

    try:
        eval_str = f"{function_name}(**{function_args})"
        print(f"Running function: {eval_str}")
        output = eval(eval_str)

        if isinstance(output, bytes):
            output = output.decode("utf-8")
        print(output)

    except Exception as e:
        print(f"Error running function {function_name}: {e}")
        output = f"{e}"

    return output


def print(*tup):
    """
    Custom print function that logs the output using the logger.

    Args:
        *tup: Variable number of arguments to be printed.

    Returns:
        None
    """
    logger.info(" ".join(str(x) for x in tup))


async def cleanup():
    """
    Clean up the user session.

    Returns:
        None
    """
    # await cl.user_session.clear()
    thread = cl.user_session.get("thread")
    run_id = cl.user_session.get("run_id")
    async_openai_client = cl.user_session.get("async_openai_client")

    if run_id is not None:
        await async_openai_client.beta.threads.runs.cancel(
            thread_id=thread.id, run_id=cl.user_session.get("run_id")
        )
    print("Stopped the run")


# @cl.on_stop
async def on_stop():
    await cleanup()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    """
    Transcribes the given audio file to text using the OpenAI Whisper model.

    Parameters:
        audio_file (str): The path to the audio file.

    Returns:
        str: The transcribed text.

    Raises:
        Any exceptions raised by the OpenAI API.

    """
    async_openai_client = cl.user_session.get("async_openai_client")
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    """
    Uploads a list of files to the OpenAI API.

    Args:
        files (List[Element]): A list of files to be uploaded.

    Returns:
        List[str]: A list of file IDs corresponding to the uploaded files.
    """
    async_openai_client = cl.user_session.get("async_openai_client")
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    """
    Process the given list of files.

    Args:
        files (List[Element]): A list of files to be processed.

    Returns:
        List[dict]: A list of dictionaries containing the file_id and tools for each file.
    """

    code_interpretor_types = ["xlsx", "csv", "json", "txt"]

    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    results = []
    for i in range(len(files)):
        file = files[i]
        file_use = "file_search"
        for ci_type in code_interpretor_types:
            if ci_type in file.path:
                file_use = "code_interpreter"
                break
        results.append(
            {
                "file_id": file_ids[i],
                "tools": [{"type": file_use}],
            }
        )

    return results


@cl.on_chat_start
async def start_chat():
    """
    Starts a chat session with the assistant.

    This function creates a new thread using the OpenAI API and stores the thread ID in the user session for later use.
    It also sends an avatar and a welcome message to the chat.

    Returns:
        dict: The thread object returned by the OpenAI API.
    """

    # Setup clients
    assistant, async_openai_client, sync_openai_client = setup(cl)
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("async_openai_client", async_openai_client)
    cl.user_session.set("sync_openai_client", sync_openai_client)

    # Create a Thread
    thread = await async_openai_client.beta.threads.create()

    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)

    await cl.Message(
        content="Hi. I'm your humanitarian AI assistant.", disable_feedback=True
    ).send()

    cl.user_session.set("chat_history", [])

    return thread


def get_metadata_footer(metadata):
    """
    Set the metadata footer for the response.

    Args:
        metadata (dict): The metadata dictionary.

    Returns:
        str: The metadata footer.
    """

    time_period_str = ""
    if "time_period" in metadata:
        if "start" in metadata["time_period"]:
            time_period_str += f"{metadata['time_period']['start']}"
        if "end" in metadata["time_period"]:
            time = metadata["time_period"]["end"].split("T")[0]
            time_period_str += f" to {time}"
        time_period_str = time_period_str.replace("T00:00:00", "")

    label_map = {
        "attribution": {
            "label": "Attribution",
            "value": f"[Source]({metadata['attribution']})",
        },
        "data_url": {
            "label": "Data URL",
            "value": f"[Raw data]({metadata['data_url']})",
        },
        "time_period": {"label": "Time Period", "value": time_period_str},
    }

    footer = f"""
        {human_footer}"""

    for label in label_map:
        if label in metadata:
            if label_map[label]["value"] != "":
                val = label_map[label]["value"]
                if "()" not in val and len(val) > 5:
                    footer += f"; {val}"

    return footer


def check_memories_recipes(user_input: str, history=[]) -> str:
    """
    Check memories and recipes for a given message, and will display the results.
    The answer is passed back to called, so it can be added as an assistant message
    so it knows what it did!

    Args:
        user_input (str): The user input message.
        history (list): The chat history.

    Returns:
        bool: Whether a memory or recipe was found.
        str: The content of the memory or recipe.

    """

    memory_found = False
    memory_content = None
    memory_response = None
    meta_data_msg = ""

    print("History:")
    print(history)

    memory = call_get_memory_recipe_api(
        user_input, history=str(history), generate_intent="true"
    )
    print("RAW memory:")
    print(memory)

    memory_found_api = memory["memory_found"]
    if memory_found_api == "true":
        memory_found = True

    if memory_found is True:

        result = memory["result"]
        try:
            metadata = json.loads(memory["metadata"])
        except Exception:
            print("metadata already dictionary")
            metadata = memory["metadata"]

        memory_found = True
        elements = []
        msg_text = ""

        # Fix image paths
        print(result["type"] == "image")
        if ".png" in result["file"]:
            png_file = result["file"].split("/")[-1]
            result["file"] = f"{os.getenv('IMAGE_HOST')}/{png_file}"
            image = cl.Image(
                path=f"{images_loc}{png_file}", display="inline", size="large"
            )
            elements.append(image)
        else:
            if result["type"] == "text":
                msg_text = str(result["value"])
                elements.append(cl.Text(name="", content=msg_text, display="inline"))
            elif result["type"] == "number":
                value = result["value"]
                if isinstance(value, str):
                    value = float(value)
                value = "{:,}".format(value)
                msg_text = f"The answer is: **{value}**"
                elements.append(
                    cl.Text(
                        name="",
                        content=msg_text,
                        display="inline",
                    )
                )
            elif result["type"] == "csv":
                data = result["value"]

                # Convert the CSV string to a list of lists
                data = [row.split(",") for row in data.split("\n") if row]

                # TODO, we should present a file download here too
                if len(data) > 50:
                    data = data[:50]
                    data.append(["..."])

                data = str(data)

                elements.append(cl.Text(name="", content=data, display="inline"))

            else:
                raise Exception(f"Unknown result type: {result['type']}")

        memory_content = f"""

            The answer is:

            {result['file']}
            {msg_text}

            Metadata for the answer:
            {memory['metadata']}
        """
        print(memory_content)

        meta_data_msg = get_metadata_footer(metadata)
        # elements.append(cl.Text(name="", content=meta_data_msg, display="inline"))

        memory_response = {}
        memory_response["content"] = ""
        memory_response["elements"] = elements

    return memory_found, memory_content, memory_response, meta_data_msg


async_check_memories_recipes = make_async(check_memories_recipes)


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """
    Process an audio chunk and transcribe the audio stream.

    Args:
        chunk (cl.AudioChunk): The audio chunk to process.

    Returns:
        None
    """
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[Element]):
    """
    Process the audio when it ends.

    Args:
        elements (list[Element]): The list of elements to include in the message.

    Returns:
        None
    """
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    # input_audio_el = cl.Audio(
    #    mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    # )

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        # elements=[input_audio_el, *elements],
    ).send()


# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     """
#     Authenticates a user based on the provided username and password.

#     Args:
#         username (str): The username of the user.
#         password (str): The password of the user.

#     Returns:
#         cl.User or None: If the authentication is successful, returns a User object with the user's identifier, role, and provider. Otherwise, returns None.
#     """
#     if (username, password) == (user, password):
#         return cl.User(
#             identifier=user, metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None


async def add_message_to_thread(thread_id, role, content, message=None):
    """
    Add a message to a thread.

    Args:
        thread_id (str): The ID of the thread.
        role (str): The role of the message author.
        content (str): The content of the message.
        message (cl.Message): The message object.

    Returns:
        None
    """

    async_openai_client = cl.user_session.get("async_openai_client")

    print(f"Content: {content}")

    params = {"thread_id": thread_id, "role": role, "content": content}

    attachments = []

    # Process user-uploaded files
    if message is not None and hasattr(message, "elements"):
        if len(message.elements) > 0:
            attachments = await process_files(message.elements)
            params["attachments"] = attachments

    await async_openai_client.beta.threads.messages.create(**params)


@cl.on_message
async def process_message(message: cl.Message):
    """
    Process the user's message and interact with the assistant.

    Args:
        message (cl.Message): The user's message.

    Returns:
        None
    """

    thread_id = cl.user_session.get("thread_id")
    chat_history = cl.user_session.get("chat_history")
    assistant = cl.user_session.get("assistant")
    sync_openai_client = cl.user_session.get("sync_openai_client")

    chat_history.append({"role": "user", "content": message.content})

    # Add user message to thread
    print(f"Adding user message {message.content} to thread {thread_id}")
    await add_message_to_thread(thread_id, "user", message.content, message)

    # Check memories/recipes
    msg = await cl.Message("").send()
    memory_found, memory_content, memory_response, meta_data_msg = (
        await async_check_memories_recipes(message.content, chat_history)
    )

    if memory_found is True:
        print("Adding memory to thread as assistant")
        await add_message_to_thread(thread_id, "assistant", memory_content)

        # output memory artifacts
        msg.content = memory_response["content"]
        msg.elements = memory_response["elements"]
        await msg.update()

        # TODO really should be part of message above so feedback can apply
        await cl.Message(meta_data_msg).send()

    else:

        # msg.content = "Can't find anything in my memories, let me do some analysis ..."
        msg.content = ""
        await msg.update()

        # Create and Stream a Run to assistant
        print(f"Creating and streaming a run {assistant.id}")
        event_handler = get_event_handler(cl, assistant.name, sync_openai_client)
        with sync_openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant.id,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()

    cl.user_session.set("chat_history", chat_history)
