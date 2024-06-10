import ast
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import List

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from literalai.helper import utc_now
from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI

from utils.general import call_execute_query_api, call_get_memory_recipe_api

environment = Environment(loader=FileSystemLoader("./templates/"))
chat_ui_assistant_prompt_template = environment.get_template(
    "chat_ui_assistant_prompt.jinja2"
)

logging.basicConfig(filename="output.log", level=logging.DEBUG)
logger = logging.getLogger()

load_dotenv("../../.env")

images_loc = "./public/images/"

async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

assistant = sync_openai_client.beta.assistants.retrieve(
    os.environ.get("OPENAI_ASSISTANT_ID")
)

config.ui.name = assistant.name


class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        """
        Initializes a new instance of the ChatChainlitAssistant class.

        Args:
            assistant_name (str): The name of the assistant.

        Returns:
            None
        """
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_text_created(self, text) -> None:
        """
        Handles the event when a new text is created.

        Args:
            text: The newly created text.

        Returns:
            None
        """
        self.current_message = await cl.Message(
            author=self.assistant_name, content=""
        ).send()

    async def on_text_delta(self, delta, snapshot):
        """
        Handles the text delta event.

        Parameters:
        - delta: The text delta object.
        - snapshot: The current snapshot of the document.

        Returns:
        - None
        """
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        """
        Callback method called when text input is done.

        Args:
            text (str): The text input provided by the user.

        Returns:
            None
        """
        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        """
        Callback method called when a tool call is created.

        Args:
            tool_call: The tool call object representing the created tool call.
        """
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool")
        self.current_step.language = "python"
        self.current_step.created_at = utc_now()
        await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot):
        """
        Handles the tool call delta event.

        Args:
            delta (ToolCallDelta): The delta object representing the tool call event.
            snapshot (Snapshot): The snapshot object representing the current state.

        Returns:
            None
        """
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = utc_now()
            await self.current_step.send()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(name=delta.type, type="tool")
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)

    async def on_tool_call_done(self, tool_call):
        """
        Callback method called when a tool call is done.

        Args:
            tool_call: The tool call object representing the completed tool call.

        Returns:
            None
        """
        self.current_step.end = utc_now()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        """
        Callback function called when an image file is done processing.

        Args:
            image_file: The image file object that has finished processing.

        Returns:
            None
        """
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id, content=response.content, display="inline", size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


def print(*tup):
    """
    Custom print function that logs the output using the logger.

    Args:
        *tup: Variable number of arguments to be printed.

    Returns:
        None
    """
    logger.info(" ".join(str(x) for x in tup))


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
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def check_memories_recipes(user_input: str, history=[]) -> str:
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

    found_memory = False
    memory_content = None

    memory = await call_get_memory_recipe_api(
        user_input, history=str(history), generate_intent="true"
    )
    print(memory)
    memory = memory.decode("utf-8")

    if "memory_type" in memory:

        found_memory = True
        elements = []
        msg_text = ""

        # TODO SHouldn't need two json.loads, fix this
        memory = json.loads(json.loads(memory))

        # TODO Fix this in the recipe server
        try:
            memory["result"] = ast.literal_eval(memory["result"])
        except Exception as e:
            print(e)
            print("Error converting memory result to dict")
            print(memory["result"]["answer"])

        # Fix image paths
        if ".png" in memory["result"]["answer"]:
            print(memory)
            png_file = memory["result"]["answer"].split("/")[-1]
            memory["result"]["answer"] = f"{os.getenv('IMAGE_HOST')}/{png_file}"
            # TODO Testing, remove this line
            memory["result"][
                "answer"
            ] = f"http://localhost:8000/public/images/{png_file}"

            image = cl.Image(
                path=f"{images_loc}{png_file}", display="inline", size="large"
            )
            elements.append(image)
        else:
            msg_text = memory["result"]["answer"]

        memory_content = f"""

            The answer is:
            {memory['result']['answer']}

            ***

            Metadata for the answer:
            {memory['metadata']}
        """
        print(memory_content)
        await cl.Message(
            content=msg_text,
            elements=elements,
        ).send()

    return found_memory, memory_content


async def upload_files(files: List[Element]):
    """
    Uploads a list of files to the OpenAI API.

    Args:
        files (List[Element]): A list of files to be uploaded.

    Returns:
        List[str]: A list of file IDs corresponding to the uploaded files.
    """
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
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}],
        }
        for file_id in file_ids
    ]


@cl.on_chat_start
async def start_chat():
    """
    Starts a chat session with the assistant.

    This function creates a new thread using the OpenAI API and stores the thread ID in the user session for later use.
    It also sends an avatar and a welcome message to the chat.

    Returns:
        None
    """
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    await cl.Avatar(name=assistant.name, path="./public/logo.png").send()
    await cl.Message(
        content=f"Hello, I'm {assistant.name}!", disable_feedback=True
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Process the user's message and interact with the assistant.

    Args:
        message (cl.Message): The user's message.

    Returns:
        None
    """
    thread_id = cl.user_session.get("thread_id")

    attachments = await process_files(message.elements)

    # Add a Message to the Thread
    await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
    )

    found_memory, memory_content = await check_memories_recipes(message.content)

    # Message to the thread. If a memory add it as the assistant
    if found_memory is True:
        await async_openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=memory_content,
            # attachments=attachments,
        )

        # No need to send anything
        return

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()


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

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg)
