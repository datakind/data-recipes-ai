import asyncio
import inspect
import sys
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar

import chainlit as cl
from promptflow.core import tool


@tool
async def call_assistant(chat_history: list) -> dict:
    """
    Calls the assistant API with the given input and retrieves the response.

    Args:
        chat_history (list): A list containing the chat history, of the format ...

        [
            {
                "author": "user",
                "content": "Hi",
                "elements": []
            },
            {
                "author": "assistant",
                "content": "Hello! How can I help you today?",
                "elements": []
            }
        ]

    Returns:
        dict: A dictionary containing the response from the assistant, function name, function arguments,
              function output, and the number of tokens in the function output.
    """
    # print("Initializing chainlit thread ...")
    # thread = await start_chat()
    # print(thread)


def setup_mock_class():
    """
    Creates and returns a mock class for testing purposes.

    Returns:
        cl_mock (MockChainlit): The mock class instance.
    """

    class MockMessage:
        """
        A class representing a mock message.

        Attributes:
            author (str): The author of the message.
            content (str): The content of the message.
            elements (list): The elements of the message.
            disable_feedback (bool): Flag indicating whether feedback is disabled.

        Methods:
            send(): Sends the message.
            stream_token(content): Streams a token.
            update(): Updates the message.
        """

        def __init__(
            self, author=None, content=None, elements=None, disable_feedback=False
        ):
            if content is None:
                content = ""
            self.author = author
            self.content = content
            self.disable_feedback = disable_feedback
            self.elements = elements if elements is not None else []

        async def send(self):
            """
            Sends the message.

            Returns:
                MockMessage: The sent message.
            """
            print(
                f"Sending message: Author: {self.author}, Content: {self.content}, Elements: {self.elements}"
            )
            return self

        async def stream_token(self, content):
            """
            Streams a token.

            Args:
                content (str): The content of the token.

            Returns:
                MockMessage: The updated message.
            """
            print(f"Streaming token: Author: {self.author}, Content: {content}")
            return self

        async def update(self):
            """
            Updates the message.

            Returns:
                MockMessage: The updated message.
            """
            print(
                f"Updating message: Author: {self.author}, Content: {self.content}, Elements: {self.elements}"
            )
            return self

    class MockUserSession:
        """
        A class representing a mock user session.

        Attributes:
            session_data (dict): A dictionary to store session data.

        Methods:
            get(key): Retrieves the value associated with the given key from the session data.
            set(key, value): Sets the value associated with the given key in the session data.
        """

        def __init__(self):
            self.session_data = {}

        def get(self, key):
            """
            Retrieves the value associated with the given key from the session data.

            Args:
                key (str): The key to retrieve the value for.

            Returns:
                The value associated with the given key, or None if the key is not found.
            """
            return self.session_data.get(key, None)

        def set(self, key, value):
            """
            Sets the value associated with the given key in the session data.

            Args:
                key (str): The key to set the value for.
                value: The value to be associated with the key.
            """
            self.session_data[key] = value

    class MockChainlit:
        """
        A mock implementation of the Chainlit class.
        """

        def __init__(self):
            self.Message = MockMessage
            self.user_session = MockUserSession()
            self.__name__ = "chainlit"
            self.step = None

    cl_mock = MockChainlit()

    return cl_mock


# Method to run a supplied function to override chainlit's run_sync method
def run_async_coroutine(coroutine):
    """
    Runs an asynchronous coroutine in a separate event loop and returns the result.

    Args:
        coroutine: The coroutine to be executed asynchronously.

    Returns:
        The result of the coroutine execution.

    Raises:
        asyncio.TimeoutError: If the coroutine execution times out.

    """

    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    new_loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_loop, args=(new_loop,))
    t.start()
    future = asyncio.run_coroutine_threadsafe(coroutine, new_loop)
    try:
        return future.result(timeout=10)
    except asyncio.TimeoutError:
        print("Coroutine execution timed out.")
        return None


def run_sync(func, *args, **kwargs):
    """
    Run a function synchronously or asynchronously depending on its type.

    Args:
        func: The function to be executed.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Returns:
        The result of the function execution.

    Raises:
        None.

    """
    if inspect.iscoroutinefunction(func):
        # Use the alternative approach for coroutine functions
        coroutine = func(*args, **kwargs)
        return run_async_coroutine(coroutine)
    elif asyncio.iscoroutine(func):
        # Directly pass the coroutine object
        return run_async_coroutine(func)
    else:
        # Handle synchronous function
        return func(*args, **kwargs)


def test():
    """
    This function is used to test the functionality of the app module.
    It sets up a mock chainlit class, imports the app module, and overrides certain methods and event handlers.
    Then it calls the app start and main functions to simulate a chat interaction.
    """

    # Create a Mock chainlit class
    cl_mock = setup_mock_class()

    # Import chainlit app
    import app as app

    # Override run_sync method to use mock cl
    app.run_sync = run_sync

    # Override OpenAI event handler to use mock cl
    event_handler = app.get_event_handler(cl_mock, "assistant")
    cl_mock.user_session.set("event_handler", event_handler)

    # Patch 'cl' with our Mock class
    app.cl = cl_mock

    # Call app start to set up variables
    asyncio.run(app.start_chat())

    # Here insert history to thread

    # msg = cl_mock.Message(author="You", content="What is the total population of Mali", elements=[])
    msg = cl_mock.Message(author="You", content="Hi", elements=[])
    thread = asyncio.run(app.main(msg))
    print(thread)


if __name__ == "__main__":
    test()
