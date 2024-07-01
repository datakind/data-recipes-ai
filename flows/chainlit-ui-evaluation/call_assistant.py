import argparse
import asyncio
import inspect
import json
import os
import signal
import subprocess
import threading
import time

from promptflow.core import tool

FINISH_PHRASE = "all done"
OUTPUT_TAG = "ASSISTANT_OUTPUT"


@tool
# async def call_assistant(chat_history: list) -> dict:
def call_assistant(query: str, chat_history: str) -> dict:
    """
    Calls the assistant API with the given input and retrieves the response.

    Args:
        query: What the user asked
        chat_history (list): A list containing the chat history, of the format ...

        [
            {
                "author": "user",
                "content": "Hi"
            },
            {
                "author": "assistant",
                "content": "Hello! How can I help you today?",
            },
            {
                "author": "assistant",
                "content": "What's the total population of Mali?",
            }
        ]

    Returns:
        dict: A dictionary containing the response from the assistant, function name, function arguments,
              function output, and the number of tokens in the function output.
    """

    print(chat_history)

    chat_history = json.loads(chat_history)

    # Add user query to chat history
    chat_history.append({"author": "user", "content": query})

    # chat_history = [
    #     {"author": "user", "content": "Hi"},
    #     {
    #         "author": "assistant",
    #         "content": "Hello! How can I help you today?",
    #     },
    #     {
    #         "author": "assistant",
    #         "content": "Hi again!",
    #     },
    # ]

    chat_history = json.dumps(chat_history)
    chat_history = chat_history.replace('"', '\\"')
    chat_history = chat_history.replace("'", "\\'")

    print("History:", chat_history)

    result = run_chainlit_mock(chat_history)

    response = {"response": result}

    return response


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
            # print(f"Streaming token: Author: {self.author}, Content: {content}")
            self.content += content
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

        def Text(self, name, content, display):
            """
            Creates a text element.

            Args:
                text (str): The text content.

            Returns:
                dict: A dictionary containing the text element.
            """
            return {"type": "Text", "text": content}

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


def run_chainlit_mock(chat_history: str) -> str:
    """
    This function is used to run the chainlit script and monitor its output.
    TODO It is a temporary workaround because running the exact chainlit code
    does not exit all asynchronous threads and hangs. This workaround is temporary
    and should be replaced by breaking e2e testing into data recipes API and
    the assistant. Testing both independently is way less complicated.

    Args:
        chat_history (str): A string containing the chat history.

    Returns:
        result (str): The result of the chainlit script running with input history

    """

    all_output = ""
    result = ""
    print("Monitoring chainlit output")
    process = subprocess.Popen(
        ["python3", "call_assistant.py", "--chat_history", chat_history],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(process)
    while True:
        print("Errors:", process.stderr.readline())
        output = process.stdout.readline()
        print(output)
        print("Process poll:", process.poll())
        if output == b"" and process.poll() is not None:
            print(
                "Process finished with No output, try running call_assistant by hand to debug."
            )
            break
        if output:
            all_output += output.decode("utf-8")
            print(output.strip())
            if FINISH_PHRASE in str(output).lower():
                print(FINISH_PHRASE)
                print("Killing process")
                os.kill(process.pid, signal.SIGKILL)
                print(OUTPUT_TAG)
                if OUTPUT_TAG in all_output:
                    result = all_output.split(OUTPUT_TAG)[1].strip()
                    print("Result:", result)
                else:
                    result = "Unparsable output"
                break
        time.sleep(0.1)
    return result


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


async def test_using_app_code_async(chat_history, timeout=5):

    cl_mock = setup_mock_class()
    import app as app

    app.run_sync = run_sync
    app.cl = cl_mock

    await app.start_chat()

    thread_id = app.cl.user_session.get("thread_id")

    # Here build history
    chat_history = chat_history.replace("\\", "")
    print(">>>>>>>> Chat history:", chat_history)
    history = json.loads(chat_history)
    last_message = history[-1]
    app_chat_history = app.cl.user_session.get("chat_history")
    for message in history:
        role = message["author"]
        msg = message["content"]
        await app.add_message_to_thread(thread_id, role, msg)
        app_chat_history.append({"role": role, "content": msg})
    app.cl.user_session.set("chat_history", history)

    print("<<<<<<<< Last message:", last_message)

    msg = cl_mock.Message(author="user", content=last_message["content"], elements=[])
    await app.process_message(msg)

    messages = app.sync_openai_client.beta.threads.messages.list(thread_id)
    result = messages.data[0].content[0].text.value

    return result


def test_using_app_code(chat_history):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_using_app_code_async(chat_history))
    loop.close()
    return result


def main_direct_function():
    """
    TODO
    For testing direct function call, which hangs even though finished because of
    some issue with async. Left here for future reference for somebody to fix so
    the script execution and kill hack can be retired.

    """
    # chat_history = '[{\"author\": \"user\",\"content\": \"Hi\"},{\"author\":\"assistant\content\": \"Hello! How can I help you today?\"},{\"author\": \"assistant\",\"content\": \"What is the total population of Mali?\"}]'
    chat_history = '[{"author": "user","content": "Hi"}'

    result = test_using_app_code(chat_history)
    print("OUTPUT")
    print(result)
    print("OUTPUT")


def main():

    parser = argparse.ArgumentParser(
        description="Process check in and check out operations (i.e. extracting recipes and recipes from the database for quality checks and edits)."
    )

    parser.add_argument(
        "--chat_history",
        type=str,
        required=True,
        help="""
            A list containing the chat history, of the format (but in one line) ...

            '[{\"author\": \"user\",\"content\": \"Hi\"},{\"author\":\"assistant\",\"content\": \"Hello! How can I help you today?\"},{\"author\": \"assistant\",\"content\": \"What is the total population of Mali?\"}]'
        """,
    )

    args = parser.parse_args()
    chat_history = args.chat_history

    if chat_history:
        result = test_using_app_code(chat_history)
        print(OUTPUT_TAG)
        print(result)
        print(OUTPUT_TAG)

        # Do not remove this line
        print(FINISH_PHRASE)


if __name__ == "__main__":
    # main_direct_function()
    main()
