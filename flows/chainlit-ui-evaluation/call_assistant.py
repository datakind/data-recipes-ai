import json
import os
import sys
import time

import requests
from promptflow.core import tool
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from utils.llm import call_llm

# Time interval to poll for results in UI
POLL_TIME = 1

# TIme waiting for response before exiting
TIMEOUT_TIME = 120
RETRY_WAIT = 5

# Web elements used
LOGIN_EMAIL_FIELD = "email"
LOGIN_PASSWORD_FIELD = "password"
LOGIN_BUTTON_XPATH = '//button[contains(., "Continue")]'
CHAT_INPUT_CLASS = "chat-input"
# Used to identify changes
MARKDOWN_BODY_CLASS = "markdown-body"

# Used to extract messages
MESSAGES_CLASS = "message-content"

TMP_IMAGE = "temp.png"
CHAT_URL = os.getenv("CHAT_URL")

IMAGE_SUMMARIZATION_PROMPT = "Summarize the image"

def set_chrome_options() -> Options:
    """Sets chrome options for Selenium.
    Chrome options for headless browser is enabled.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_prefs = {}
    chrome_options.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}
    return chrome_options


def check_element_exists(element, by, value):
    """
    Checks if an element exists on a web page.

    Args:
        driver: The WebDriver instance.
        by: The method used to locate the element (e.g., By.ID, By.XPATH, etc.).
        value: The value used to locate the element.

    Returns:
        True if the element exists, False otherwise.
    """
    try:
        element.find_element(by, value)
        return True
    except NoSuchElementException:
        return False


def poll_page(element_name=MARKDOWN_BODY_CLASS):
    """
    Polls the page for new messages until a new message appears or a timeout occurs.

    Args:
        driver: The WebDriver instance used to interact with the web page.
        element_name (str): The name of the element class to search for new messages.

    Raises:
        status: True if a new message appears, False otherwise.

    Returns:
        None
    """
    markdown_body_elements = driver.find_elements(By.CLASS_NAME, element_name)
    chats = len(markdown_body_elements)

    # Loop waiting for the new message to appear, or timeout
    tot_time = 0
    while len(markdown_body_elements) == chats:
        print(f"         ... {tot_time} s")
        time.sleep(POLL_TIME)
        markdown_body_elements = driver.find_elements(By.CLASS_NAME, element_name)
        tot_time += POLL_TIME
        if tot_time > TIMEOUT_TIME:
            print(f"ERROR: Timed out waiting for new message to appear in {element_name}")
            return False
    
    return True


def get_history():
    """
    Retrieves the chat history from the web driver.

    Args:
        driver: The web driver object.

    Returns:
        A list containing the chat history elements.
    """
    markdown_body_elements = driver.find_elements(By.CLASS_NAME, MESSAGES_CLASS)
    history = []
    for element in markdown_body_elements:
        history.append(element)

    # Remove the first greeting
    history = history[1:]

    return history

def send_message(message, num_tries=0, tot_tries=3):
    """
    Sends a message to the chat box and retrieves the bot's response.

    Args:
        message: The message to send to the chat box.
        num_tries: The number of times to try sending the message

    Returns:
        A list of outputs generated by the bot in response to the message.
    """

    history = get_history()
    len_history_original = len(history)

    try:
        print(f"\nYOU: {message}")
        chat_box = driver.find_element(By.ID, CHAT_INPUT_CLASS)
        chat_box.send_keys(message)
        chat_box.send_keys(Keys.RETURN)
    except Exception as e:
        print(f"Error sending message: {e}")
        if num_tries <= tot_tries:
            print(f"Retrying ... {num_tries}")
            time.sleep(RETRY_WAIT)
            return send_message(message, num_tries + 1, tot_tries)
        else:
            print(f"Failed to send message after {tot_tries} tries")
            return ["ERROR: TIMED OUT SENDING MESSAGE"]

    # Poll for the new message to appear, or timeout
    status = poll_page(MARKDOWN_BODY_CLASS)
    if status is False:
        print("Failed to get response")
        return ["ERROR: TIMED OUT WAITING FOR RESPONSE"]

    history = get_history()
    len_history_new = len(history)

    num_new_outputs = len_history_new - len_history_original

    # Get all the new outputs and output them
    history[-1 * num_new_outputs]
    outputs = []

    for i in range(num_new_outputs - 1, 0, -1):
        record = history[-1 * i]

        if check_element_exists(record, By.TAG_NAME, "img"):
            image = record.find_element(By.TAG_NAME, "img")
            url = image.get_attribute("src")
            output = get_image_summary(url)
        else:
            output = record.text
            
        outputs.append(output)

        # Print the last response
        print(f"\n🤖 BOT: {output}")

    return outputs


def download_image(url, filename):
    """
    Downloads an image from the specified URL and saves it to the specified filename.

    Args:
        url (str): The URL of the image to download.
        filename (str): The name of the file to save the image as.

    Returns:
        None
    """
    response = requests.get(url)
    file = open(filename, "wb")
    file.write(response.content)
    file.close()

    return


def get_image_summary(url):
    """
    Downloads an image from the given URL, calls LLM to summarize the image,
    and returns the generated description.

    Args:
        url (str): The URL of the image to download and summarize.

    Returns:
        str: The LLM-generated description of the image.
    """
    print("         Downloading image ...")
    download_image(url, TMP_IMAGE)
    print("         Calling LLM to summarize ...")
    summary = call_llm("", IMAGE_SUMMARIZATION_PROMPT, image=TMP_IMAGE, debug=True)
    summary = summary["content"]
    summary = (
        f"*AN IMAGE WAS OUTPUT, HERE IS ITS LLM-GENERATED DESCRIPTION* ... {summary}"
    )
    print(summary)
    return summary


def login(num_tries=0, tot_tries=3):
    """
    Logs into the application using the provided driver.

    Args:
        num_tries: The number of times to try logging in.
        tot_tries: The total number of times to try logging in. 

    Returns:
        None
    """

    print("Logging in ...")

    try:
        login_box = driver.find_element(By.NAME, LOGIN_EMAIL_FIELD)
        password_box = driver.find_element(By.NAME, LOGIN_PASSWORD_FIELD)
        login_box.send_keys(os.getenv("USER_LOGIN"))
        password_box.send_keys(os.getenv("USER_PASSWORD"))
        button = driver.find_element(By.XPATH, LOGIN_BUTTON_XPATH)
        button.click()

        time.sleep(5)

        # Check for login success
        if check_element_exists(driver, By.ID, CHAT_INPUT_CLASS):
            print("Login successful")
        else:
            print("Login failed")

    except Exception as e:
        print(f"Error logging in: {e}")
        if num_tries <= tot_tries:
            print(f"Retrying ... {num_tries}")
            time.sleep(RETRY_WAIT)
            return login(num_tries + 1, tot_tries)
        else:
            print(f"Failed to login after {tot_tries} tries")
            return


@tool
def call_assistant(query, chat_history):
    """
    Calls the assistant using the provided user input and history. It will first play the
    messages in the history (as the user) and then send the user input to the assistant.

    Args:
        query (str): The user input to send to the assistant.
        chat_history (str): A JSON list of previous messages sent to the assistant.

    Returns:
        str: The response from the assistant.
    """

    global driver
    driver = webdriver.Chrome(options=set_chrome_options())
    driver.get(CHAT_URL)

    login()

    # First replay history
    chat_history = json.loads(chat_history)
    for message in chat_history:
        send_message(message)

    # Now send the user input
    response = send_message(query)

    if response is None:
        response = "ERROR: No response from assistant"

    return response


if __name__ == "__main__":

    chat_history = [
        "Hello! How can I assist you today?",
        "What is the total population of Mali",
        "plot a line chart of fatalities by month for Chad using HDX data as an image",
        "Plot population pyramids for Nigeria",
        "How many rows does the population table have for Nigeria",
        "Plot f{x}=10"
    ]
    
    #user_input = chat_history[4]
    #print(user_input)
    #user_input="Plot f{x}=10"
    #call_assistant(user_input, "[]")
    #sys.exit()

    # read data.jsonl
    with open("data.jsonl") as f:
        data = f.readlines()

    data_new = []
    for d in data:
        d = json.loads(d)
        user_input = d["query"]
        chat_history = d["chat_history"]
        output = call_assistant(user_input, "[]")
        d["output"] = str(output)
        data_new.append(d)

    with open("data.new.jsonl", "w") as f:
        for d in data_new:
            f.write(json.dumps(d) + "\n")
        print("\n\nReview data.new.jsonl for the output and copy to data.jsonl if satisfied.")

