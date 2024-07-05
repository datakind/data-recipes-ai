import json

import pytest

from utils.general import call_execute_query_api, call_get_memory_recipe_api


# load json file into variable and print it
@pytest.fixture
def get_test_cases():
    """
    Loads test cases from test_cases_get_memory.json.
    """
    with open("test_cases_get_memory.json") as f:
        test_data = json.load(f)
        return test_data


def test_get_memory_recipe(get_test_cases):
    """
    Tests the get memory recipe API endpoint.
    """

    for test in get_test_cases.get("tests", []):
        user_input = test["user_input"]
        chat_history = test["chat_history"]
        generate_intent = test["generate_intent"]
        expected_output = test["expected_output"]
        response = call_get_memory_recipe_api(user_input, chat_history, generate_intent)
        assert response == expected_output
