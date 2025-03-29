import json
import os
import readline
import shutil
import sys

import pandas as pd
from dotenv import load_dotenv
from recipe_sync import create_new_recipe, llm_validate_recipe

load_dotenv()

input_data = "./tests/humanitarian_user_inputs_short.csv"
work_dir = "./work/checked_out"

env_cmd = " python "
author = "matt"

data = pd.read_csv(input_data)

user_inputs = data["user_input"]

#
# This code will read an input file of user questions,
# automatically generate recipes and have an LLM review the output
#
#


results = []

for input in user_inputs[0:3]:
    print(input)

    input = input + " /nochecks"

    create_new_recipe(input, author)
    print("\n\n")

    # Find most recent directory by timestamp in ./management/work
    dirs = os.listdir(work_dir)
    dirs = sorted(dirs, key=lambda x: os.path.getmtime(f"{work_dir}/{x}"), reverse=True)
    recent_dir = work_dir + "/" + dirs[0] + "/recipe.py"

    validation_result = llm_validate_recipe(input, recent_dir)

    r = {
        "input": input,
        "validation_result": validation_result["answer"],
        "validation_reason": validation_result["reason"],
    }

    results.append(r)

    print("\n\n")

results = pd.DataFrame(results)
results.to_csv("results.csv")
