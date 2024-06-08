import sys

from actions import get_memory_recipe
from fastapi import FastAPI
from pydantic import BaseModel


class MemoryRecipeInput(BaseModel):
    user_input: str
    chat_history: str
    generate_intent: str = "true"


app = FastAPI()


@app.post("/get_memory_recipe")
def memory_recipe(data: MemoryRecipeInput):
    return get_memory_recipe(data.user_input, data.chat_history, data.generate_intent)
