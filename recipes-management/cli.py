import json
import os
import readline
import shutil
from typing import Optional

import typer

user_name = None
recipes = {}
commands_help = """
    Here are the commands you can run:

    'checkout': Check out recipes for you to work on
    'list': List all recipes that are checked out
    'run': Run a recipe, you will be prompted to choose which one
    'add': Add a new recipe (using LLM)
    'edit': Edit a new recipe (using LLM)
    'delete': Delete a recipe, you will be prompted to choose which one
    'checkin': Check in recipes you have completed
    'makemem': Create a memory using recipe sample output
    'help': Show a list of commands
    'quit': Exit this recipes CLI

    Type one of the commands above to do some stuff.

"""

to_be_deleted_file = "work/checked_out/to_be_deleted.txt"
cli_config_file = ".cli_config"
checked_out_dir = "work/checked_out"


def _get_checkout_folders():
    """
    Retrieves the list of checked out folders.

    Returns:
        list: A list of checked out folders.
    """
    global recipes

    if not os.path.exists(checked_out_dir):
        checked_out_folders = []
    else:
        checked_out_folders = os.listdir(checked_out_dir)
        checked_out_folders = [
            folder
            for folder in checked_out_folders
            if os.path.isdir(os.path.join(checked_out_dir, folder))
        ]
        count = 0
        for folder in checked_out_folders:
            recipes[count] = folder
            count += 1

    return checked_out_folders


def _updated_recipes_to_be_deleted(custom_id):
    """
    Appends the given custom_id to the file containing recipes to be deleted.

    If the file does not exist, it creates a new file and writes an empty string to it.
    Then, it appends the custom_id to the file, followed by a newline character.

    Args:
        custom_id (str): The custom ID of the recipe to be deleted.

    Returns:
        None
    """
    if not os.path.exists(to_be_deleted_file):
        with open(to_be_deleted_file, "w") as f:
            f.write("")
    with open(to_be_deleted_file, "a") as f:
        f.write(f"{custom_id}\n")


def _get_session_defaults():
    """
    Retrieves the session defaults from the .cli_config file, if it exists.

    Returns:
        dict: A dictionary containing the session defaults.
    """
    # if .cli_config exists, read it
    if os.path.exists(cli_config_file):
        with open(cli_config_file, "r") as f:
            config = f.read()
            config = json.loads(config)
            return config
    else:
        return {}


def _update_session_defaults(config):
    """
    Update the session defaults in the CLI configuration file.

    Args:
        config (dict): The updated configuration to be written to the file.

    Returns:
        None
    """
    with open(cli_config_file, "w") as f:
        f.write(json.dumps(config))


def checkout():
    """
    Check out a recipe using the recipe_sync.py script.

    This function executes a Docker command to run the recipe_sync.py script with the `--check_out` flag.
    It also includes the `--recipe_author` and `--force_checkout` options, which are passed as command-line arguments.

    After executing the command, it calls the `list()` function to display the updated list of recipes.

    Note: The `user_name` variable should be defined before calling this function.

    Example usage:
        checkout()

    """
    cmd = f"docker exec haa-recipe-manager python recipe_sync.py --check_out --recipe_author {user_name} --force_checkout"
    typer.echo(f"Checking out as user {user_name}")
    os.system(cmd)
    list()


def list():
    """
    Lists the checked out recipes.

    This function retrieves the list of checked out recipe folders and displays them in the console.

    Parameters:
        None

    Returns:
        None
    """
    checked_out_folders = _get_checkout_folders()
    typer.echo("\nChecked out recipes:\n")
    for recipe_index, folder in enumerate(checked_out_folders):
        typer.echo(f"{recipe_index + 1}. {folder}")
    typer.echo("\n")


def run(recipe_index: Optional[int] = typer.Argument(None)):
    """
    Run a recipe based on the provided recipe index.

    Args:
        recipe_index (Optional[int]): The index of the recipe to run. If not provided, a list of recipes will be shown and the user will be prompted to enter the recipe number.

    Returns:
        None
    """
    # If recipe is undefined, show the list of recipes
    if recipe_index is None:
        list()
        # Ask which one to run
        recipe_index = input("Enter the recipe number to run: ")

    if int(recipe_index) - 1 not in recipes:
        typer.echo(
            "Invalid recipe number. Please try again. Type 'list' to see recipes."
        )
        return

    recipe = recipes[int(recipe_index) - 1]
    cmd = f"docker exec haa-recipe-manager python recipe_sync.py --run_recipe --recipe_path work/checked_out/{recipe}/recipe.py"
    typer.echo(f"Running recipe {recipe} ...\n\n")
    os.system(cmd)


def edit():
    """
    Edit a recipe by interacting with the user and running a command.

    This function prompts the user to enter the recipe number they want to edit.
    If the entered recipe number is invalid, an error message is displayed.
    Otherwise, the function retrieves the recipe based on the entered number.
    Then, the function prompts the user to enter how they would like to adjust/change the recipe.
    Finally, a command is constructed and executed to edit the recipe using the entered prompt.

    Note: This function assumes the existence of a `recipes` dictionary and a `typer` module.

    Args:
        None

    Returns:
        None
    """
    list()
    recipe_index = input("Enter the recipe number to edit: ")

    if int(recipe_index) - 1 not in recipes:
        typer.echo(
            "Invalid recipe number. Please try again. Type 'list' to see recipes."
        )
        return
    recipe = recipes[int(recipe_index) - 1]

    prompt = input("Enter how you would like to adjust/change the recipe: ")

    cmd = f'docker exec haa-recipe-manager python recipe_sync.py --edit_recipe --recipe_path work/checked_out/{recipe}/recipe.py --llm_prompt "{prompt}"    '
    typer.echo(f"Running recipe {recipe}")
    os.system(cmd)


def help():
    """
    Display the help information for the CLI commands.
    """
    typer.echo(commands_help)


def checkin():
    """
    Check in recipes and delete specified recipes.

    This function executes a Docker command to check in recipes as a specific user. It also reads a file containing
    custom IDs of recipes to be deleted, and deletes those recipes using another Docker command.

    Args:
        None

    Returns:
        None
    """
    cmd = f"docker exec haa-recipe-manager python recipe_sync.py --check_in --recipe_author {user_name}"
    typer.echo(f"Checking in as user {user_name}")
    os.system(cmd)

    # Read to_be_deleted file and delete the recipes
    if os.path.exists(to_be_deleted_file):
        with open(to_be_deleted_file, "r") as f:
            custom_ids = f.readlines()
            for custom_id in custom_ids:
                cmd = f"docker exec haa-recipe-manager python recipe_sync.py --delete_recipe --recipe_custom_id {custom_id}"
                os.system(cmd)
        os.remove(to_be_deleted_file)


def add(intent: Optional[str] = typer.Argument(None)):
    """
    Add a new recipe with the specified intent.

    Args:
        intent (str, optional): The intent of the new recipe. If not provided, the user will be prompted to enter it.

    Returns:
        None
    """
    if intent is None:
        intent = input("Enter the intent of your new recipe: ")
    cmd = f"docker exec haa-recipe-manager python recipe_sync.py --create_recipe --recipe_intent '{intent}' --recipe_author '{user_name}'"
    typer.echo(f"Creating new recipe with intent {intent}")
    os.system(cmd)
    list()
    typer.echo(
        "Now 'run' your new recipe and if needed edit its recipe.py (in folder ./work/checked_out) then do a 'checkin'"
    )


def delete():
    """
    Deletes a recipe from the recipe management system.

    This function prompts the user to enter the recipe number to delete. If the provided
    recipe number is invalid, an error message is displayed. Otherwise, the recipe is
    deleted by removing its corresponding folder and updating the 'to_be_deleted' list.

    Note: After deleting a recipe, you need to run 'checkin' to update the recipes
    database or 'checkout' to undelete the recipe.

    Returns:
        None
    """
    list()
    # Ask which one to run
    recipe_index = input("Enter the recipe number to delete: ")
    if int(recipe_index) - 1 not in recipes:
        typer.echo(
            "Invalid recipe number. Please try again. Type 'list' to see recipes."
        )
        return

    recipe = recipes[int(recipe_index) - 1]

    recipe_folder = f"work/checked_out/{recipe}"

    # Update to_be_deleted
    metadata_file = os.path.join(recipe_folder, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = f.read()
        metadata = json.loads(metadata)
        custom_id = metadata["custom_id"]
        _updated_recipes_to_be_deleted(custom_id)

    if os.path.exists(recipe_folder):
        typer.echo(f"Deleting recipe {recipe}")
        shutil.rmtree(recipe_folder)
        typer.echo("Run 'checkin' to update recipes DB, or checkout to undelete")


def makemem():
    """
    Saves memory for a specific recipe.

    This function prompts the user to enter the recipe number and saves memory for the corresponding recipe.
    It checks if the recipe number is valid, and if so, it creates a recipe folder and executes a command to save memory for the recipe.

    Args:
        None

    Returns:
        None
    """
    list()
    # Ask which one to run
    recipe_index = input("Enter the recipe number to save memory for: ")
    if int(recipe_index) - 1 not in recipes:
        typer.echo(
            "Invalid recipe number. Please try again. Type 'list' to see recipes."
        )
        return

    recipe = recipes[int(recipe_index) - 1]

    recipe_folder = f"work/checked_out/{recipe}"

    if os.path.exists(recipe_folder):
        typer.echo(f"Saving memory for recipe: {recipe}")
        typer.echo(f"Saving memory for recipe: {recipe_folder}")
        cmd = f"docker exec haa-recipe-manager python recipe_sync.py --save_as_memory --recipe_path {recipe_folder}"
        os.system(cmd)


def main():
    """
    Entry point function for the recipes management CLI.

    This function initializes the CLI, sets up commands, checks the current directory,
    retrieves user name from configuration or prompts for it, displays a welcome message,
    sets the recipes list, and enters a loop to process user commands.

    Returns:
        None
    """
    global user_name
    app = typer.Typer()
    app.command()(checkout)
    app.command()(list)
    app.command()(checkin)
    app.command()(run)
    app.command()(add)
    app.command()(edit)
    app.command()(delete)
    app.command()(makemem)
    app.command()(help)

    # check cli is running in folder recipes-management
    current_dir = os.getcwd()
    if not current_dir.endswith("recipes-management"):
        typer.echo("Please run the CLI from the recipes-management folder")
        return

    config = _get_session_defaults()

    if "user_name" in config:
        user_name = config["user_name"]
    else:
        user_name = input("Enter your name: ")
        config["user_name"] = user_name
        _update_session_defaults(config)

    welcome_message = f"""\nWelcome to the recipes management CLI, {user_name}!\n"""
    welcome_message += commands_help

    typer.echo(welcome_message)

    # Set resipes list
    _get_checkout_folders()

    while True:
        command = input(">> ")
        if not command.strip():  # Check if command is empty
            continue
        if command.lower() in ["quit", "exit", "stop"]:
            break
        if command.lower().split()[0] not in [
            "checkout",
            "run",
            "checkin",
            "list",
            "add",
            "edit",
            "delete",
            "makemem",
            "help",
        ]:
            typer.echo("Invalid command, type 'list' to see available options.")
            continue
        readline.add_history(command)
        args = command.split()
        try:
            app(args, standalone_mode=False)
        except Exception as e:
            typer.echo(f"Error: {e}")


if __name__ == "__main__":
    main()
