import typer
import os
import readline
from typing import Optional
import shutil
import json

user_name = None
recipes = {}
commands_help = f"""
    Here are the commands you can run:
    
    'checkout': Check out recipes for you to work on
    'list': List all recipes that are checked out
    'run': Run a recipe, you will be prompted to choose which one
    'add': Add a new recipe
    'delete': Delete a recipe, you will be prompted to choose which one
    'checkin': Check in recipes you have completed
    'help': Show a list of commands
    'quit': Exit this recipes CLI

    Type one of the commands above to do some stuff.

"""

to_be_deleted_file = 'work/checked_out/to_be_deleted.txt'
cli_config_file = '.cli_config'

def _get_checkout_folders():
    global recipes
    checked_out_dir = 'work/checked_out'

    if not os.path.exists(checked_out_dir):
        checked_out_folders = []
    else:
        checked_out_folders = os.listdir(checked_out_dir)
        checked_out_folders = [folder for folder in checked_out_folders if not folder.endswith('.txt')]
        count = 0
        for folder in checked_out_folders:
            recipes[count] = folder
            count += 1
            
    return checked_out_folders

def _updated_recipes_to_be_deleted(uuid):
    if not os.path.exists(to_be_deleted_file):
        with open(to_be_deleted_file, 'w') as f:
            f.write("")
    with open(to_be_deleted_file, 'a') as f:
        f.write(f"{uuid}\n")

def _get_session_defaults():
    #if .cli_config exists, read it
    if os.path.exists(cli_config_file):
        with open(cli_config_file, 'r') as f:
            config = f.read()
            config = json.loads(config)
            return config
    else:
        return {}

def _update_session_defaults(config):
    with open(cli_config_file, 'w') as f:
        f.write(json.dumps(config))

def checkout():
    cmd = f'docker exec haa-recipe-manager python recipe_sync.py --check_out --recipe_author {user_name} --force_checkout'
    typer.echo(f"Checking out as user {user_name}")
    os.system(cmd)
    list()

def list():
    checked_out_folders = _get_checkout_folders()
    typer.echo(f"\nChecked out recipes:\n")
    count = 0
    for recipe_index, folder in enumerate(checked_out_folders):
        typer.echo(f"{recipe_index + 1}. {folder}")
    typer.echo("\n")

def run(recipe_index: Optional[int] = typer.Argument(None)):
    # If recipe is undefined, show the list of recipes
    if recipe_index is None:
        list()
        # Ask which one to run
        recipe_index = input("Enter the recipe number to run: ")

    if int(recipe_index) - 1 not in recipes:
        typer.echo("Invalid recipe number. Please try again. Type 'list' to see recipes.")
        return

    recipe = recipes[int(recipe_index) - 1] 
    cmd = f"docker exec haa-recipe-manager python work/checked_out/{recipe}/recipe.py"
    typer.echo(f"Running recipe {recipe}")
    os.system(cmd)


def help():
    typer.echo(commands_help)

def checkin():
    cmd = f'docker exec haa-recipe-manager python recipe_sync.py --check_in --recipe_author {user_name}'
    typer.echo(f"Checking in as user {user_name}")
    os.system(cmd)

    # Read to_be_deleted file and delete the recipes
    if os.path.exists(to_be_deleted_file):
        with open(to_be_deleted_file, 'r') as f:
            uuids = f.readlines()
            for uuid in uuids:
                cmd = f"docker exec haa-recipe-manager python recipe_sync.py --delete_recipe --recipe_uuid {uuid}"
                os.system(cmd)
        os.remove(to_be_deleted_file)

def add(intent: Optional[str] = typer.Argument(None)):
    if intent is None:
        intent = input("Enter the intent of your new recipe: ")
    cmd = f"docker exec haa-recipe-manager python recipe_sync.py --create_recipe --recipe_intent '{intent}' --recipe_author '{user_name}'"
    typer.echo(f"Creating new recipe with intent {intent}")
    os.system(cmd)
    list()
    typer.echo(f"Now edit your new recipe in folder ./work/checked_out and when done do a 'checkin'")

def delete():
    list()
    # Ask which one to run
    recipe_index = input("Enter the recipe number to delete: ")
    if int(recipe_index) - 1 not in recipes:
        typer.echo("Invalid recipe number. Please try again. Type 'list' to see recipes.")
        return
    
    recipe = recipes[int(recipe_index) - 1] 

    recipe_folder = f"work/checked_out/{recipe}"

    # Update to_be_deleted
    metadata_file = os.path.join(recipe_folder, 'metadata.json')
    with open(metadata_file, 'r') as f:
        metadata = f.read()
        metadata = json.loads(metadata)
        uuid = metadata['uuid']
        _updated_recipes_to_be_deleted(uuid)

    if os.path.exists(recipe_folder):
        typer.echo(f"Deleting recipe {recipe}")
        shutil.rmtree(recipe_folder)
        typer.echo(f"Run 'checkin' to update recipes DB, or checkout to undelete")

def main():
    global user_name
    app = typer.Typer()
    app.command()(checkout)
    app.command()(list)
    app.command()(checkin)
    app.command()(run)
    app.command()(add)
    app.command()(delete)
    app.command()(help)

    config = _get_session_defaults()

    if 'user_name' in config:
        user_name = config['user_name']
    else:
        user_name = input("Enter your name: ")
        config['user_name'] = user_name
        _update_session_defaults(config)

    welcome_message = f"""\nWelcome to the recipes management CLI, {user_name}!\n"""
    welcome_message += commands_help

    typer.echo(welcome_message)

    # Set resipes list
    _get_checkout_folders()

    while True:
        command = input(">> ")
        if command.lower() == 'quit':
            break
        if command.lower().split()[0] not in ['checkout', 'run', 'checkin', 'list', 'add', 'delete', 'help']:
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