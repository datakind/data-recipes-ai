import typer
import os
import readline
from typing import Optional

user_name = None
recipes = {}
commands_help = f"""
    Here are the commands you can run:
    
    'checkout': Check out recipes for you to work on
    'list': List all recipes that are checked out
    'run': Run a recipe, you will be prompted to choose which
    'checkin': Check in recipes you have completed
    'help': Show a list of commands
    'quit': Exit this recipes CLI

    Type one of the commands above to do some stuff.

"""

def _get_checkout_folders():
    global recipes
    checked_out_dir = 'work/checked_out'
    checked_out_folders = os.listdir(checked_out_dir)
    count = 0
    for folder in checked_out_folders:
        recipes[count] = folder
        count += 1
    return checked_out_folders

def checkout():
    cmd = f'docker exec haa-recipe-manager python recipe_sync.py --check_out "{user_name}" --force_checkout'
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

    # Your checkin logic here
    typer.echo(f"Checked in ")

def main():
    global user_name
    app = typer.Typer()
    app.command()(checkout)
    app.command()(list)
    app.command()(checkin)
    app.command()(run)
    app.command()(help)

    user_name = input("Enter your name: ")

    welcome_message = f"""\nWelcome to the recipes management CLI, {user_name}!\n"""
    welcome_message += commands_help

    typer.echo(welcome_message)

    # Set resipes list
    _get_checkout_folders()

    while True:
        command = input(">> ")
        if command.lower() == 'quit':
            break
        if command.lower().split()[0] not in ['checkout', 'run', 'checkin', 'list', 'help']:
            typer.echo("Invalid command. Please try again.")
            continue
        readline.add_history(command)
        args = command.split()
        try:
            app(args, standalone_mode=False)
        except Exception as e:
            typer.echo(f"Error: {e}")

if __name__ == "__main__":
    main()