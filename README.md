# Introduction

This repo contains components for the humanitarian AI Assitant developed by DataKind. It has the following components:

- [LibraChat](https://docs.librechat.ai/) chat interface
- [Robocorp actions-server](https://github.com/robocorp/robocorp)

# To start the environment

1. Copy `.env.example` to `.env` and set variables
2. `docker compose down`
3. `docker compose pull`
4. `docker compose up`

# Apps

Chatbot - [http://localhost:3080/](http://localhost:3080/)
Robocorp AI Actions (used for SQL querying), Dashboard - [http://localhost:4001/](http://localhost:4001/)
Robocorp AI Actions API - [http://localhost:3001/](http://localhost:3001/)

# One time setup

TODO: This will be automated, but for now ...

1. Initialize the DB connection by going to [http://localhost:4001/](http://localhost:4001/) and running action `init_postgres_connection` to set Recipes DB in Azure (TO DO will be changed once we finish ingestion folders)
1. Got to  [chat app](http://localhost:3080/) and register a user on the login page
2. Log in
3. Select Assistants, choose HDeXpert SQL
4. Under actions, create a new action and use the function definition from [here](http://localhost:4001/openapi.json). You'll need to remove the comments at the top and change the host to be 'url' in 'servers' to be "http://actions:8080"
5. Save the action
6. Update the agent

Note: You can reset Libre chat by removing contents of `ui/recipes_assistant_chat/data-node/`. This is sometimes neccesary due to a bug in specifying actions.

## Testing connection to actions server

1. `exec -it LibreChat /bin/sh`
2. `curl -X POST -H "Content-Type: application/json" \
    -d '{"dsn": "postgresql://username:password@host:port/database"}' \
    "http://actions:8080/api/actions/postgresql-universal-actions/init-postgres-connection/run"` .... replacing with correct postgres credentials`


## Deploying to Azure

The environment has been configured to run on a multicontainer web app in Azure. This actually isn't a very robust solution, as we shall see below, and should be migrated onto something more formal for launch.

### Not on a Mac

Run ...

`python3 deploy_azure.py`

### On a Mac

Make dure docker has 'Use Rosetta for x86_64/amd64 emulation on Apple Silicon' set in settings if using a MAC silicon chip. You will also need to build the Docker image for platform: linux/amd64 so when pushed to Azure, they'll work. This is a bit tricky ...

1. Make some change that needs to be released
2. `docker compose down -v`
3. `docker compose -f docker-compose-build-azure.yml pull`
4. `docker compose -f docker-compose-build-azure.yml build`
5. `docker compose -f docker-compose-build-azure.yml up -d`

Note, this break the app for running locally, it's just to build images that will run in Azure when using Mac

6. `python3 deploy_azure.py`

Then Revert back, to work on Mac 

7. `docker compose -f docker-compose-build-azure.yml down -v`
8. `docker compose pull`
9. `docker compose build`
10. `docker compose up -d`


