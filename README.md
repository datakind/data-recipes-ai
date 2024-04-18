# Introduction

NOTE: For now we reference internal documents, but this will be adjusted and added to the repo over time.

This repo contains components for the humanitarian AI Assitant developed by DataKind. For more information see [here](https://datakind.atlassian.net/wiki/spaces/TT/pages/187105282/Technical+Summary)

It has the following components:

- [LibraChat](https://docs.librechat.ai/) chat interface
- [Robocorp actions-server](https://github.com/robocorp/robocorp)

Being added soon ....

- Databases
- Data Ingestion Pipeline
- Assistant creation

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
3. Select Assistants, choose Humanitarian AI Assistant (alpha) (Note: As this is still a work in progress, the names might change over time)
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

The environment has been configured to run on a [ai-assistant-prototypes](https://portal.azure.com/#@DataKindO365.onmicrosoft.com/resource/subscriptions/21fe0672-504b-4b05-b7e1-a154142c9fd4/resourceGroups/DK-DS-Prototypes/providers/Microsoft.Web/sites/ai-assistants-prototypes/appServices) multicontainer web app in Azure. This actually isn't a very robust solution, as we shall see below, and should be migrated onto something more formal for launch.

### Not on a Mac

Run ...

`python3 deploy_azure.py`

One thing to mention on an Azure deploy, it that doesn't get pushed to the web app sometimes, until a user tries to access the web app's published URL. No idea why, but if your release is 'stuck', try this.

### On a Mac

Make dure docker has 'Use Rosetta for x86_64/amd64 emulation on Apple Silicon' set in settings if using a MAC silicon chip. The deploy script can then build images that wwok on Azure then revert to images that work on your Mac.

Note: 

`docker-compose-azure.yml` is the configurtation used in the deployment center screen on the web app
`docker-compose.yml` is used for building locally

## Databases

When running in Azure it is sueful to use remote databases, at least for the mongodb instance so that user logins are retained with each release. For example, a databse can be configured by following [these instructions](https://docs.librechat.ai/install/configuration/mongodb.html). If doing this, then docker-compose-azure.yml in Azure can have the mongo DB section removed, and any instance of the Mongo URL used by other containers updated with the cloud connection string accordingly.
