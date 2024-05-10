# Data Recipes AI

**Please note**: This repo is being actively developed and is in a pre-release mode. Expect lots of changes and updates!

Data recipes AI is a Large Language Model-powered platform for creating a library of data analysis 'Recipes' people can use in a chat interface ... 

![alt text](./assets/data-recipes-concept.png)

By building a library, certain risks of using LLMs are reduced because data recipes will give exactly the same result every time and can be reviewed to ensure the answer is right. It's a way of using the awesome power of LLMs in a safer and more cost effective way. As an extension of [Large Language Models as Tool makers (LATM)](https://arxiv.org/abs/2305.17126), data recipes ai extends this to include memory management as well as human-in-the-loop review concept. 

# Design Concepts

Given the rapidly changing landscape of LLMs, we have tried as much as possible to implement data recipes in such as way that it can be intergrated with various 

# What's in this repo?

![alt text](./assets/system.png)

This repo contains the following main components:



NOTE: For now we reference internal documents, but this will be adjusted and added to the repo over time.

This repo contains components for the humanitarian AI Assitant developed by DataKind. For more information see [here](https://datakind.atlassian.net/wiki/spaces/TT/pages/187105282/Technical+Summary)

It has the following components:

- [LibreChat](https://docs.librechat.ai/) chat interface
- [Robocorp actions-server](https://github.com/robocorp/robocorp)
- Databases
- Data Ingestion Pipeline
- Assistant creation

# Design principals

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

1. Got to  [chat app](http://localhost:3080/) and register a user on the login page
2. Select "Plugins' endpoint at top, then in the plugin box go to store and activate 
   - Humanitariuan Data Assistant
   - Humanitarian Data Recipes
   - Code sherpa, when asked enter URL http://code-interpretor:3333
3. Populate system prompts, see `./assistant/recipes_assistant/prompts`, build presets
2. Log in
3. `docker exec -it haa-ingestion /bin/bash`
4. `python3 ingest.py`
5. Select Assistants, choose HDeXpert SQL
6. Under actions, create a new action and use the function definition from [here](http://localhost:4001/openapi.json). You'll need to remove the comments at the top and change the host to be 'url' in 'servers' to be "http://actions:8080"
7. Save the action
8. Update the agent

>>>>>> GOT TO HAPI TO GET KEY

Note: You can reset Libre chat by removing contents of `ui/recipes-chat/data-node/`. This is sometimes neccesary due to a bug in specifying actions.

## managing Assistants

You can create new Azure OpenAI or OpenAI assistants as follows:

<TO DO Add instructions on create_update_assistant.py and importing >

## Reseting your environment

If running locally, you can reset your environment - removing any data for your databases, which means re-registration - by running `./cleanuop.sh`.

## Testing connection to actions server

1. `docker exec -it haa-libre-chat  /bin/sh`
2. To test the SQL query action, run `curl -X POST -H "Content-Type: application/json" \
    -d '{"query": "select 1"}' \
    "http://actions:8080/api/actions/postgresql-universal-actions/execute-query/run"`
3. To get get-memory action, run ... `curl -X POST -H "Content-Type: application/json" \
    -d '{"chat_history": "[]", "user_input":"population of Mali", "generate_intent":"true"}' \
    "http://actions:8080/api/actions/get-data-recipe-memory/get-memory/run"``

## LibreChat Plugins

With a defined set of functionalities, [plugins](https://docs.librechat.ai/features/plugins/introduction.html) act as tools for the LLM application to use and extend their capabilities.

To create an additional plugin, perform the following steps:
1. Create a new robocorp action in a new folder under [actions_plugins](./actions/actions_plugins/). You can reference the [recipe-server](./actions/actions_plugins/recipe-server/) action to see the relevant files, etc. 
2. Create OpenAPI specification to describe your enpoint. The openapi.json file of the robocorps actions (available on localhost:3001 when the containers are up and running) should contain all necessary information of the endpoint and can be easily converted into a openapi.yaml file. For local development, the open api spec file has to be added to the [openapi directory](./ui/recipes-chat/tools/.well-known/openapi/) and can then be referenced as the url in the manifest. You can use the [haa_datarecipes.yaml](./ui/recipes-chat/tools/.well-known/openapi/haa_datarecipes.yaml) as a template. Please note that robocorp expects inputs to the actions in the body of the API call. For the docker setup, the url can be set to http://actions:8080 as all containers are running in the same network, but this has to be adjusted for the production environment. 
3. Create plugin manigest to describe the plugin for the LLM to determine when and how to use it. You can use [haa_datarecipes.json](./ui/recipes-chat/tools/haa_datarecipes.json) as a template 

As the robocorp actions might differ slightly, this can lead to differing requirements in the openapi spec, and manifest files. The [LibraChat documentation](https://docs.librechat.ai/features/plugins/chatgpt_plugins_openapi.html) provides tips and examples to form the files correctly. 

## Deploying to Azure

The environment has been configured to run on a [ai-assistant-prototypes](https://portal.azure.com/#@DataKindO365.onmicrosoft.com/resource/subscriptions/21fe0672-504b-4b05-b7e1-a154142c9fd4/resourceGroups/DK-DS-Prototypes/providers/Microsoft.Web/sites/ai-assistants-prototypes/appServices) multicontainer web app in Azure. This actually isn't a very robust solution, as we shall see below, and should be migrated onto something more formal for launch.

### Not on a Mac

Run ...

`python3 ./deployment/deploy_azure.py`

One thing to mention on an Azure deploy, it that doesn't get pushed to the web app sometimes, until a user tries to access the web app's published URL. No idea why, but if your release is 'stuck', try this.

### On a Mac

Make sure docker has 'Use Rosetta for x86_64/amd64 emulation on Apple Silicon' set in settings if using a MAC silicon chip. The deploy script can then build images that wwok on Azure then 
revert to images that work on your Mac.

Note: 

`./deployment/./deployment/docker-compose-azure.yml` is the configutation used in the deployment center screen on the web app
`./deployment/./deployment/docker-compose-deploy.yml` is the configutation used when building the deployment
`docker-compose.yml` is used for building locally

TODO: The azure versions will be refactored in future to leverage the main `docker-compose.yml` and even retird for the production infratsructure.

## Databases

When running in Azure it is useful to use remote databases, at least for the mongodb instance so that user logins are retained with each release. For example, a databse can be configured by following [these instructions](https://docs.librechat.ai/install/configuration/mongodb.html). If doing this, then docker-compose-azure.yml in Azure can have the mongo DB section removed, and any instance of the Mongo URL used by other containers updated with the cloud connection string accordingly.