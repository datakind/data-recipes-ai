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
4. Under actions, create a new action and use the function definition from [here](http://localhost:4001/openapi.json). You'll need to remove the comments at the top and change the host to be 'url' in 'servers' to be "http://actions:3001"
5. Save the action
6. Update the agent

Note: You can reset Libre chat by removing contents of `ui/recipes_assistant_chat/data-node/`. This is sometimes neccesary due to a bug in specifying actions.

