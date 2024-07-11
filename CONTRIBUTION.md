# Contributing to Data Recipes AI

Hi! Thanks for your interest in contributing to Data Recipes AI, we're really excited to see you! In this document we'll try to summarize everything that you need to know to do a good job.

## New contributor guide

To get an overview of the project, please read the [README](README.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.


## Getting started
### Creating Issues

If you spot a problem, [search if an issue already exists](https://github.com/datakind/data-recipes-ai/issues). If a related issue doesn't exist, 
you can open one there, selecting an appropriate issue type.

As a general rule, we don’t assign issues to anyone. If you find an issue to work on, you are welcome to open a PR with a fix.

## Making Code changes

## Setting up a Development Environment

To set up your local development environment for contributing follow the steps
in the paragraphs below.

The easiest way to develop is to run in the Docker environment, see [README](./README.md) for more details. 

### Resetting your environment

If running locally, you can reset your environment - removing any data for your databases, which means re-registration - by running `./cleanup.sh`.

## Code quality tests

The repo has been set up with black and flake8 pre-commit hooks. These are configured in the ``.pre-commit-config.yaml` file and initialized with `pre-commit autoupdate`.

On a new repo, you must run `pre-commit install` to add pre-commit hooks.

To run code quality tests, you can run `pre-commit run --all-files`

GitHub has an action to run the pre-commit tests to ensure code adheres to standards. See folder `'github/workflows` for more details.

## Tests

### Unit tests

You should write tests for every feature you add or bug you solve in the code.
Having automated tests for every line of our code lets us make big changes
without worries: there will always be tests to verify if the changes introduced
bugs or lack of features. If we don't have tests we will be blind and every
change will come with some fear of possibly breaking something.

For a better design of your code, we recommend using a technique called
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development),
where you write your tests **before** writing the actual code that implements
the desired feature.

You can use `pytest` to run your tests, no matter which type of test it is.

### End-to-end tests (using Selenium and Promptflow)

End-to-end tests have been configured in GitHub actions which use promptflow to call a wrapper around the chainlit UI, or order to test when memories/recipes are used as well as when the assistant does some on-the-fly analysis. To do this, the chainlit class is patched heavily, and there are limitations in how cleanly this could be done, so it isn't an exact replica of the true application, but does capture changes with the flow as well as test the assistant directly. The main body of integration tests will test recipes server and the assistant independently.

Additionally, there were some limitation when implementing in GitHub actions where workarounds were implemented
until a lter data, namely: promptflow is run on the GitHub actions host rather than in docker, and the promptflow wrapper to call chainlit has to run as a script and kill the script based on a STDOUT string. These should be fixed in future.

Code for e2e tests can be found in `flows/chainlit-ui-evaluation` as run by `.github/workflows/e2e_tests.yml`

The tests work using promptflow evaluation and a call to an LLM to guage groundedness, due to the fact LLM assistants can produce slightly different results if not providing answers from memory/recipes. The promptflow evaluation test data can be found in `flows/chainlit-ui-evaluation/data.jsonl`. 

See "Evaluating with Promptflow" below to see how to run e2e tests locally.

#### Running Promptflow evaluation locally

First, you will need to build the environment to include Prompt Flow ...

`docker compose -f docker-compose.yml -f docker-compose-dev.yml up -d --build`

Then ...

1. Install the DevContainers VSCode extension 
2. Build data recipes using the `docker compose` command mentioned above
3. Open the command palette in VSCode (CMD + Shift + P on Mac; CTRL + Shift + P on Windows) and select 

   `Dev Containers: Attach to remote container`. 

   Select the promptflow container. This opens a new VSCode window - use it for the next steps.
4. Install Promptflow add-in
5. Open folder `/app`
6. Click on `flow.dag.yaml`
7. Top left of main pane, click on 'Visual editor'
     - If you are taken to the promptflow 'Install dependencies'' screen, change the Python runtime to be ` /azureml-envs/prompt-flow/runtime/bin/python` 'runtime', then close and re-open `flow.dag.yaml`
8. On the Groundedness node, select your new connection
9. You can no run by clicking the play icon. See Promptflow documentation for more details

#### Changing between Azure OpenAI <> OpenAI

As noted in the README, the repo supports assistants on OpenAI or Azure OpenAI. The README has instructions on how to change in the `.env` file, but you will also have to change the connection in the promptflow groundedness node accordingly.

## GitHub Workflow

As many other open source projects, we use the famous
[gitflow](https://nvie.com/posts/a-successful-git-branching-model/) to manage our
branches.

Summary of our git branching model:

- Get all the latest work from the upstream repository (`git checkout main`)
- Create a new branch off with a descriptive name (for example:
  `feature/new-test-macro`, `bugfix/bug-when-uploading-results`). You can
  do it with (`git checkout -b <branch name>`)
- Make your changes and commit them locally  (`git add <changed files>>`,
  `git commit -m "Add some change" <changed files>`). Whenever you commit, the self-tests 
  and code quality will kick in; fix anything that gets broken
- Push to your branch on GitHub (with the name as your local branch:
  `git push origin <branch name>`). This will output a URL for creating a Pull Request (PR)
- Create a pull request by opening the URL a browser. You can also create PRs in the GitHub
  interface, choosing your branch to merge into main
- Wait for comments and respond as-needed
- Once PR review is complete, your code will be merged. Thanks!!

### Tips

- Write [helpful commit
  messages](https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message)
- Anything in your branch must have no failing tests. You can check by looking at your PR
  online in GitHub
- Never use `git add .`: it can add unwanted files;
- Avoid using `git commit -a` unless you know what you're doing;
- Check every change with `git diff` before adding them to the index (stage
  area) and with `git diff --cached` before committing;
- If you have push access to the main repository, please do not commit directly
  to `dev`: your access should be used only to accept pull requests; if you
  want to make a new feature, you should use the same process as other
  developers so your code will be reviewed.

## Code Guidelines

- Use [PEP8](https://www.python.org/dev/peps/pep-0008/);
- Write tests for your new features (please see "Tests" topic below);
- Always remember that [commented code is dead
  code](https://www.codinghorror.com/blog/2008/07/coding-without-comments.html);
- Name identifiers (variables, classes, functions, module names) with readable
  names (`x` is always wrong);
- When manipulating strings, we prefer either [f-string
  formatting](https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals)
  (f`'{a} = {b}'`) or [new-style
  formatting](https://docs.python.org/library/string.html#format-string-syntax)
  (`'{} = {}'.format(a, b)`), instead of the old-style formatting (`'%s = %s' % (a, b)`);
- You will know if any test breaks when you commit, and the tests will be run
  again in the continuous integration pipeline (see below);

# Demo Data

The quick start instructions and self-tests require demo data in the data db. This can be downloaded from Google drive.

## Uploading new demo data

To upload new demo data ...

1. Run the ingestion (see main README)
2. In the data directory, `tar -cvf datadb-<DATE>.tar ./datadb` then `gzip datadb-<DATE>.tar`
3. Upload file to [this folder](https://drive.google.com/drive/folders/1E4G9HM-QzxdXVNkgP3fQXsuNcABWzdus?usp=drive_link)
4. Edit `data/download_demo_data.py` to use file URL

## Downloading demo data

To download demo data ...

1. `docker compose stop datadb`
2. `cd data && python3 download_demo_data.py && cd ..`
3. `docker compose start datadb` 

# Evaluation with Prompt Flow

First, you will need to build the environment to include Prompt Flow ...

`docker compose -f docker-compose.yml -f docker-compose-dev.yml up -d --build`

Then ...

1. Install the DevContainers VSCode extension 
2. Build data recipes using the `docker compose` command mentioned above
3. Open the command palette in VSCode (CMD + Shift + P on Mac; CTRL + Shift + P on Windows) and select 

   `Dev Containers: Attach to remote container`. 

   Select the promptflow container. This opens a new VSCode window - use it for the next steps.
4. Install Promptflow add-in
5. Open folder `/app`
6. Click on `flow.dag.yaml`
7. Top left of main pane, click on 'Visual editor'
8. On the Groundedness node, select your new connection
9. You can no run by clicking the play icon. See Promptflow documentation for more details

# Deployment

We will add more details here soon, for now, here are some notes on Azure ...

## Deploying to Azure

A deployment script './deployment/deploy_azure.py' is provided to deploy to an Azure Multicontainer web app you have set up with [these instructions](https://learn.microsoft.com/en-us/azure/app-service/tutorial-multi-container-app). The script is run from the top directory. Note: This is for demo purposes only, as Multicontainer web app are still in Public Preview. 

To run the deployment ...

`python3 ./deployment/deploy_azure.py`

One thing to mention on an Azure deploy, it doesn't get pushed to the web app sometimes, until a user tries to access the web app's published URL. No idea why, but if your release is 'stuck', try this.

Note: 

- `./deployment/./deployment/docker-compose-azure.yml` is the configuration used in the deployment center screen on the web app
- `./deployment/./deployment/docker-compose-deploy.yml` is the configuration used when building the deployment
- `docker-compose.yml` is used for building locally

:warning: *This is very much a work in progress, deployment will be automated with fewer compose files soon*

You will need to set key environment variables, see your local `.env` for examples.
