#
# This Python script is used to tag and push Docker images to Azure Container Registry.
#
# Befor running you'll need to have the Azure command line tools installed and logged in. To log in ..
#
#  az login
#
#  az acr login --name dkcontainerregistryus
#

import os
import sys

import docker
from dotenv import load_dotenv

# client = docker.from_env()
load_dotenv()

container_registry = os.getenv("AZURE_CONTAINER_REGISTRY")
repo = os.getenv("AZURE_CONTAINER_REGISTRY_REPO")

# Script is run from top directory
docker_compose_file = "docker-compose-deploy.yml"
azure_platform = "linux/amd64"

if sys.platform == "darwin":
    print("Running on Mac")
    print(f"container_registry: {container_registry}")
    print(f"repo: {repo}")
    client = docker.DockerClient(
        # base_url="unix:///Users/matthewharris/.docker/run/docker.sock "
        base_url="unix:///Users/t.o./.docker/run/docker.sock "
    )
else:
    client = docker.from_env()


def run_cmd(cmd):
    """
    Executes a command in the shell and prints the command before executing.

    Args:
        cmd (str): The command to be executed.

    Returns:
        None
    """
    print(cmd)
    os.system(cmd)


def deploy():
    """
    Deploys the application to Azure using Docker Compose.

    This function performs the following steps:
    1. Logs into Azure using the 'az login' command.
    2. Logs into the Azure Container Registry using the 'az acr login' command.
    3. Stops and removes any existing containers using the 'docker compose down' command.
    4. Pulls the latest images from the Docker Compose file using the 'docker compose pull' command.
    5. Builds the Docker images using the 'docker compose build' command.
    6. Tags and pushes the Docker images to the Azure Container Registry.
    7. Reverts to the host architecture by stopping and removing containers again.
    8. Pulls the latest images from the Docker Compose file.
    9. Builds and starts the containers using the 'docker compose up -d' command.
    10. Prints the URL to trigger the deployment.

    Note: The variables 'container_registry', 'repo', 'azure_platform', and 'docker_compose_file'
    should be defined before calling this function.
    """

    # if container resistry not set
    if container_registry is None or container_registry == "":
        print("You must set your AZURE_CONTAINER_REGISTER in .env")
        sys.exit()

    tags = {
        "data-recipes-ai-server:latest": [f"{container_registry}/{repo}", "server"],
        "data-recipes-ai-chat:latest": [f"{container_registry}/{repo}", "chat"],
    }

    run_cmd("az login")
    run_cmd(f"az acr login --name {container_registry}")

    run_cmd(f"docker compose -f {docker_compose_file} down")
    run_cmd(
        f"DOCKER_DEFAULT_PLATFORM={azure_platform} && docker compose -f {docker_compose_file} pull"
    )
    run_cmd(
        f"DOCKER_DEFAULT_PLATFORM={azure_platform} && docker compose -f {docker_compose_file} build"
    )

    # run_cmd("docker compose build")

    for image in tags.keys():
        print(f"Tagging {image} image ... with tag {tags[image][0]}:{tags[image][1]}")
        client.images.get(image).tag(tags[image][0], tags[image][1])
        print(f"Pushing {image} image ... to {tags[image][0]}:{tags[image][1]}")
        client.images.push(tags[image][0], tags[image][1])

    # sys.exit()

    run_cmd(f"docker compose -f {docker_compose_file} down")
    run_cmd(f"docker compose -f {docker_compose_file} pull")
    run_cmd(f"docker compose -f {docker_compose_file} build")
    run_cmd(f"docker compose -f {docker_compose_file} up -d")

    print(
        "Now go and click on https://ai-assistants-prototypes.azurewebsites.net/c/new to trigger to deploy"
    )


if __name__ == "__main__":
    deploy()
