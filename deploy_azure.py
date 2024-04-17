#
# This Python script is used to tag and push Docker images to Azure Container Registry.
#
# Befor running you'll need to have the Azure command line tools installed and logged in. To log in ..
#
#  az login
#
#  az acr login --name dkcontainerregistryus
#

import docker
import re
import sys
import os

#client = docker.from_env()
# On Mac, see 'docker context ls'
client = docker.DockerClient(base_url='unix:///Users/matthewharris/.docker/run/docker.sock ')

container_registry = "dkdsprototypesreg01.azurecr.io"
tags = {
    "ankane/pgvector:latest":     [f"{container_registry}/containergroup","vectordb"],
    "getmeili/meilisearch:v1.7.3":  [f"{container_registry}/containergroup","meilisearch"],
    "mongo":      [f"{container_registry}/containergroup","mongodb"],
    "humanitarian_ai_assistant-actions":      [f"{container_registry}/containergroup","actions"],
    "ghcr.io/danny-avila/librechat-rag-api-dev-lite:latest":      [f"{container_registry}/containergroup","rag_api"],
    "humanitarian_ai_assistant-api":              [f"{container_registry}/containergroup","api"],
}
docker_compose_file = "docker-compose.yml"
azure_platform = "linux/amd64"

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

# Log into Azure, the first will open a browser window to log in
run_cmd("az login")
run_cmd(f"az acr login --name {container_registry}")

# Reuild assuming Azure platform
run_cmd(f"docker compose -f {docker_compose_file} down")
run_cmd(f"DOCKER_DEFAULT_PLATFORM={azure_platform} && docker compose -f {docker_compose_file} pull")
run_cmd(f"DOCKER_DEFAULT_PLATFORM={azure_platform} && docker compose -f {docker_compose_file} build")

for image in tags.keys():
    print(f"Tagging {image} image ... with tag {tags[image][0]}:{tags[image][1]}")
    client.images.get(image).tag(tags[image][0],tags[image][1])
    print(f"Pushing {image} image ... to {tags[image][0]}:{tags[image][1]}")
    client.images.push(tags[image][0], tags[image][1])

# Revert to host architecture
run_cmd(f"docker compose -f {docker_compose_file} down")
run_cmd(f"docker compose -f {docker_compose_file} pull")
run_cmd(f"docker compose -f {docker_compose_file} build")
run_cmd(f"docker compose -f {docker_compose_file} up -d")

print("Done")

