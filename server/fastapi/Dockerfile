# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

ARG DATA_DB_CONN_STRING
ENV DATA_DB_CONN_STRING=$DATA_DB_CONN_STRING

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY ./server/fastapi /app
COPY ./server/robocorp/actions_plugins/recipe-server/actions.py /app/actions.py
COPY ./templates /app/templates
COPY ./utils /app/utils
COPY ./management/skills.py /app/recipes/skills.py
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
# EXPOSE 8080

RUN echo "{\"dsn\": \"$DATA_DB_CONN_STRING\"}" > ./postgres_data_connection.json

RUN echo "{\"dsn\": \"$RECIPE_DB_CONN_STRING\"}" > ./postgres_recipe_connection.json


# Run the command to start uWSGI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]