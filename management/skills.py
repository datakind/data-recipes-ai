import json
import logging
import os
import sys
import uuid
# TODO, temporary while we do user testing
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import requests
from dotenv import load_dotenv
from hdx.api.configuration import Configuration
from hdx.data.resource import Resource

# This is copied or mounted into Docker image
from utils import *

warnings.filterwarnings("ignore")

# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)

load_dotenv()


def get_hdx_dataset_url(resource_id):
    """
    Retrieves the dataset URL based on the given resource ID. Also returns
    the resource download URL

    Args:
        resource_id (str): The ID of the resource.

    Returns:
        str: The dataset URL.
        str: The resource download URL.

    Raises:
        Exception: If the resource cannot be fetched or the dataset ID cannot be obtained.
    """

    try:
        Configuration.create(hdx_site='prod', user_agent='Data Recipes AI', hdx_read_only=True)
    except Exception:
        print('HDX already activated')

    print(resource_id)

    # Fetch the resource
    resource = Resource.read_from_hdx(resource_id)

    # Get the link to download the raw data
    resource_url = resource["download_url"]

    # Get the dataset ID
    dataset_id = resource['package_id']

    # Construct the dataset URL
    dataset_url = f'https://data.humdata.org/dataset/{dataset_id}'

    return dataset_url, resource_url


def get_connection():
    """
    This function gets a connection to the database
    """
    host = os.getenv("POSTGRES_DATA_HOST")
    port = os.getenv("POSTGRES_DATA_PORT")
    database = os.getenv("POSTGRES_DATA_DB")
    user = os.getenv("POSTGRES_DATA_USER")
    password = os.getenv("POSTGRES_DATA_PASSWORD")

    conn = psycopg2.connect(
        dbname=database, user=user, password=password, host=host, port=port
    )
    return conn


def execute_query(query):
    """
    This skill executes a query in the data database.

    To find out what tables and columns are available, you can run "select table_name, api_name, summary, columns from table_metadata"

    """
    conn = get_connection()
    cur = conn.cursor()

    # Execute the query
    cur.execute(query)

    # Fetch all the returned rows
    rows = cur.fetchall()

    # Close the cursor and connection
    cur.close()
    conn.close()

    return rows


if __name__ == "__main__":
    # Example usage of the function:
    pass
