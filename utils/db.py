import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

from utils.general import call_execute_query_api, is_running_in_docker

load_dotenv()


def get_connection(instance="data"):
    """
    This function gets a connection to the database

    Args:

        instance (str): The instance of the database to connect to, "recipe" or "data". Default is "data"
    """
    instance = instance.upper()

    host = os.getenv(f"POSTGRES_{instance}_HOST")
    port = os.getenv(f"POSTGRES_{instance}_PORT")
    database = os.getenv(f"POSTGRES_{instance}_DB")
    user = os.getenv(f"POSTGRES_{instance}_USER")
    password = os.getenv(f"POSTGRES_{instance}_PASSWORD")

    conn = psycopg2.connect(
        dbname=database, user=user, password=password, host=host, port=port
    )
    return conn


def execute_query(query, instance="data"):
    """
    Executes a SQL query and returns the result as a DataFrame.

    Parameters:
        query (str): The SQL query to execute.
        instance (str): The database instance to connect to. Default is "data".

    Returns:
        pandas.DataFrame: The result of the query as a DataFrame
    """

    conn = get_connection(instance)
    cur = conn.cursor()

    # Set to read-only mode
    cur.execute("SET TRANSACTION READ ONLY;")

    print(f"Executing query: {query}")

    # Execute the query
    cur.execute(query)

    # Fetch all the returned rows
    rows = cur.fetchall()

    print(f"Query returned {len(rows)} rows")

    # Get column names
    column_names = [desc[0] for desc in cur.description]

    # Close the cursor and connection
    cur.close()
    conn.close()

    # Convert rows to DataFrame
    df = pd.DataFrame(rows, columns=column_names)

    return df


def connect_to_db(instance="recipe"):
    """
    Connects to the specified database instance (RECIPE or DATA) DB and returns a connection object.

    Args:
        instance (str): The name of the database instance to connect to. Defaults to "RECIPE".

    Returns:
        sqlalchemy.engine.base.Engine: The connection object for the specified database instance.
    """

    instance = instance.upper()

    # Fallback for CLI running outside of docker
    if not is_running_in_docker():
        os.environ[f"POSTGRES_{instance}_HOST"] = "localhost"
        os.environ[f"POSTGRES_{instance}_PORT"] = "5433"

    host = os.getenv(f"POSTGRES_{instance}_HOST")
    port = os.getenv(f"POSTGRES_{instance}_PORT")
    database = os.getenv(f"POSTGRES_{instance}_DB")
    user = os.getenv(f"POSTGRES_{instance}_USER")
    password = os.getenv(f"POSTGRES_{instance}_PASSWORD")
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    # add an echo=True to see the SQL queries
    conn = create_engine(conn_str)
    return conn


async def get_data_info():
    """
    Get data info from the database.

    Returns:
        str: The data info.
    """

    global data_info

    # run this query: select table_name, summary, columns from table_metadata

    query = """
        SELECT
            table_name,
            summary,
            columns
        FROM
            table_metadata
        --WHERE
        --    countries is not null
        """

    data_info = await call_execute_query_api(query)

    return data_info
