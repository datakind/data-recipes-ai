import json
import os
import re
import shutil
import sys
import time
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from shapefiles import download_hdx_boundaries
from sqlalchemy import create_engine, text

# This is copied into Docker env
from utils.utils import is_running_in_docker, read_integration_config

INTEGRATION_CONFIG = "ingestion.config"

load_dotenv("../.env")


def get_api_def(api):

    api_name = api["api_name"]
    api_host = api["openapi_def"].split("/")[2]
    openapi_def = api["openapi_def"]
    openapi_filename = f"./api/{api_name}/{openapi_def.split('/')[-1]}"

    print(
        f"Getting {api_name} API definition from {openapi_def} and saving it to {openapi_filename}"
    )

    api = requests.get(openapi_def)
    api = json.loads(api.text)

    with open(openapi_filename, "w") as f:
        apis_formatted = json.dumps(api, indent=4, sort_keys=True)
        f.write(apis_formatted)

    # Needed for some application using openapi's definition
    if "servers" not in api:
        api["servers"] = [{"url": f"https:/{api_host}"}]

    return api


def get_api_data(endpoint, params, data_node=None):
    """
    Retrieves data from an API endpoint.

    Args:
        endpoint (str): The URL of the API endpoint.
        params (dict): The parameters to be sent with the API request.
        data_node (str, optional): The key of the data node to extract from the API response. Defaults to None.

    Returns:
        list: A list of data items retrieved from the API endpoint.

    Raises:
        None

    """
    print("URL", endpoint + "/?" + urlencode(params))
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if data_node and data_node != "":
            data = data[data_node]
        if isinstance(data, list):
            return data
        else:
            return [data]
    else:
        msg = "No data was returned for endpoint:" + endpoint
        print(msg)
        return msg


def download_openapi_data(
    api_host, openapi_def, excluded_endpoints, data_node, save_path, query_extra="", skip_downloaded=False
):
    """
    Downloads data based on the functions specified in the openapi.json definition file.

    TODO: This currently assumes paging per the new HAPI API, and would need work to
    extend to other approaches.

    Args:
        api_host: Host URL
        openapi_def (str): The path to the openapi JSON file.
        save_path (str): Where to save the data
        excluded_endpoints (list): List of endpoints to exclude
        data_node (str): The node in the openapi JSON file where the data is stored
        query_extra (str): Extra query parameters to add to the request
        skip_downloaded (bool): If True, skip downloading data that already exists

    """

    limit = 1000
    offset = 0

    files = os.listdir(save_path)
    if skip_downloaded==False:
        for f in files:
            if "openapi.json" not in f:
                filename = f"{save_path}/{f}"
                os.remove(filename)

    for endpoint in openapi_def["paths"]:
        if endpoint in excluded_endpoints:
            print(f"Skipping {endpoint}")
            continue
        print(endpoint)
        if "get" not in openapi_def["paths"][endpoint]:
            print(f"Skipping endpoint with no 'get' method {endpoint}")
            continue
        url = f"https://{api_host}/{endpoint}"

        print(url)

        endpoint_clean = endpoint.replace("/", "_")
        if endpoint_clean[0] == "_":
            endpoint_clean = endpoint_clean[1:]
        file_name = f"{save_path}/{endpoint_clean}.csv"

        if skip_downloaded and os.path.exists(file_name):
            print(f"Skipping {endpoint} as {file_name} already exists")
            continue

        data = []
        offset = 0
        output = []
        while len(output) > 0 or offset == 0:
            query = {"limit": limit, "offset": offset}
            if query_extra:
                query.update(query_extra)
            output = get_api_data(url, query, data_node)
            if "No data" in output:
                break
            print(output)
            data = data + output
            print(len(data), len(output))
            offset = offset + limit
            time.sleep(1)

        if len(data) > 0:
            print(len(data), "Before DF")
            df = pd.DataFrame(data)
            print(df.shape[0], "After DF")
            df.to_csv(file_name, index=False)
            with open(f"{save_path}/{endpoint_clean}_meta.json", "w") as f:
                full_meta = openapi_def["paths"][endpoint]
                f.write(json.dumps(full_meta, indent=4))

            print(f"Saved {file_name}")


def connect_to_db():
    """
    Connects to the PostgreSQL database using the environment variables for host, port, database, user, and password.

    Returns:
        sqlalchemy.engine.base.Engine: The database connection engine.
    """

    load_dotenv("../.env")

    if is_running_in_docker():
        print("Running in Docker ...")
        host = os.getenv("POSTGRES_DATA_HOST")
    else:
        host = "localhost"
    port = os.getenv("POSTGRES_DATA_PORT")
    database = os.getenv("POSTGRES_DATA_DB")
    user = os.getenv("POSTGRES_DATA_USER")
    password = os.getenv("POSTGRES_DATA_PASSWORD")
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    try:
        conn = create_engine(conn_str)
        return conn
    except Exception as error:
        print("--------------- Error while connecting to PostgreSQL", error)


def sanitize_name(name):
    """
    Sanitizes the given name by removing '.csv', replacing '-' with '_', removing 'a__',
    removing 'd__', and replacing '.' with '_'.

    Args:
        name (str): The name to be sanitized.

    Returns:
        str: The sanitized name.
    """
    table_name = (
        name.replace(".csv", "")
        .replace("-", "_")
        .replace("a__", "")
        .replace("d__", "")
        .replace(".", "_")
        .replace("api_v1_themes_", "")
        .replace("api_v1_", "")
    )

    return table_name


def get_cols_string(table, conn):
    """
    Get the columns of a table as a string.

    Args:
        table (str): The table name.
        conn (sqlalchemy.engine.base.Connection): The database connection object.

    Returns:
        str: The columns of the table as a string.
    """
    cols = ""
    with conn.connect() as connection:
        statement = text(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'"
        )
        result = connection.execute(statement)
        cols = result.fetchall()
        cols_str = ""
        for c in cols:
            cols_str += f"{c[0]} ({c[1]}); "
    return cols_str


def process_openapi_data(api_name, files_dir, field_map, standard_names):
    """
    Process OpenAPI data by reading CSV files from a directory, mapping field names,
    filtering specific data based on the API name, and saving the modified data back to the CSV files.

    Args:
        api_name (str): The name of the OpenAPI.
        files_dir (str): The directory path where the CSV files are located.
        field_map (dict): A dictionary mapping original field names to new field names.
        standard_names (dict): A dictionary containing the standard names for the fields.

    Returns:
        None
    """
    datafiles = os.listdir(files_dir)
    for f in datafiles:
        if f.endswith(".csv"):
            filename = f"{files_dir}/{f}"
            df = pd.read_csv(filename)
            df = map_field_names(df, field_map)

            # Import API-specific processing functions
            import_str = f"from api.{api_name}_utils import post_process_data"
            print(f"Processing {filename} with {import_str}")
            exec(import_str)
            post_process_str = "post_process_data(df, standard_names)"
            print("Post processing with", post_process_str)
            print("      Before shape", df.shape)
            df = eval(post_process_str)
            print("      After shape", df.shape)

            df.to_csv(filename, index=False)


def save_openapi_data(files_dir, conn, api_name):
    """
    Uploads CSV files from a directory to Postgres. It assumes files_dir contains CSV files as
    well as a metadata json file for each CSV file.

    Args:
        files_dir (str): The directory path where the CSV files are located.
        conn (sqlite3.Connection): The SQLite database connection object.
        api_name (str): Name of the api, eg hapi

    Returns:
        None
    """
    datafiles = os.listdir(files_dir)
    table_metadata = []
    for f in datafiles:
        if f.endswith(".csv"):
            df = pd.read_csv(f"{files_dir}/{f}")
            table = f"{api_name}_{sanitize_name(f)}"
            print(f"Creating table {table} from {f}")
            df.to_sql(table, conn, if_exists="replace", index=False)

            # Collate metadata
            meta_file = f"{files_dir}/{f.replace('.csv', '_meta.json')}"
            if os.path.exists(meta_file):
                with open(meta_file) as mf:
                    meta = json.load(mf)
                    r = {}
                    r["api_name"] = api_name
                    r["table_name"] = table
                    r["summary"] = str(meta["get"]["tags"])
                    r["columns"] = get_cols_string(table, conn)
                    r["api_description"] = meta["get"]["summary"]
                    if "description" in meta["get"]:
                        r["api_description"] += f' : {meta["get"]["description"]}'
                    r["api_definition"] = str(meta)
                    r["file_name"] = f
                    table_metadata.append(r)

    # We could also use Postgres comments, but this is simpler for LLM agents for now
    table_metadata = pd.DataFrame(table_metadata)
    table_metadata.to_sql("table_metadata", conn, if_exists="replace", index=False)


def empty_folder(folder):
    """
    Remove all files and subdirectories within the specified folder.

    Args:
        folder (str): The path to the folder.

    Raises:
        IsADirectoryError: If the specified folder is a directory.
    """
    for f in os.listdir(folder):
        try:
            os.remove(f"{folder}/{f}")
        except IsADirectoryError:
            shutil.rmtree(f"{folder}/{f}")


def upload_hdx_shape_files(files_dir, conn):
    """
    Uploads shape files from a directory to a PostgreSQL database.

    Args:
        files_dir (str): The directory path where the shape files are located.
        conn (psycopg2.extensions.connection): The PostgreSQL database connection.

    Returns:
        None
    """

    shape_files_table = "hdx_shape_files"

    df_list = []
    for f in os.listdir(files_dir):
        if f.endswith(".shp"):
            df = gpd.read_file(f"{files_dir}/{f}")
            table = sanitize_name(f)
            print(f"Processing table {table} from {f}")
            df_list.append(df)
    all_shapes = pd.concat(df_list, ignore_index=True)
    print(all_shapes.shape)
    all_shapes.to_postgis(shape_files_table, conn, if_exists="replace")

    cols = get_cols_string(shape_files_table, conn)
    with conn.connect() as connection:
        print(f"Creating metadata for {shape_files_table}")
        statement = text(
            f"INSERT INTO table_metadata (api_name, table_name, summary, columns, api_description, api_definition, file_name) VALUES ('hdx', '{shape_files_table}', \
                         'HDX Shape Files', '{cols}', 'HDX Shape Files', 'HDX Shape Files', 'HDX Shape Files')"
        )
        connection.execute(statement)
        connection.commit()


def map_field_names(df, field_map):
    """
    Map columns in a DataFrame to a new set of column names.

    Args:
        df (pandas.DataFrame): The DataFrame to be mapped.
        field_map (dict): A dictionary containing the mapping of old column names to new column names.

    Returns:
        pandas.DataFrame: The mapped DataFrame.
    """
    for c in field_map:
        if c in df.columns:
            df.rename(columns={c: field_map[c]}, inplace=True)

    return df


def main(skip_downloaded=False):
    """
    Main function for data ingestion.

    Args:
        skip_downloaded (bool, optional): Flag to skip downloaded data. Defaults to False.
    """
    apis, field_map, standard_names = read_integration_config(INTEGRATION_CONFIG)
    conn = connect_to_db()

    for api in apis:

        openapi_def = get_api_def(api)
        api_name = api["api_name"]
        save_path = f"./api/{api_name}/"
        api_host = api["openapi_def"].split("/")[2]
        excluded_endpoints = api["excluded_endpoints"]
        data_node = api["data_node"]

        if "authentication" in api:
            query_extra = ""
            print("Authentication required for", api_name)
            if api["authentication"]["type"] == "bearer_token":
                print("Bearer token required for", api_name)
            elif api["authentication"]["type"] == "api_key":
                print("API key required for", api_name)
            elif api["authentication"]["type"] == "basic":
                print("Basic authentication required for", api_name)
            elif api["authentication"]["type"] == "query_parameter":
                print("Query parameter required for", api_name)
                query_extra = {
                    api["authentication"]["name"]: api["authentication"]["value"]
                }
            else:
                print("Unknown authentication type for", api_name)

        # Extract data from remote APIs which are defined in apis.config
        download_openapi_data(
            api_host, openapi_def, excluded_endpoints, data_node, save_path, query_extra, skip_downloaded
        )

        # Standardize column names
        process_openapi_data(api_name, save_path, field_map, standard_names)

        # Upload CSV files to the database, with supporting metadata
        save_openapi_data(save_path, conn, api_name)

    # Download shapefiles from HDX. Note, this also standardizes column names
    download_hdx_boundaries(
        datafile="./api/hapi/api_v1_themes_population.csv",
        datafile_country_col=standard_names["country_code_field"],
        target_dir="./api/hdx/",
        field_map=field_map,
        map_field_names=map_field_names,
    )

    # Upload shapefiles to the database
    upload_hdx_shape_files("./api/hdx", conn)


if __name__ == "__main__":
    main(skip_downloaded=True)
