import requests
import json
import os
import time
from urllib.parse import urlencode
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

APIS_CONFIG = "apis.config"

# We use this to map column names to be the same in all files
# Note that shapefile geopandas columns must be less than 10 characters
# TODO, move this to API config
admin0_code_name = "adm0_code"
admin1_code_name = "adm1_code"
admin2_code_name = "adm2_code"
admin3_code_name = "adm3_code"
col_map = {
    "location_code": admin0_code_name,
    "admin1_code": admin1_code_name,
    "admin2_code": admin2_code_name,
    "admin3_code": admin3_code_name,
    "ADM0_PCODE": admin0_code_name,
    "ADM1_PCODE": admin1_code_name,
    "ADM2_PCODE": admin2_code_name,
    "ADM3_PCODE": admin3_code_name,
    "admin0Pcod": admin0_code_name,
    "admin1Pcod": admin1_code_name,
    "admin2Pcod": admin2_code_name,
}

def get_api_def(api):

    api_name = api["api_name"]
    api_host = api["openapi_def"].split("/")[2]
    openapi_def = api["openapi_def"]
    openapi_filename = f"./api/{api_name}/{openapi_def.split('/')[-1]}"

    print(f"Getting {api_name} API definition from {openapi_def} and saving it to {openapi_filename}")

    api = requests.get(openapi_def)
    api = json.loads(api.text)

    with open(openapi_filename, "w") as f:
        apis_formatted = json.dumps(api, indent=4, sort_keys=True)
        f.write(apis_formatted)

    # Needed for some application using openapi's definition
    if 'servers' not in api:
        api['servers'] =   [
            {
                "url": f"https:/{api_host}"
            }
        ]
    
    return api

def get_api_data(endpoint, params):

    print('URL', endpoint + '/?' + urlencode(params))
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            return [data]
    else:
        msg = "No data was returned for endpoint:" + endpoint
        print(msg)
        return msg


def download_openapi_data(api_host, openapi_def, excluded_endpoints, save_path):
    """
    Downloads data based on the functions specified in the openapi.json definition file.

    TODO: This currently assumes paging per the new HAPI API, and would need work to 
    extend to other approaches.

    Args:
        api_hiost: Host URL
        openapi_def (str): The path to the openapi JSON file.
        save_path (str): Where to save the data

    """

    limit = 1000
    offset = 0

    files = os.listdir(save_path)    
    for f in files:
        if 'openapi.json' not in f:
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

        data = []
        offset = 0
        output = []
        while len(output) > 0 or offset == 0:
            output = get_api_data(url, {'limit':limit, 'offset': offset}) 
            print(output)
            data = data + output
            print(len(data), len(output))
            offset = offset + limit
            time.sleep(1)

        if len(data) > 0:
            endpoint_clean = endpoint.replace("/", "_")
            if endpoint_clean[0] == "_":
                endpoint_clean = endpoint_clean[1:]

            print(len(data), "Before DF")
            df = pd.DataFrame(data)
            print(df.shape[0], "After DF")
            file_name = f"{save_path}/{endpoint_clean}.csv"
            df.to_csv(file_name, index=False)
            with open(f"{save_path}/{endpoint_clean}_meta.json", "w") as f:
                full_meta = openapi_def["paths"][endpoint]
                f.write(json.dumps(full_meta, indent=4))

            print(f"Saved {file_name}")


def read_apis_config():
    with open(APIS_CONFIG) as f:
        print("Reading apis.config")
        apis = json.load(f)
    return apis

def is_running_in_docker():
    return os.path.exists('/.dockerenv')

def connect_to_db():
    """
    Connects to the PostgreSQL database using the environment variables for host, port, database, user, and password.

    Returns:
        sqlalchemy.engine.base.Engine: The database connection engine.
    """

    load_dotenv("../.env")

    if is_running_in_docker():
        host = os.getenv("POSTGRES_DATA_HOST")
    else:
        host = 'localhost'        

    host = 'localhost'
    port = os.getenv("POSTGRES_DATA_PORT")
    database = os.getenv("POSTGRES_DATA_DB")
    user = os.getenv("POSTGRES_DATA_USER")
    password = os.getenv("POSTGRES_DATA_PASSWORD")
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    print(conn_str)
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


def upload_csv_files(files_dir, conn, api_name):
    """
    Uploads CSV files from a directory to a SQLite database.

    Args:
        files_dir (str): The directory path where the CSV files are located.
        conn (sqlite3.Connection): The SQLite database connection object.
        api_name (str): Name of the api, eg hapi

    Returns:
        None
    """
    datafiles = os.listdir(files_dir)
    for f in datafiles:
        if f.endswith(".csv"):
            df = pd.read_csv(f"{files_dir}/{f}")
            df = map_code_cols(df, col_map)
            # TODO: This is a temporary workaround to account for HAPI having 
            # aggregate and disaggregated data in the same tables, where the hierarchy differs by country
            if api_name == "hapi":
                df = filter_hdx_df(df)
            table = f"{api_name}_{sanitize_name(f)}"
            print(f"Creating table {table} from {f}")
            df.to_sql(table, conn, if_exists="replace")


def upload_shape_files(files_dir, conn):
    """
    Uploads shape files from a directory to a PostgreSQL database.

    Args:
        files_dir (str): The directory path where the shape files are located.
        conn (psycopg2.extensions.connection): The PostgreSQL database connection.

    Returns:
        None
    """
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    else:
        for f in os.listdir("./tmp"):
            os.remove(f"./tmp/{f}")
    for f in os.listdir(files_dir):
        if f.endswith(".zip"):
            print(f"Unzipping {f}")
            os.system(f"unzip {files_dir}/{f} -d ./tmp")
    df_list = []
    for f in os.listdir("./tmp"):
        if f.endswith(".shp"):
            df = gpd.read_file(f"./tmp/{f}")
            table = sanitize_name(f)
            print(f"Processing table {table} from {f}")
            df_list.append(df)
    all_shapes = pd.concat(df_list, ignore_index=True)
    print(all_shapes.shape)
    all_shapes.to_postgis("hdx_shape_files", conn, if_exists="replace")


def map_code_cols(df, col_map):
    """
    Map columns in a DataFrame to a new set of column names.

    Args:
        df (pandas.DataFrame): The DataFrame to be mapped.
        col_map (dict): A dictionary containing the mapping of old column names to new column names.

    Returns:
        pandas.DataFrame: The mapped DataFrame.
    """
    for c in col_map:
        if c in df.columns:
            df.rename(columns={c: col_map[c]}, inplace=True)

    return df

def filter_hdx_df(df, **kwargs):
    """
    Filter a pandas DataFrame by removing columns where all values are null and removing rows where any value is null.
    Hack to get around the fact HDX mixes total values in with disaggregated values in the API

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        **kwargs: Additional keyword arguments.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    df_orig = df.copy()

    if df.shape[0] == 0:
        return df_orig

    dfs = []
    if admin0_code_name in df.columns:
        for country in df[admin0_code_name].unique():
            df2 = df.copy()
            df2 = df2[df2[admin0_code_name] == country]

            # Remove any columns where all null
            df2 = df2.dropna(axis=1, how="all")

            # Remove any rows where one of the values is null
            df2 = df2.dropna(axis=0, how="any")

            dfs.append(df.iloc[df2.index])

        df = pd.concat(dfs)

    return df


def main():
    apis = read_apis_config()
    conn = connect_to_db()
    for api in apis:
        openapi_def = get_api_def(api)
        api_name = api["api_name"]
        save_path = f'./api/{api_name}/'
        api_host = api["openapi_def"].split("/")[2]
        excluded_endpoints = api["excluded_endpoints"]

        # Extract data from remote APIs which are defined in apis.config
        #download_openapi_data(api_host, openapi_def, excluded_endpoints, save_path)

        # Upload CSV files to the database
        upload_csv_files(save_path, conn, api_name)

        # Upload metadata file here

if __name__ == "__main__":
    main()