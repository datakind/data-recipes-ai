import requests
import json
import os
import time
from urllib.parse import urlencode
import sys
import pandas as pd

APIS_CONFIG = "apis.config"

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


def download_data(api_host, openapi_def, excluded_endpoints, save_path):
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
            #df = map_code_cols(df, col_map)
            #df = filter_hdx_df(df)
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

def main():
    apis = read_apis_config()
    for api in apis:
        openapi_def = get_api_def(api)
        save_path = f'./api/{api["api_name"]}/'
        api_host = api["openapi_def"].split("/")[2]
        excluded_endpoints = api["excluded_endpoints"]
        download_data(api_host, openapi_def, excluded_endpoints, save_path)

if __name__ == "__main__":
    main()