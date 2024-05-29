import glob
import os
import shutil
import zipfile

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
from hdx.utilities.easy_logging import setup_logging


def get_hdx_config():
    """
    Get the HDX configuration for connecting to HDX API

    Returns:
        None
    """
    try:
        Configuration.create(
            hdx_site="prod", user_agent="HAPI-test1", hdx_read_only=True
        )
    except Exception as e:
        if str(e) == "Configuration already created!":
            print("DEBUG: HDX Configuration already created. Continuing...")
        else:
            print(f"DEBUG: Exception: {e}")
            return {"file_location": ""}
    return Configuration


def get_hdx_shapefile(country_code, admin_level):
    """
    Retrieves the shapefile for the specified country and administrative level.

    Args:
        country_code (str): The 3-letter ISO name of the country, eg MLI
        admin_level (str): The administrative level.

    Returns:
        dict: A dictionary containing the file location of the shapefile.

    Raises:
        None
    """
    admin_level = admin_level.replace("admin", "")

    data_dir = "./tmp"
    datasets = Dataset.search_in_hdx(
        f"cod shapefile {country_code} administrative boundaries"
    )
    shape_dataset = f"cod-ab-{country_code.lower()}"
    response = ""
    # Iterate over the results and download the shapefiles
    for dataset in datasets:
        if dataset["name"] == shape_dataset:
            print(dataset["name"])
            resources = dataset.get_resources()
            for resource in resources:
                print(resource["name"], resource["format"])
                if "shp" in resource["format"].lower():
                    url, path = resource.download()
                    new_loc = f"{data_dir}/{country_code}.zip"
                    shutil.move(path, new_loc)
                    print(f"Shapefile downloaded from {url} and saved to {new_loc}")

                    with zipfile.ZipFile(new_loc, "r") as zip_ref:
                        zip_ref.extractall(f"{data_dir}")

                        # Remove extra files
                        for file in os.listdir(data_dir):
                            if "zip" in file or "ALL" in file or "pdf" in file:
                                os.remove(os.path.join(data_dir, file))

    # Tidy up, we will be zipping this folder. The zipfiles from HDX have differing unpacked layout
    files = glob.glob("./tmp/*/*")
    for f in files:
        if not os.path.isfile(f"./tmp/{f.split('/')[-1]}"):
            print(f"Moving {f} to ./tmp")
            shutil.move(f, "./tmp")
        else:
            os.remove(f)
    dirs = glob.glob("./tmp/*/")
    for d in dirs:
        os.rmdir(d)
    files = glob.glob("./tmp/*.zip")
    for f in files:
        os.remove(f)

    return response


def normalize_hdx_boundaries(
    datafile, field_map, map_field_names, datafile_country_col
):
    """
    HDX Boundaries have inconsistent naming conventions and pcode variable names. This function
    attempts to standardize them for easier use in HDeXpert.

    Args:
        datafile (str): Path to the data file containing location codes. Default is "./data/hdx_population.csv".
        files_prefix (str): The prefix to use for the files.
        field_map (dict): A dictionary of column names mapping, used to rename columns to standard names.
        map_field_names (function): A function to map the code columns to standard names.
        datafile_country_col (str): The column name in the data file containing the country codes.

    Returns:
        None
    """

    output_dir = "./tmp/processed/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(f"{output_dir}/*")
    for f in files:
        os.remove(f)

    df = pd.read_csv(datafile)
    print(df.columns)
    countries = df[datafile_country_col].unique()
    countries = [c.lower() for c in countries]
    # TODO: Remove Columbia, it's too big
    countries = [c for c in countries if "col" not in c]
    for country in countries:
        for admin in ["adm1", "adm2"]:
            # list off .shp files starting with country_code
            match_str = f"./tmp/{country}*{admin}*.shp"
            shp_file = glob.glob(match_str)
            if len(shp_file) > 0:
                print(shp_file)
                if len(shp_file) > 0:
                    if len(shp_file) > 1:
                        print(f"Multiple shape files found for {country} {admin}")
                        shp_file = [f for f in shp_file if f"{admin}.shp" in f]
                    shp_file = shp_file[0]
                    gdf = gpd.read_file(shp_file)
                    gdf = map_field_names(gdf, field_map)
                    shp_file = shp_file.split(admin)[0] + admin + ".shp"
                    shp_file = shp_file.replace("./tmp", "")
                    shp_file = f"{output_dir}/{shp_file[1:]}"
                    gdf.to_file(shp_file)
    return output_dir


def download_hdx_boundaries(
    datafile="./api/hapi/api_v1_population-social_population.csv",
    datafile_country_col="location_code",
    target_dir="./api/hdx/",
    field_map={},
    map_field_names=None,
):
    """
    Downloads HDX boundaries for all countries and administrative levels.

    We may use this in some form, but it doesn't seem to have pcodes or admin codes. TODO.

    Args:
        datafile (str): Path to the data file containing location codes. Default is "./data/hdx_population.csv".
        datafile_country_col (str): The column name in the data file containing the country codes.
        files_prefix (str): The prefix to use for the files.
        field_map (dict): A dictionary of column names mapping, used to rename columns to standard names.
        map_field_names (function): A function to map the code columns to standard names.

    Returns:
        None
    """
    tmp_dir = "./tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Create connection to HDX
    get_hdx_config()

    df = pd.read_csv(datafile)
    countries = df[datafile_country_col].unique()
    countries = [c.lower() for c in countries]

    for country in countries:
        for admin in ["admin1", "admin2"]:
            print(country, admin)
            get_hdx_shapefile(country, admin)

    # Align field names with other datasets
    output_dir = normalize_hdx_boundaries(
        datafile, field_map, map_field_names, datafile_country_col
    )

    # Copy normalized files to target_dir
    files = glob.glob(f"{output_dir}/*")
    for f in files:
        shutil.copy(f, target_dir)

    # Split shape files into zip files by country letter range and admin level. These
    # Are useful for assistants
    for admin in ["adm0", "adm1", "adm2"]:
        files = glob.glob(f"{output_dir}/*{admin}*")
        if len(files) > 0:
            ranges = ["a-c", "d-h", "i-z"]
            for letter_range in ranges:
                letters = letter_range.split("-")
                letters = [chr(i) for i in range(ord(letters[0]), ord(letters[1]) + 1)]
                country_sublist = [c for c in countries if c[0].lower() in letters]

                zipfile_path = (
                    f"{target_dir}/geoBoundaries-{admin}-countries_{letter_range}.zip"
                )

                with zipfile.ZipFile(zipfile_path, "w") as zipf:
                    # Iterate over all files in the output directory
                    for foldername, subfolders, filenames in os.walk(output_dir):
                        for filename in filenames:
                            # Check if the file matches the pattern
                            if any(admin in filename for admin in country_sublist):
                                # Get the full file path
                                file_path = os.path.join(foldername, filename)
                                # Add the file to the zip file
                                zipf.write(
                                    file_path,
                                    arcname=os.path.relpath(file_path, output_dir),
                                )
