import os
import shutil
import tarfile
import time

import gdown

# Note: When running this script, make sure datadb is stopped in docker.

# This is the exact link you get when you get the link under sharing. See this folder:
# https://drive.google.com/drive/folders/1E4G9HM-QzxdXVNkgP3fQXsuNcABWzdus?usp=drive_link for files
url_demo_data = "https://drive.google.com/file/d/1hHjruN6Xzmtdg-CQELQMzp5nlDoHfwc3/view?usp=drive_link"
filename_demo_data = "datadb.tar.gz"

if not os.getcwd().endswith("data"):
    print("Please run this script from the 'data' directory")
    exit(1)

print("\nDownloading demo data file....\n")

# Download Demo Data from Google Drive
gdown.download(url_demo_data, filename_demo_data, quiet=False, fuzzy=True)

print("Demo data has been downloaded\n")

print("Extracting demo data....\n")
tar = tarfile.open(filename_demo_data)
tar.extractall()
tar.close()
