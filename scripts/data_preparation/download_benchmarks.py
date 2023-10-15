import os
import gdown
import zipfile
import argparse

def download_and_extract_zip(url, destination_folder):
    # Download the ZIP file
    file_id = url.split("/")[-2]
    file_name = "archive.zip"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

    # Extract the contents of the ZIP file
    with zipfile.ZipFile(file_name) as zf:
        zf.extractall(destination_folder)

    # Delete the downloaded ZIP file
    os.remove(file_name)

def main():
    parser = argparse.ArgumentParser(description="Download and extract ZIP file from a given URL.")
    parser.add_argument("zip_url", help="URL of the ZIP file to download")
    parser.add_argument("destination_folder", help="Destination folder to extract the contents")
    
    args = parser.parse_args()
    zip_url = args.zip_url
    destination_folder = args.destination_folder

    download_and_extract_zip(zip_url, destination_folder)

if __name__ == "__main__":
    main()