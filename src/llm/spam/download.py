import argparse
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def download_and_unzip_spam_data(
    url: str,
    extracted_path: str,
    data_file_path: Path,
    overwrite: bool,
    delete_zip: bool,
):
    if data_file_path.exists() and not overwrite:
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    elif data_file_path.exists() and overwrite:
        print(f"{data_file_path} already exists. Overwriting.")
        os.remove(data_file_path)
    else:
        print(f"{data_file_path} does not exist. Downloading and extracting.")

    # Download and extract the zip file
    # Copy the context to data_file_path

    with (
        urllib.request.urlopen(url) as response,
        tempfile.NamedTemporaryFile(delete=delete_zip, mode="wb") as temp_file,
    ):
        temp_file.write(response.read())
        temp_file.flush()
        with zipfile.ZipFile(temp_file.name, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        # Ensure the target directory exists
        data_file_path.parent.mkdir(parents=True, exist_ok=True)

        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        print(f"File downloaded and saved as {data_file_path}")


def main(data_dir: str, overwrite: bool, delete_zip: bool = True):
    """
    Args:
        data_dir: str, data directory
        overwrite: bool, whether to overwrite existing data
        delete_zip: bool, whether to delete the zip file after extraction
    """
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(data_dir) / Path(extracted_path) / "SMSSpamCollection.tsv"
    download_and_unzip_spam_data(
        url,
        extracted_path,
        data_file_path,
        overwrite=overwrite,
        delete_zip=delete_zip,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data"
    )
    parser.add_argument(
        "-d", "--data-dir", type=str, default="data", help="Data directory"
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )
