# backend/fetch_s3_data.py
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")  # e.g. single-city-weather-data
DATA_DIR = Path(os.environ.get("CARBSE_DATA_DIR", "/tmp/data_private"))
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

FILES_TO_FETCH = [
    "AdaptiveModels.xlsx",
    "City_Data_ASHRAE55_2025.xlsx",
    "City_Data_IMAC_MM_2025.xlsx",
    "City_Data_IMAC_NV_2025.xlsx",
    "City_Data_IMAC_R_2025.xlsx",
    "WeathertoolLocations.xlsx",
]

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def fetch():
    if not S3_BUCKET:
        print("S3_BUCKET_NAME not set; skipping S3 download.")
        return

    ensure_dir(DATA_DIR)
    session = boto3.session.Session(region_name=AWS_REGION)
    s3 = session.client("s3")

    for fname in FILES_TO_FETCH:
        dest = DATA_DIR / fname
        try:
            print(f"Downloading s3://{S3_BUCKET}/{fname} -> {dest}")
            s3.download_file(S3_BUCKET, fname, str(dest))
        except ClientError as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            print(f"Failed to download {fname}: {e} (code={code})")
        except Exception as e:
            print(f"Unexpected error while downloading {fname}: {e}")

if __name__ == "__main__":
    fetch()
