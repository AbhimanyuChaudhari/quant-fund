import io
import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")


def get_bucket():
    client = storage.Client(project=PROJECT_ID)
    return client.bucket(BUCKET_NAME)


def upload_dataframe(df: pd.DataFrame, blob_name: str):
    """
    Write a DataFrame directly to GCS as Parquet.
    No local file created.
    """
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        buf,
        compression="zstd"
    )
    buf.seek(0)

    bucket = get_bucket()
    blob   = bucket.blob(blob_name)
    blob.upload_from_file(
        buf,
        content_type="application/octet-stream"
    )
    return blob_name


def list_files(prefix: str = "raw/orderbook/") -> list[str]:
    """List all files under a GCS prefix."""
    bucket = get_bucket()
    return [b.name for b in bucket.list_blobs(prefix=prefix)]


def download_dataframe(blob_name: str) -> pd.DataFrame:
    """Download a Parquet file from GCS into a DataFrame."""
    bucket = get_bucket()
    blob   = bucket.blob(blob_name)
    buf    = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pq.read_table(buf).to_pandas()
