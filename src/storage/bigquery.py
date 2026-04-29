import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
DATASET_ID  = "marketdata"


def get_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def create_dataset():
    """Create BigQuery dataset if it doesn't exist."""
    client  = get_client()
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset.location = "asia-south1"

    try:
        client.create_dataset(dataset, exists_ok=True)
        print(f"Dataset {DATASET_ID} ready.")
    except Exception as e:
        print(f"Dataset error: {e}")


def create_raw_orderbook_table():
    client   = get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.raw_orderbook"

    from src.storage.gcs import list_files
    files = list_files("raw/orderbook/")

    if not files:
        print("No raw files found in GCS.")
        return

    # Skip old April 27 files — different schema
    # Only load files from April 28 onwards
    files = [f for f in files if "2026-04-27" not in f]

    if not files:
        print("No files to load after filtering old schema files.")
        return

    uris = [f"gs://{BUCKET_NAME}/{f}" for f in files]
    print(f"Loading {len(uris)} files into BigQuery...")

    job_config = bigquery.LoadJobConfig(
        source_format     = bigquery.SourceFormat.PARQUET,
        autodetect        = True,
        write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        client.delete_table(table_id, not_found_ok=True)
        load_job = client.load_table_from_uri(
            uris, table_id, job_config=job_config
        )
        load_job.result()
        table = client.get_table(table_id)
        print(f"Loaded {table.num_rows:,} rows into raw_orderbook")
    except Exception as e:
        print(f"Table error: {e}")


def create_processed_features_table():
    """Load processed feature files into BigQuery."""
    client   = get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.processed_features"

    from src.storage.gcs import list_files
    files = list_files("processed/features/")

    if not files:
        print("No processed files found in GCS.")
        return

    uris = [f"gs://{BUCKET_NAME}/{f}" for f in files]
    print(f"Loading {len(uris)} processed files into BigQuery...")

    job_config = bigquery.LoadJobConfig(
        source_format     = bigquery.SourceFormat.PARQUET,
        autodetect        = True,
        write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        client.delete_table(table_id, not_found_ok=True)
        load_job = client.load_table_from_uri(
            uris, table_id, job_config=job_config
        )
        load_job.result()
        table = client.get_table(table_id)
        print(f"Loaded {table.num_rows:,} rows into processed_features")
    except Exception as e:
        print(f"Table error: {e}")

def query(sql: str) -> "pd.DataFrame":
    """Run a SQL query against BigQuery and return a DataFrame."""
    client = get_client()
    print(f"Running query...")
    result = client.query(sql).to_dataframe()
    print(f"Returned {len(result):,} rows")
    return result


def setup_all():
    """Run once to set up all BigQuery tables."""
    print("Setting up BigQuery...")
    create_dataset()
    create_raw_orderbook_table()
    create_processed_features_table()
    print("BigQuery setup complete.")


if __name__ == "__main__":
    setup_all()

    # Test query
    client = get_client()
    df = query(f"""
        SELECT
            symbol,
            COUNT(*) as tick_count,
            MIN(last_price) as min_price,
            MAX(last_price) as max_price,
            AVG(spread) as avg_spread
        FROM `{PROJECT_ID}.{DATASET_ID}.raw_orderbook`
        GROUP BY symbol
        ORDER BY tick_count DESC
    """)
    print("\nData summary:")
    print(df.to_string())