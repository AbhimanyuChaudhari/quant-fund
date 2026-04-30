import schedule
import time
from datetime import datetime, timedelta
from src.storage.gcs import list_files
from src.processing.duckdb_pipeline import run_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_unprocessed_jobs() -> list[tuple]:
    """
    Find all symbol/date combinations that have raw data
    but no processed data yet.
    """
    raw_files       = list_files("raw/orderbook/")
    processed_files = list_files("processed/features/")

    # Extract symbol/date from raw files
    # raw/orderbook/SYMBOL/DATE/TIME.parquet
    raw_jobs = set()
    for f in raw_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3]
            raw_jobs.add((symbol, date))

    # Extract symbol/date from processed files
    # processed/features/SYMBOL/DATE.parquet
    processed_jobs = set()
    for f in processed_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3].replace(".parquet", "")
            processed_jobs.add((symbol, date))

    # Jobs that need processing
    pending = raw_jobs - processed_jobs
    return sorted(pending)

def run_daily_processing():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting daily processing...")

    pending = get_unprocessed_jobs()

    if not pending:
        print("Nothing to process.")
        return

    print(f"Found {len(pending)} pending jobs. Running with 8 parallel workers...")

    def process_job(job):
        symbol, date = job
        try:
            run_pipeline(symbol, date)
            return f"OK: {symbol} | {date}"
        except Exception as e:
            return f"ERROR: {symbol} | {date}: {e}"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_job, job): job for job in pending}
        for future in as_completed(futures):
            print(future.result())

    print(f"\nDaily processing complete.")


if __name__ == "__main__":
    # Run immediately on start
    run_daily_processing()

    # Then schedule daily at 6:05am EST (3:35pm IST — after market close)
    schedule.every().day.at("06:05").do(run_daily_processing)

    print("\nScheduler running. Next run at 06:05 EST daily.")
    while True:
        schedule.run_pending()
        time.sleep(60)