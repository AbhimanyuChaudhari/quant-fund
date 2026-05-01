import schedule
import time
from datetime import datetime
from src.storage.gcs import list_files
from src.processing.duckdb_pipeline import run_pipeline
from src.processing.options_pipeline import run_options_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─────────────────────────────────────
# Futures processing
# ─────────────────────────────────────
def get_unprocessed_futures_jobs() -> list[tuple]:
    """
    Find futures symbol/date combos that have raw data
    but no processed features yet.
    Excludes options (CE/PE) and spot symbols.
    """
    raw_files       = list_files("raw/orderbook/")
    processed_files = list_files("processed/features/")

    raw_jobs = set()
    for f in raw_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3]
            # Only futures — skip options and spot
            if symbol.endswith(("CE", "PE", "_SPOT")):
                continue
            raw_jobs.add((symbol, date))

    processed_jobs = set()
    for f in processed_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3].replace(".parquet", "")
            processed_jobs.add((symbol, date))

    pending = raw_jobs - processed_jobs
    return sorted(pending)


# ─────────────────────────────────────
# Options processing
# ─────────────────────────────────────
def get_unprocessed_options_jobs() -> list[tuple]:
    """
    Find underlying/date combos that have raw options data
    but no processed options features yet.
    Groups all strikes under one underlying per day.
    """
    raw_files       = list_files("raw/orderbook/")
    processed_files = list_files("processed/options/")

    # Find dates that have raw options data per underlying
    raw_jobs = set()
    for f in raw_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3]
            if symbol.endswith(("CE", "PE")):
                # Extract underlying from symbol
                if symbol.startswith("BANKNIFTY"):
                    underlying = "BANKNIFTY"
                elif symbol.startswith("NIFTY"):
                    underlying = "NIFTY"
                else:
                    continue
                raw_jobs.add((underlying, date))

    # Find already processed options
    processed_jobs = set()
    for f in processed_files:
        parts = f.split("/")
        if len(parts) >= 4:
            underlying = parts[2]
            date       = parts[3].replace(".parquet", "")
            processed_jobs.add((underlying, date))

    pending = raw_jobs - processed_jobs
    return sorted(pending)


# ─────────────────────────────────────
# Daily processing runs
# ─────────────────────────────────────
def run_futures_processing():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Starting futures processing...")

    pending = get_unprocessed_futures_jobs()

    if not pending:
        print("Futures: nothing to process.")
        return

    print(f"Futures: {len(pending)} pending jobs | 8 parallel workers...")

    def process_job(job):
        symbol, date = job
        try:
            run_pipeline(symbol, date)
            return f"OK: {symbol} | {date}"
        except Exception as e:
            return f"ERROR: {symbol} | {date}: {e}"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures_map = {executor.submit(process_job, job): job
                       for job in pending}
        for future in as_completed(futures_map):
            print(future.result())

    print("Futures processing complete.")


def run_options_processing():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Starting options processing...")

    pending = get_unprocessed_options_jobs()

    if not pending:
        print("Options: nothing to process.")
        return

    print(f"Options: {len(pending)} pending jobs...")

    # Options pipeline is heavier (Greeks computation) — run sequentially
    for underlying, date in pending:
        try:
            run_options_pipeline(underlying, date)
            print(f"OK: {underlying} | {date}")
        except Exception as e:
            print(f"ERROR: {underlying} | {date}: {e}")

    print("Options processing complete.")


def run_daily_processing():
    """Run both futures and options processing."""
    print(f"\n{'='*55}")
    print(f"Daily processing — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")

    run_futures_processing()
    run_options_processing()

    print(f"\nAll processing complete.")


if __name__ == "__main__":
    # Run immediately on start
    run_daily_processing()

    # Schedule daily at 6:05am EST (after Indian market close + processing buffer)
    schedule.every().day.at("06:05").do(run_daily_processing)

    print("\nScheduler running. Next run at 06:05 EST daily.")
    while True:
        schedule.run_pending()
        time.sleep(60)