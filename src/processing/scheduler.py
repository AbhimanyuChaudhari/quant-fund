import schedule
import time
from datetime import datetime
from src.storage.gcs import list_files
from src.processing.duckdb_pipeline import run_pipeline
from src.processing.options_pipeline import run_options_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

# Minimum bars a processed file must have to be considered complete.
# A full trading day (9:15-15:30 IST = 375 mins = 22,500 seconds) at 1s bars
# should have ~18,000-22,000 bars. We use 10,000 as a conservative floor —
# anything below this means the file is partial and needs reprocessing.
MIN_BARS_COMPLETE = 10_000


# ─────────────────────────────────────
# Futures processing
# ─────────────────────────────────────
def get_processed_bar_counts() -> dict:
    """
    Return a dict of {(symbol, date): bar_count} for all processed files.
    Uses GCS metadata (file size as proxy) if bar count isn't directly available,
    or reads parquet metadata via pyarrow for exact counts.
    """
    from src.storage.gcs import get_gcs_client
    import pyarrow.parquet as pq
    import gcsfs

    try:
        fs      = gcsfs.GCSFileSystem()
        files   = list_files("processed/features/")
        counts  = {}

        for f in files:
            parts = f.split("/")
            if len(parts) < 4:
                continue
            symbol = parts[2]
            date   = parts[3].replace(".parquet", "")
            try:
                # Read only metadata — does not download full file
                gcs_path = f"gs://hedge-fund-494103-marketdata-mumbai/{f}"
                pf       = pq.read_metadata(gcs_path, filesystem=fs)
                n_rows   = pf.num_rows
                counts[(symbol, date)] = n_rows
            except Exception:
                # If metadata read fails, assume incomplete (0 bars)
                counts[(symbol, date)] = 0

        return counts
    except Exception as e:
        print(f"[scheduler] Could not read bar counts: {e} — using file-exists check")
        return {}


def get_unprocessed_futures_jobs() -> list[tuple]:
    """
    Find futures symbol/date combos that need processing:
    1. Raw data exists but no processed file → process
    2. Processed file exists but has < MIN_BARS_COMPLETE bars → reprocess

    Excludes options (CE/PE) and spot symbols.
    """
    raw_files       = list_files("raw/orderbook/")
    processed_files = list_files("processed/features/")

    # Build set of raw jobs
    raw_jobs = set()
    for f in raw_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3]
            if symbol.endswith(("CE", "PE", "_SPOT")):
                continue
            raw_jobs.add((symbol, date))

    # Build set of fully processed jobs (file exists AND has enough bars)
    processed_jobs = set()
    for f in processed_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3].replace(".parquet", "")
            processed_jobs.add((symbol, date))

    # Jobs where file doesn't exist at all
    missing = raw_jobs - processed_jobs

    # Jobs where file exists but may be incomplete — check bar counts
    bar_counts  = get_processed_bar_counts()
    incomplete  = set()

    for (symbol, date) in (raw_jobs & processed_jobs):
        n_bars = bar_counts.get((symbol, date), None)
        if n_bars is None:
            # Could not read metadata — skip reprocessing to be safe
            continue
        if n_bars < MIN_BARS_COMPLETE:
            print(f"[scheduler] Incomplete: {symbol} | {date} "
                  f"({n_bars:,} bars < {MIN_BARS_COMPLETE:,}) → reprocessing")
            incomplete.add((symbol, date))

    pending = missing | incomplete

    if incomplete:
        print(f"[scheduler] {len(missing)} missing + "
              f"{len(incomplete)} incomplete = {len(pending)} total jobs")

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

    raw_jobs = set()
    for f in raw_files:
        parts = f.split("/")
        if len(parts) >= 4:
            symbol = parts[2]
            date   = parts[3]
            if symbol.endswith(("CE", "PE")):
                if symbol.startswith("BANKNIFTY"):
                    underlying = "BANKNIFTY"
                elif symbol.startswith("NIFTY") and not symbol.startswith("NIFTYNXT"):
                    underlying = "NIFTY"
                else:
                    continue
                raw_jobs.add((underlying, date))

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

    # ── FIXED: was "06:05" (EST) which fires at 3:35pm IST (mid-market) ──────
    # Now runs at 11:00 EST = 9:30pm IST = safely after market close
    schedule.every().day.at("8:30").do(run_daily_processing)

    print("\nScheduler running. Next run at 8:30 EST (9:30pm IST) daily.")
    while True:
        schedule.run_pending()
        time.sleep(60)
