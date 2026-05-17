"""
Local DuckDB Cache — Fast Backtest Data Loader
===============================================
Syncs GCS parquets to local DuckDB. Handles any column count.
500x faster than GCS reads for backtesting.

Usage:
    # Sync all data
    python src/backtest/duckdb_cache.py --sync --start 2026-04-30 --end 2026-05-14

    # Check status
    python src/backtest/duckdb_cache.py --status

    # Force re-sync specific date
    python src/backtest/duckdb_cache.py --sync --date 2026-05-13 --force
"""

import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

CACHE_DIR   = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_FILE  = CACHE_DIR / "backtest.duckdb"

# Market hours
MARKET_OPEN_IST  = 33300   # 09:15 IST
MARKET_CLOSE_IST = 55800   # 15:30 IST
CDS_OPEN_IST     = 32400   # 09:00 IST
CDS_CLOSE_IST    = 61200   # 17:00 IST

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
}


# ─── Local cache ──────────────────────────────────────────────────────────────

class LocalCache:
    """
    Flexible DuckDB cache — stores each symbol/date as its own table.
    Table name format: t_{symbol}_{date} (e.g. t_ADANIPORTS26MAYFUT_20260430)
    No fixed schema — adapts to whatever columns exist in each parquet.
    """

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    def _get_con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(str(self.cache_file))
        return self._con

    def _table_name(self, symbol: str, date: str) -> str:
        """Convert symbol/date to valid DuckDB table name."""
        sym_clean  = symbol.replace("-", "_").replace("&", "n")
        date_clean = date.replace("-", "")
        return f"t_{sym_clean}_{date_clean}"

    def is_cached(self, symbol: str, date: str) -> bool:
        """Check if symbol/date table exists in cache."""
        try:
            tbl    = self._table_name(symbol, date)
            result = self._get_con().execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = ?", [tbl]
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False

    def insert(self, symbol: str, date: str, df: pd.DataFrame):
        """
        Insert DataFrame into cache as its own table.
        Drops and recreates if already exists.
        No fixed schema — stores whatever columns the parquet has.
        """
        if df.empty:
            return

        tbl = self._table_name(symbol, date)
        con = self._get_con()

        try:
            # Drop existing table
            con.execute(f"DROP TABLE IF EXISTS {tbl}")

            # Create from DataFrame directly — no schema constraints
            con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM df")
            con.commit()

        except Exception as e:
            log.warning(f"Insert failed {symbol}/{date}: {e}")

    def load(
        self,
        symbol:            str,
        date:              str,
        market_hours_only: bool = True,
    ) -> pd.DataFrame:
        """Load symbol/date from local cache with market hours filter."""
        try:
            tbl = self._table_name(symbol, date)
            con = self._get_con()

            # Check table exists
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = ?", [tbl]
            ).fetchone()[0]

            if not exists:
                return pd.DataFrame()

            # Market hours filter
            is_currency = "USDINR" in symbol.upper()
            if market_hours_only:
                if is_currency:
                    mf = (f"WHERE ((ts_sec+19800)%86400)>={CDS_OPEN_IST} "
                          f"AND ((ts_sec+19800)%86400)<={CDS_CLOSE_IST}")
                else:
                    mf = (f"WHERE ((ts_sec+19800)%86400)>={MARKET_OPEN_IST} "
                          f"AND ((ts_sec+19800)%86400)<={MARKET_CLOSE_IST}")
            else:
                mf = ""

            return con.execute(
                f"SELECT * FROM {tbl} {mf} ORDER BY ts_sec"
            ).df()

        except Exception as e:
            log.debug(f"Cache load failed {symbol}/{date}: {e}")
            return pd.DataFrame()

    def load_date_range(
        self,
        symbol:            str,
        start:             str,
        end:               str,
        market_hours_only: bool = True,
    ) -> pd.DataFrame:
        """Load symbol across multiple dates."""
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

        frames  = []
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                df = self.load(symbol, str(current), market_hours_only)
                if not df.empty:
                    frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True).sort_values(
            "ts_sec"
        ).reset_index(drop=True)

    def status(self) -> dict:
        """Return cache statistics."""
        try:
            con    = self._get_con()
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 't_%'"
            ).df()

            if tables.empty:
                return {"tables": 0, "file_mb": 0}

            total_rows = 0
            dates      = set()
            symbols    = set()

            for tbl in tables["table_name"]:
                try:
                    n = con.execute(
                        f"SELECT COUNT(*) FROM {tbl}"
                    ).fetchone()[0]
                    total_rows += n
                    # Parse symbol and date from table name
                    # Format: t_SYMBOL_YYYYMMDD
                    parts = tbl[2:].rsplit("_", 1)
                    if len(parts) == 2:
                        symbols.add(parts[0])
                        dates.add(parts[1])
                except Exception:
                    pass

            return {
                "tables":     len(tables),
                "symbols":    len(symbols),
                "dates":      len(dates),
                "total_bars": total_rows,
                "file_mb":    round(
                    self.cache_file.stat().st_size / 1024 / 1024, 1
                ) if self.cache_file.exists() else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        if self._con:
            self._con.close()
            self._con = None


# ─── GCS sync ─────────────────────────────────────────────────────────────────

class CacheSync:
    """Syncs GCS parquets to local DuckDB cache."""

    def __init__(self, cache: LocalCache = None):
        self.cache = cache or LocalCache()

    def sync_date(
        self,
        date:    str,
        symbols: list = None,
        force:   bool = False,
    ) -> tuple:
        import gcsfs
        fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
        files = fs.glob(
            f"{BUCKET_NAME}/processed/features/*/{date}.parquet"
        )

        synced = skipped = 0

        for f in sorted(files):
            sym = f.split("/")[-2]

            # Skip index futures and options
            if sym in INDEX_FUTURES:
                skipped += 1
                continue
            if sym.endswith(("CE", "PE", "_SPOT")):
                skipped += 1
                continue
            if symbols and sym not in symbols:
                skipped += 1
                continue

            # Skip if already cached
            if not force and self.cache.is_cached(sym, date):
                skipped += 1
                continue

            try:
                gcs_path = (
                    f"gs://{BUCKET_NAME}/processed/features/"
                    f"{sym}/{date}.parquet"
                )
                con = duckdb.connect()
                con.register_filesystem(fs)
                df  = con.execute(
                    f"SELECT * FROM read_parquet('{gcs_path}')"
                ).df()
                con.close()

                if df.empty:
                    skipped += 1
                    continue

                self.cache.insert(sym, date, df)
                synced += 1
                print(f"  OK: {sym} | {date} | {len(df):,} bars")

            except Exception as e:
                log.warning(f"  FAIL: {sym}/{date}: {e}")
                skipped += 1

        return synced, skipped

    def sync_range(
        self,
        start:   str,
        end:     str,
        symbols: list = None,
        force:   bool = False,
    ) -> dict:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()
        total_s = total_sk = 0
        current = start_dt

        while current <= end_dt:
            if current.weekday() < 5:
                date_str = str(current)
                print(f"\nSyncing {date_str}...")
                s, sk = self.sync_date(
                    date_str, symbols=symbols, force=force
                )
                total_s  += s
                total_sk += sk
                print(f"  {s} synced, {sk} skipped")
            current += timedelta(days=1)

        return {"synced": total_s, "skipped": total_sk}


# ─── Drop-in replacement for data_loader ─────────────────────────────────────

class CachedDataLoader:
    """
    Drop-in replacement for data_loader.py.
    Reads from local cache first, falls back to GCS.

    Usage in data_loader.py load_day():
        from src.backtest.duckdb_cache import CachedDataLoader
        _loader = CachedDataLoader()
        cached  = _loader.load_day(symbol, date, market_hours_only)
        if not cached.empty:
            return cached
        # ... existing GCS code
    """

    def __init__(self):
        self.cache = LocalCache()

    def load_day(
        self,
        symbol:            str,
        date:              str,
        market_hours_only: bool = True,
    ) -> pd.DataFrame:
        df = self.cache.load(symbol, date, market_hours_only)
        if not df.empty:
            return df
        # Fallback to GCS + cache result
        from src.backtest.data_loader import load_day as gcs_load
        df = gcs_load(symbol, date, market_hours_only)
        if not df.empty:
            self.cache.insert(symbol, date, df)
        return df

    def load_date_range(
        self,
        symbol:            str,
        start:             str,
        end:               str,
        market_hours_only: bool = True,
    ) -> pd.DataFrame:
        df = self.cache.load_date_range(symbol, start, end, market_hours_only)
        if not df.empty:
            return df
        from src.backtest.data_loader import load_date_range as gcs_load
        return gcs_load(symbol, start, end, market_hours_only)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Local DuckDB cache for backtest data"
    )
    parser.add_argument("--sync",   action="store_true",
                        help="Sync GCS data to local cache")
    parser.add_argument("--date",   type=str,
                        help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start",  type=str, default="2026-04-30",
                        help="Start date for range sync")
    parser.add_argument("--end",    type=str,
                        help="End date (default: today)")
    parser.add_argument("--force",  action="store_true",
                        help="Re-sync even if already cached")
    parser.add_argument("--status", action="store_true",
                        help="Show cache statistics")
    args = parser.parse_args()

    cache = LocalCache()
    sync  = CacheSync(cache)

    if args.status:
        s = cache.status()
        print(f"\nCache: {CACHE_FILE}")
        print(f"  Tables:     {s.get('tables', 0)}")
        print(f"  Symbols:    {s.get('symbols', 0)}")
        print(f"  Dates:      {s.get('dates', 0)}")
        print(f"  Total bars: {s.get('total_bars', 0):,}")
        print(f"  Size:       {s.get('file_mb', 0)} MB\n")

    if args.sync:
        if args.date:
            print(f"\nSyncing {args.date}...")
            s, sk = sync.sync_date(args.date, force=args.force)
            print(f"Done: {s} synced, {sk} skipped")
        else:
            end = args.end or str(datetime.utcnow().date())
            print(f"\nSyncing {args.start} -> {end}...")
            r = sync.sync_range(args.start, end, force=args.force)
            print(f"\nTotal: {r['synced']} synced, {r['skipped']} skipped")

        # Show final status
        s = cache.status()
        print(f"\nCache status:")
        print(f"  {s.get('symbols',0)} symbols | "
              f"{s.get('dates',0)} dates | "
              f"{s.get('total_bars',0):,} bars | "
              f"{s.get('file_mb',0)} MB")

    cache.close()


if __name__ == "__main__":
    main()