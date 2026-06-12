import duckdb
import gcsfs

fs  = gcsfs.GCSFileSystem(project='hedge-fund-494103')
con = duckdb.connect()
con.register_filesystem(fs)

for symbol in ['USDINR26508FUT', 'USDINR26515FUT']:
    for date in ['2026-05-01', '2026-05-02']:
        path = (f"gs://hedge-fund-494103-marketdata-mumbai/processed/"
                f"features/{symbol}/{date}.parquet")
        try:
            row = con.execute(f"""
                SELECT COUNT(*)        as bars,
                       MIN(ts_ist)     as first,
                       MAX(ts_ist)     as last,
                       AVG(close)      as avg_price,
                       AVG(spread_mean)as avg_spread,
                       AVG(tick_count) as avg_ticks
                FROM read_parquet('{path}')
            """).fetchone()
            print(f"{symbol} | {date}:")
            print(f"  bars={row[0]:,} | first={row[1]} | last={row[2]}")
            print(f"  price={row[3]:.4f} | spread={row[4]:.4f} | "
                  f"ticks={row[5]:.2f}")
        except Exception as e:
            print(f"{symbol} | {date}: {e}")
