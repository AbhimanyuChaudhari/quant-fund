import duckdb
import gcsfs

fs  = gcsfs.GCSFileSystem(project='hedge-fund-494103')
con = duckdb.connect()
con.register_filesystem(fs)

print("=== BANKNIFTY May 5 Options Data Quality ===\n")

try:
    row = con.execute("""
        SELECT
            COUNT(*)              as total_bars,
            COUNT(DISTINCT symbol) as symbols,
            SUM(CASE WHEN close > 1 AND iv > 0 THEN 1 ELSE 0 END) as good_bars,
            AVG(CASE WHEN close > 1 THEN close END)  as avg_premium,
            AVG(CASE WHEN iv > 0 THEN iv END)        as avg_iv,
            AVG(CASE WHEN delta IS NOT NULL THEN ABS(delta) END) as avg_abs_delta,
            MIN(ts_ist) as first_bar,
            MAX(ts_ist) as last_bar
        FROM read_parquet('gs://hedge-fund-494103-marketdata/processed/options/BANKNIFTY/2026-05-05.parquet')
    """).fetchone()

    print(f"Total bars:    {row[0]:,}")
    print(f"Symbols:       {row[1]}")
    print(f"Good bars:     {row[2]:,} (premium>1, IV>0)")
    print(f"Avg premium:   Rs.{row[3]:.2f}" if row[3] else "Avg premium:   N/A")
    print(f"Avg IV:        {row[4]:.3f}" if row[4] else "Avg IV:        N/A")
    print(f"Avg |delta|:   {row[5]:.3f}" if row[5] else "Avg |delta|:   N/A")
    print(f"First bar:     {row[6]}")
    print(f"Last bar:      {row[7]}")

    # Check columns available
    cols = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'read_parquet'
    """)
    
    # Sample a few rows
    print("\nSample data (ATM options):")
    sample = con.execute("""
        SELECT symbol, ts_ist, close, iv, delta, gamma, theta, spread_mean
        FROM read_parquet('gs://hedge-fund-494103-marketdata/processed/options/BANKNIFTY/2026-05-05.parquet')
        WHERE close > 10 AND iv > 0
        ORDER BY ts_ist
        LIMIT 5
    """).df()
    print(sample.to_string())

except Exception as e:
    print(f"Error: {e}")