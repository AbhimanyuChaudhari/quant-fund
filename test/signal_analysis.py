"""
Signal analysis — do our microstructure features predict
short-term price direction in NIFTY futures?

For each bar we check:
  - imbalance_last
  - price_mom_30s
  - realized_vol_60s
  - volume_ratio
  - spread_zscore

Then look at forward returns at 10s, 30s, 60s, 300s
to see if signals have predictive power.
"""

import duckdb
import gcsfs
import pandas as pd
import numpy as np

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
con = duckdb.connect()
con.register_filesystem(fs)

# Load NIFTY Apr 30 processed features
gcs_path = f"gs://{BUCKET_NAME}/processed/features/NIFTY26MAYFUT/2026-04-30.parquet"
df = con.execute(f"SELECT * FROM read_parquet('{gcs_path}') ORDER BY ts_sec").df()

print(f"Loaded {len(df):,} bars\n")

# ── Compute forward returns ───────────────────────────
for horizon in [10, 30, 60, 300]:
    df[f"fwd_ret_{horizon}s"] = df["close"].shift(-horizon) / df["close"] - 1
    df[f"fwd_dir_{horizon}s"] = np.sign(df[f"fwd_ret_{horizon}s"])

# Drop NaN rows (end of day)
df = df.dropna(subset=["fwd_ret_300s", "imbalance_last",
                        "price_mom_30s", "realized_vol_60s"])

print(f"Bars after dropna: {len(df):,}\n")

# ── Signal analysis ───────────────────────────────────
signals = {
    "imbalance_last":   {"bins": [-1, -0.5, -0.2, 0.2, 0.5, 1]},
    "price_mom_30s":    {"bins": [-0.002, -0.001, -0.0002, 0.0002, 0.001, 0.002]},
    "volume_ratio":     {"bins": [0, 0.5, 0.8, 1.2, 2.0, 10]},
}

for signal, cfg in signals.items():
    print(f"{'='*55}")
    print(f"Signal: {signal}")
    print(f"{'='*55}")

    df["bucket"] = pd.cut(df[signal], bins=cfg["bins"], include_lowest=True)

    for horizon in [30, 60, 300]:
        fwd = f"fwd_ret_{horizon}s"
        agg = df.groupby("bucket")[fwd].agg(["mean", "std", "count"])
        agg["mean_pts"] = agg["mean"] * df["close"].mean()  # convert to price pts
        agg["t_stat"]   = agg["mean"] / (agg["std"] / np.sqrt(agg["count"]))

        print(f"\n  Forward return at {horizon}s:")
        print(f"  {'Bucket':<25} {'Mean(pts)':>10} {'Count':>8} {'T-stat':>8}")
        print(f"  {'-'*55}")
        for bucket, row in agg.iterrows():
            print(f"  {str(bucket):<25} {row['mean_pts']:>10.4f} "
                  f"{row['count']:>8.0f} {row['t_stat']:>8.2f}")

print(f"\n{'='*55}")
print("Correlation matrix (signals vs fwd_ret_60s):")
print(f"{'='*55}")
cols = ["imbalance_last", "price_mom_30s", "volume_ratio",
        "realized_vol_60s", "spread_zscore", "fwd_ret_60s"]
corr = df[cols].corr()["fwd_ret_60s"].drop("fwd_ret_60s")
for col, val in corr.items():
    bar = "█" * int(abs(val) * 50)
    sign = "+" if val > 0 else "-"
    print(f"  {col:<25} {sign}{abs(val):.4f}  {bar}")
