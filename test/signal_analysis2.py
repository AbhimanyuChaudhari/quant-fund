"""
Combined signal analysis — imbalance × momentum interaction
Also detrends returns to remove the Apr 30 uptrend bias.
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

gcs_path = f"gs://{BUCKET_NAME}/processed/features/NIFTY26MAYFUT/2026-04-30.parquet"
df = con.execute(f"SELECT * FROM read_parquet('{gcs_path}') ORDER BY ts_sec").df()

# Forward returns
for h in [30, 60, 300]:
    df[f"fwd_ret_{h}s"] = df["close"].shift(-h) - df["close"]  # in price pts

# Detrend — subtract rolling mean to remove uptrend bias
for h in [30, 60, 300]:
    rolling_mean = df[f"fwd_ret_{h}s"].rolling(300).mean()
    df[f"fwd_ret_{h}s_dt"] = df[f"fwd_ret_{h}s"] - rolling_mean

df = df.dropna(subset=["fwd_ret_300s_dt", "imbalance_last", "price_mom_30s"])
print(f"Bars: {len(df):,}\n")

# ── Combined signal ───────────────────────────────────
# Strong imbalance + mean reversion momentum signal
df["imbalance_strong"] = df["imbalance_last"].abs() > 0.2
df["imbalance_dir"]    = np.sign(df["imbalance_last"])
df["mom_dir"]          = np.sign(df["price_mom_30s"])

# Mean reversion: trade AGAINST momentum, in direction of imbalance
df["signal"] = df["imbalance_dir"]  # trade in direction of imbalance

# Filter: only trade when imbalance is strong AND momentum is against us
# (mean reversion setup: price pulled away from fair value)
df["reversion_setup"] = (
    (df["imbalance_strong"]) &
    (df["imbalance_dir"] != df["mom_dir"])  # imbalance and momentum disagree
)

print("=== Mean Reversion Setup Analysis ===")
print(f"Total bars:          {len(df):,}")
print(f"Reversion setups:    {df['reversion_setup'].sum():,} "
      f"({df['reversion_setup'].mean()*100:.1f}% of bars)\n")

for h in [30, 60, 300]:
    fwd = f"fwd_ret_{h}s_dt"

    # When setup fires — trade in direction of imbalance
    setup    = df[df["reversion_setup"]]
    long_set = setup[setup["imbalance_dir"] > 0]
    short_set= setup[setup["imbalance_dir"] < 0]

    if len(long_set) > 0 and len(short_set) > 0:
        long_pnl  = long_set[fwd].mean()
        short_pnl = (-short_set[fwd]).mean()  # short = negative direction
        all_pnl   = pd.concat([long_set[fwd], -short_set[fwd]])
        t_stat    = all_pnl.mean() / (all_pnl.std() / np.sqrt(len(all_pnl)))
        win_rate  = (all_pnl > 0).mean()

        print(f"Horizon {h}s:")
        print(f"  Long  setups: {len(long_set):>5,}  avg PnL: {long_pnl:>7.2f} pts")
        print(f"  Short setups: {len(short_set):>5,}  avg PnL: {short_pnl:>7.2f} pts")
        print(f"  Combined avg: {all_pnl.mean():>7.2f} pts  "
              f"t-stat: {t_stat:.2f}  win_rate: {win_rate*100:.1f}%")
        print()

# ── Stronger filter ───────────────────────────────────
print("\n=== Stronger Filter (imbalance > 0.3) ===")
df["strong_setup"] = (
    (df["imbalance_last"].abs() > 0.3) &
    (df["imbalance_dir"] != df["mom_dir"])
)

print(f"Strong setups: {df['strong_setup'].sum():,} "
      f"({df['strong_setup'].mean()*100:.1f}% of bars)\n")

for h in [60, 300]:
    fwd   = f"fwd_ret_{h}s_dt"
    setup = df[df["strong_setup"]]
    long_s  = setup[setup["imbalance_dir"] > 0]
    short_s = setup[setup["imbalance_dir"] < 0]

    if len(long_s) > 0 and len(short_s) > 0:
        all_pnl  = pd.concat([long_s[fwd], -short_s[fwd]])
        t_stat   = all_pnl.mean() / (all_pnl.std() / np.sqrt(len(all_pnl)))
        win_rate = (all_pnl > 0).mean()
        print(f"Horizon {h}s:  avg={all_pnl.mean():.2f}pts  "
              f"t-stat={t_stat:.2f}  win_rate={win_rate*100:.1f}%  "
              f"n={len(all_pnl)}")
