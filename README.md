# Quant Fund - Market Making Infrastructure

Solo quant fund focused on Indian derivatives markets.
Built in Python, running on GCP, collecting live tick data via Zerodha.

---

## Architecture

```
Zerodha WebSocket (261 instruments)
        ↓
collector.service  →  GCS raw parquet (every 60s)
        ↓
scheduler.service  →  DuckDB processing (daily 6:05am EST)
        ↓
GCS processed parquet  →  Backtester / Strategy research
```

---

## Infrastructure

| Component | Details |
|---|---|
| Cloud | GCP project: `hedge-fund-494103` |
| VM | `data-collector`, `e2-micro`, `asia-south1-c` |
| Storage | GCS bucket: `hedge-fund-494103-marketdata` |
| Database | BigQuery dataset: `marketdata` |
| Broker | Zerodha Kite Connect |
| Auth | GCP Secret Manager |

**SSH to VM:**
```bash
gcloud compute ssh data-collector --zone=asia-south1-c --project=hedge-fund-494103
```

---

## Data Collection

### Instruments (261 total)
```
Equity Futures:   89  (5 index + 84 stock futures, nearest expiry)
NIFTY Options:    84  (ATM ± 10 strikes, 2 expiries - weekly + monthly)
BANKNIFTY Opts:   84  (ATM ± 10 strikes, 2 expiries)
USDINR Futures:    2  (nearest 2 expiries, CDS segment)
NIFTY Spot:        1  (NSE index quote, token 256265)
BANKNIFTY Spot:    1  (NSE index quote, token 260105)
```

### GCS Structure
```
gs://hedge-fund-494103-marketdata/
  raw/
    orderbook/
      {SYMBOL}/
        {YYYY-MM-DD}/
          {HH-MM-SS}.parquet    ← raw ticks, flushed every 60s

  processed/
    features/
      {SYMBOL}/
        {YYYY-MM-DD}.parquet    ← 33 features per 1-second bar (futures)
    options/
      {UNDERLYING}/
        {YYYY-MM-DD}.parquet    ← Greeks + microstructure (options)
```

### Processed Features (futures - 33 columns)
```
Price:        open, high, low, close, vwap
Volume:       volume, tick_count, oi
Spread:       spread_mean, spread_max, spread_bps
Imbalance:    imbalance_mean, imbalance_std, imbalance_last
Depth:        total_bid_qty, total_ask_qty, weighted_mid
Rolling vol:  realized_vol_10s, realized_vol_30s, realized_vol_60s, realized_vol_300s
Imbalance MA: imbalance_ma_10s, imbalance_ma_30s, imbalance_ma_60s
Momentum:     price_mom_10s, price_mom_30s, price_mom_60s
Other:        price_impact, volume_ratio, spread_zscore, ts_ist
```

### Processed Features (options - additional columns)
```
Greeks:       iv, delta, gamma, vega, theta
IV features:  iv_ma_60s, iv_zscore
Metadata:     strike, opt_type, expiry, underlying, tte, moneyness
Cross-asset:  spot_price
```

---

## Services (VM)

```bash
# Check status
sudo systemctl status collector scheduler

# Restart
sudo systemctl restart collector
sudo systemctl restart scheduler

# Watch logs
sudo journalctl -u collector -f
sudo journalctl -u scheduler -f
```

### collector.service
- Connects to Zerodha WebSocket
- Subscribes to 261 instruments
- Flushes ticks to GCS every 60 seconds
- Auto-reconnects on disconnect

### scheduler.service
- Runs daily at 6:05am EST (after Indian market close)
- Processes all unprocessed futures via `duckdb_pipeline.py`
- Processes all unprocessed options via `options_pipeline.py`
- Parallel processing (8 workers for futures, sequential for options)

---

## Daily Token Routine
Zerodha access token expires nightly. Regenerate each evening:

```bash
# On laptop (11:30pm EST before market opens)
cd C:\projects\quant-fund
venv\Scripts\activate
python src\utils\auth.py
# Complete browser login
# Token auto-saves to Secret Manager
```

---

## Backtesting

### Run a backtest
```bash
python backtest.py --symbol NIFTY26MAYFUT \
                   --start 2026-04-30 \
                   --end 2026-04-30

# With custom parameters
python backtest.py --symbol NIFTY26MAYFUT \
                   --start 2026-04-30 \
                   --end 2026-04-30 \
                   --gamma 0.1 \
                   --max-inventory 5
```

### Backtest engine files
```
src/backtest/
  data_loader.py       ← GCS → DuckDB → bar iterator
  order_book.py        ← FIFO fill simulation
  strategy.py          ← abstract base class
  mm_strategy.py       ← Avellaneda-Stoikov MM strategy
  transaction_costs.py ← STT, brokerage, exchange fees
  metrics.py           ← PnL, Sharpe, drawdown, win rate
  engine.py            ← wires everything together
backtest.py            ← CLI entry point
```

---

## Project Structure
```
quant-fund/
  src/
    collection/
      collector.py          ← WebSocket tick collector
      brokers/
        zerodha.py          ← Zerodha implementation
        shoonya.py          ← Shoonya (pending activation)
    processing/
      duckdb_pipeline.py    ← futures processing pipeline
      options_pipeline.py   ← options processing + Greeks
      scheduler.py          ← daily processing scheduler
    storage/
      gcs.py                ← GCS upload/download helpers
    backtest/
      (see above)
    utils/
      auth.py               ← Zerodha + Secret Manager auth
  config/
    symbols.py              ← instrument definitions + auto-rollover
  deploy/
    collector.service       ← systemd service
    scheduler.service       ← systemd service
  test/                     ← test and analysis scripts
  backtest.py               ← CLI entry point
```

---

## Transaction Costs (post Budget 2026)

| Cost | Rate | Side |
|---|---|---|
| STT | 0.05% | Sell only (futures) |
| Exchange | 0.00173% | Both sides |
| SEBI | 0.0001% | Both sides |
| GST | 18% | On brokerage+exchange+SEBI |
| Brokerage | ₹20 flat | Per order (Zerodha) |
| Stamp | 0.002% | Buy only |

**Breakeven spreads:**
```
NIFTY Futures:    14.08 pts  ← unviable for retail MM
NIFTY 0DTE opts:  0.67 pts  ← viable (tiny STT on low premium)
USDINR Futures:   0.05 paise ← viable (zero STT)
```

---

## Strategies

### Built
- **Avellaneda-Stoikov MM** (`mm_strategy.py`) - futures market making
  - Confirmed unviable on futures due to STT
  - Framework ready for options/currency

### Planned
- **USDINR MM** - currency futures market making (zero STT)
- **NIFTY 0DTE MM** - weekly expiry options on Tuesday
- **Options MM** - monthly NIFTY/BANKNIFTY options

### Research findings (Apr 30 NIFTY data)
```
Mean reversion signal:  t-stat=16, win_rate=59%, avg=3.84pts
Breakeven needed:       14.08 pts
Conclusion:             Signal exists but STT kills profitability
More data needed:       30+ days for reliable strategy development
```

---

## Pending Items

```
Shoonya API:     Account FN210061, credentials in Secret Manager
                 Awaiting API activation (server 502)
                 Once active: switch ACTIVE_BROKER = "shoonya"

Python:          Still on 3.10.9/3.10.12 — upgrade to 3.11+
                 (FutureWarning from Google libraries)

Options backtest: Build after 2-3 weeks of options data

USDINR strategy: Build after 1 week of USDINR data

0DTE analysis:   Run after Tuesday May 5 (first full day)
```

---

## Key Commands

```bash
# Check GCS file count
python -c "from src.storage.gcs import list_files; print(len(list_files()))"

# Run futures pipeline manually
python src/processing/duckdb_pipeline.py

# Run options pipeline manually
python src/processing/options_pipeline.py

# Signal analysis
python test/signal_analysis.py
python test/signal_analysis2.py

# Check options data
python test/check_options_data.py

# Find available instruments
python test/find_options.py
python test/find_currency.py
python test/find_weekly_options.py
```

---

## GCP Secrets

```
KITE_API_KEY
KITE_API_SECRET
KITE_ACCESS_TOKEN        ← regenerate nightly
SHOONYA_USER_ID
SHOONYA_PASSWORD
SHOONYA_TOTP_SECRET
SHOONYA_VENDOR_CODE
SHOONYA_API_KEY
SHOONYA_IMEI
```
