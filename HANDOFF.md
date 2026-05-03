# Quant Fund — Complete Handoff Document
**Last updated:** May 2, 2026  
**Author:** Abhimanyu Chaudhari  
**Location:** Newark, NJ  
**GitHub:** github.com/AbhimanyuChaudhari/quant-fund

---

## What We're Building
A market making hedge fund focused on Indian derivatives markets.  
Solo project, bootstrapped, running from Newark NJ.  
Primary targets: USDINR Futures (zero STT) and NIFTY 0DTE Options (low STT).

---

## Current Status
```
Infrastructure:    COMPLETE — collecting 261 instruments live
Backtesting:       COMPLETE — with realistic fill simulation
Live trading:      COMPLETE — paper-live mode ready for Monday
USDINR strategy:   PENDING — need Monday's data to backtest
0DTE strategy:     PENDING — need Tuesday May 5 data
```

---

## Tech Stack
```
Language:    Python 3.10
Cloud:       GCP (project: hedge-fund-494103)
Storage:     GCS bucket: hedge-fund-494103-marketdata
Database:    BigQuery dataset: marketdata
Broker:      Zerodha Kite Connect (data + execution)
Processing:  DuckDB + pandas
Auth:        GCP Secret Manager
VM:          e2-micro, asia-south1-c, name: data-collector
```

---

## Project Structure
```
quant-fund/
  src/
    collection/
      collector.py          ← WebSocket tick collector (261 instruments)
      brokers/
        zerodha.py          ← Zerodha implementation
        shoonya.py          ← Shoonya (pending activation)
    processing/
      duckdb_pipeline.py    ← futures processing (53 cols including L5 depth)
      options_pipeline.py   ← options processing + Greeks
      scheduler.py          ← daily processing at 6:05am EST
    storage/
      gcs.py                ← GCS upload/download helpers
    backtest/
      data_loader.py        ← GCS → DuckDB → bar iterator
      order_book.py         ← simple fill simulation
      strategy.py           ← abstract base class
      mm_strategy.py        ← Avellaneda-Stoikov MM strategy
      transaction_costs.py  ← STT, brokerage, fees (all instruments)
      metrics.py            ← PnL, Sharpe, drawdown
      engine.py             ← backtest engine
    trading/
      risk.py               ← daily loss limit, kill switch, position limits
      portfolio.py          ← live position + PnL tracking
      order_manager.py      ← paper + live order placement
      feature_engine.py     ← real-time features from live ticks
      engine.py             ← live trading engine (3 modes)
      fill_simulator.py     ← realistic fill simulation (L1+L2+L3)
    utils/
      auth.py               ← Zerodha + Secret Manager auth
  config/
    symbols.py              ← 261 instruments, auto-rollover
  test/                     ← all test and analysis scripts
  backtest.py               ← backtest CLI entry point
  trade_runner.py           ← live trading CLI entry point
  README.md                 ← infrastructure docs
```

---

## Data Collection (261 instruments)
```
Equity Futures:    89  (5 index + 84 stocks, nearest expiry)
NIFTY Options:     84  (ATM ±10 strikes, 2 expiries)
BANKNIFTY Options: 84  (ATM ±10 strikes, 2 expiries)
USDINR Futures:     2  (nearest 2 expiries)
NIFTY Spot:         1  (token 256265)
BANKNIFTY Spot:     1  (token 260105)
```

### GCS Structure
```
gs://hedge-fund-494103-marketdata/
  raw/
    orderbook/{SYMBOL}/{YYYY-MM-DD}/{HH-MM-SS}.parquet
  processed/
    features/{SYMBOL}/{YYYY-MM-DD}.parquet   ← 53 cols (futures)
    options/{UNDERLYING}/{YYYY-MM-DD}.parquet ← Greeks + microstructure
```

### Processed Features (53 columns — futures)
```
Base:        symbol, ts_sec, ts_ist
OHLCV:       open, high, low, close, volume, tick_count, vwap, oi
Spread:      spread_mean, spread_max, spread_bps
Imbalance:   imbalance_mean, imbalance_std, imbalance_last
Depth agg:   total_bid_qty, total_ask_qty, weighted_mid, price_impact
L5 depth:    bid_p1..5, bid_q1..5, ask_p1..5, ask_q1..5
Rolling vol: realized_vol_10s/30s/60s/300s
Imb MA:      imbalance_ma_10s/30s/60s
Other:       spread_zscore, volume_ratio
Momentum:    price_mom_10s/30s/60s
```

---

## VM Services
```bash
# SSH into VM
gcloud compute ssh data-collector --zone=asia-south1-c --project=hedge-fund-494103

# Check both services
sudo systemctl status collector scheduler

# Restart services
sudo systemctl restart collector
sudo systemctl restart scheduler

# Watch live logs
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
- Processes futures via duckdb_pipeline.py (8 parallel workers)
- Processes options via options_pipeline.py (sequential, Greeks)

---

## Daily Token Routine
Zerodha token expires nightly. Must regenerate before market opens:

```bash
# Run on laptop at 11:30pm EST (before 9am IST market open)
cd C:\projects\quant-fund
venv\Scripts\activate
python src\utils\auth.py
# Opens browser → login → token auto-saves to Secret Manager
# VM picks it up automatically
```

---

## Key Commands (Laptop)

### Setup
```bash
cd C:\projects\quant-fund
venv\Scripts\activate
```

### Backtesting
```bash
# Simple backtest (fills every price crossing)
python backtest.py --symbol NIFTY26MAYFUT --start 2026-04-30 --end 2026-04-30

# Realistic backtest (queue + volume simulation) ← USE THIS
python backtest.py --symbol NIFTY26MAYFUT --start 2026-04-30 --end 2026-04-30 --realistic

# USDINR backtest (after Monday data)
python backtest.py --symbol USDINR26508FUT --start 2026-05-05 --end 2026-05-05 \
                   --realistic --lot-size 1000

# Custom parameters
python backtest.py --symbol NIFTY26MAYFUT --start 2026-04-30 --end 2026-04-30 \
                   --realistic --gamma 0.1 --max-inventory 5 --queue-aggression 0.3
```

### Live Trading
```bash
# Paper trading on historical data (offline, any time)
python trade_runner.py --symbol NIFTY26MAYFUT --paper

# Paper trading on LIVE ticks (run during market hours 9:15-15:30 IST)
python trade_runner.py --symbol NIFTY26MAYFUT --paper-live

# USDINR paper-live (run during CDS hours 9:00-17:00 IST)
python trade_runner.py --symbol USDINR26508FUT --paper-live --lot-size 1000

# Live trading (real money — only after extensive paper validation)
python trade_runner.py --symbol NIFTY26MAYFUT --live
```

### Data Checks
```bash
# Check GCS file counts
python test/check_gcs.py

# Check options data
python test/check_options_data.py

# Check USDINR data
python test/check_usdinr_data.py

# Check volume stats for a symbol
python test/check_volume.py

# Signal analysis (run after 30+ days of data)
python test/signal_analysis.py
python test/signal_analysis2.py
```

### Pipeline (manual)
```bash
# Process futures for a specific date
python src/processing/duckdb_pipeline.py

# Process options (NIFTY + BANKNIFTY)
python src/processing/options_pipeline.py

# Find available instruments
python test/find_options.py
python test/find_currency.py
python test/find_weekly_options.py
```

---

## Key VM Commands
```bash
# Pull latest code and restart services
cd quant-fund && git pull
sudo systemctl restart collector scheduler

# Manually process a symbol on VM
python src/processing/duckdb_pipeline.py

# Check GCS raw file count
python -c "from src.storage.gcs import list_files; print(len(list_files()))"
```

---

## Transaction Costs (Critical)
```
Instrument         Breakeven Spread    STT        Viable?
─────────────────────────────────────────────────────────
NIFTY Futures      14.08 pts           Rs.895/sell   NO
BANKNIFTY Futures  30.07 pts           Rs.892/sell   NO
USDINR Futures      0.05 paise         ZERO         YES ✓
NIFTY Options       0.66 pts           Rs.1.88/sell YES ✓
USDINR Options      0.05 paise         ZERO         YES ✓
```

**Budget 2026 raised STT on equity futures to 0.05%** — this is why
NIFTY futures MM is unviable at retail level. Focus on USDINR and options.

---

## Strategy: Avellaneda-Stoikov (2008)
```
Paper: "High-Frequency Trading in a Limit Order Book"

Reservation price: r = mid - q × gamma × sigma² × T
Optimal spread:    s = gamma × sigma² × T + (2/gamma) × ln(1 + gamma/kappa)
Bid = r - s/2
Ask = r + s/2

Parameters:
  gamma:  risk aversion (NIFTY: 0.1, USDINR: TBD after Monday data)
  kappa:  order arrival rate (default: 1.5)
  T:      fraction of session remaining (0.0 → 1.0)
  sigma:  realized_vol_60s from processed features
```

### Backtest Results (Apr 30, NIFTY26MAYFUT, realistic fills)
```
Fill rate:      3.9%  (922 / 23,746 attempts)
Gross PnL:     +Rs.1,58,850
Total costs:   -Rs.4,58,904  (STT kills it)
Net PnL:       -Rs.3,00,054
Sharpe:         7.27 (gross only, one day)
Win rate:       72.2%

Conclusion: Strategy has real edge (positive gross PnL)
            STT is the only problem on futures
            Same strategy on USDINR should be profitable
```

---

## Fill Simulation (Realistic)
```
Level 1 — Price crossing:
  BUY filled if bar.low  <= our bid
  SELL filled if bar.high >= our ask

Level 2 — Activity filter:
  Enough ticks must trade at our price level

Level 3 — Queue simulation (uses L5 depth):
  ticks_at_level  = tick_count × price_fraction
  ticks_to_clear  = queue_shares / 75
  fill_prob       = ticks_at_level / ticks_to_clear × aggression

Calibrated to real NIFTY data:
  avg_bid_q1:     427 shares
  avg_ticks/bar:  1.36
  fill_rate:      ~20% (realistic for MM)
```

---

## Risk Management
```
Daily loss limit:   Rs.10,000  (hard stop, no trading after)
Max position:       2 lots     (per symbol)
Max gross exposure: 4 lots     (all symbols combined)
Cooldown period:    300s       (after loss limit hit)
Kill switch:        manual     (activate_kill_switch())
```

---

## GCP Secrets
```
KITE_API_KEY
KITE_API_SECRET
KITE_ACCESS_TOKEN     ← regenerate nightly
SHOONYA_USER_ID
SHOONYA_PASSWORD
SHOONYA_TOTP_SECRET
SHOONYA_VENDOR_CODE
SHOONYA_API_KEY
SHOONYA_IMEI
```

---

## Pending Items
```
Shoonya API:     Account FN210061, KYC done
                 Server returning 502 — awaiting activation
                 Once active: ACTIVE_BROKER = "shoonya" in collector.py
                 Full automation (no nightly token refresh)

Python:          Still on 3.10.9 — upgrade to 3.11+
                 (FutureWarning from Google libraries)

USDINR strategy: Wait for Monday May 5 data
                 Run: python backtest.py --symbol USDINR26508FUT \
                        --start 2026-05-05 --end 2026-05-05 --realistic

0DTE strategy:   Wait for Tuesday May 5 (NIFTY weekly expiry)
                 Run: python test/check_options_data.py after market close

ML layer:        After 30+ days of data
                 Gradient boosting on 53 features → signal layer
                 RL only after ML works consistently

Level 3 queue:   Improve fill simulator with tick-by-tick data
                 Currently uses 1s bars — more data needed
```

---

## Next Steps (This Week)
```
Monday May 5 (EST morning):
  → Nightly token refresh (11:30pm Sunday EST)
  → Run paper-live:
    python trade_runner.py --symbol NIFTY26MAYFUT --paper-live
  → After market close: check USDINR data
    python test/check_usdinr_data.py
  → Run USDINR backtest:
    python backtest.py --symbol USDINR26508FUT \
      --start 2026-05-05 --end 2026-05-05 --realistic --lot-size 1000

Tuesday May 5 (0DTE day):
  → NIFTY weekly expiry — high volume options day
  → After close: run options pipeline
    python src/processing/options_pipeline.py
  → Analyze 0DTE data

End of week:
  → Run backtest across all available days
  → Compare USDINR vs NIFTY options performance
  → Decide primary strategy to go live with
```

---

## For New Developers
If you're joining this project, do this first:

```bash
# 1. Clone repo
git clone https://github.com/AbhimanyuChaudhari/quant-fund
cd quant-fund

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up GCP auth
gcloud auth application-default login

# 5. Test GCS connection
python test/test_gcs.py

# 6. Run a backtest (no market hours needed)
python backtest.py --symbol NIFTY26MAYFUT \
                   --start 2026-04-30 --end 2026-04-30 --realistic

# 7. Read these files in order:
#    src/backtest/strategy.py        ← understand base strategy
#    src/backtest/mm_strategy.py     ← understand A-S model
#    src/trading/risk.py             ← understand risk management
#    src/trading/engine.py           ← understand live engine
```

### Key concepts to understand first:
```
1. Avellaneda-Stoikov model (read the 2008 paper)
2. Market making vs directional trading
3. STT and why it matters for strategy selection
4. Queue-based fill simulation
5. NSE market hours: 9:15-15:30 IST (equity), 9:00-17:00 IST (currency)
```