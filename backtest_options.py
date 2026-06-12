"""
Options Backtest with Delta Hedging

Backtests 0DTE options MM with:
  - ATM strike filtering
  - Inventory limits per symbol
  - Simple price crossing fills
  - Delta hedging via futures
  - Correct transaction costs for both options + futures

Usage:
    python backtest_options.py --underlying BANKNIFTY --date 2026-05-05
    python backtest_options.py --underlying NIFTY --date 2026-05-06
"""

import argparse
import math
import logging
import gcsfs
import duckdb
import pandas as pd
from collections import defaultdict
from datetime import date
from src.backtest.options_data_loader import iter_option_bars
from src.backtest.transaction_costs import TransactionCosts
from src.trading.delta_hedger import DeltaHedger

logging.basicConfig(level=logging.WARNING)

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

LOT_SIZES = {
    "NIFTY":     75,
    "BANKNIFTY": 35,
    "FINNIFTY":  40,
}

FUTURES_SYMBOLS = {
    "NIFTY":     "NIFTY26MAYFUT",
    "BANKNIFTY": "BANKNIFTY26MAYFUT",
}


# ─────────────────────────────────────
# Futures price loader
# ─────────────────────────────────────
def load_futures_prices(underlying: str, date_str: str) -> dict:
    """
    Load futures prices keyed by ts_sec.
    Returns {ts_sec: close_price}
    """
    symbol = FUTURES_SYMBOLS.get(underlying)
    if not symbol:
        return {}

    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)

    path = (f"gs://{BUCKET_NAME}/processed/features/"
            f"{symbol}/{date_str}.parquet")

    if not fs.exists(path.replace("gs://", "")):
        print(f"  No futures data for {symbol} {date_str} — no delta hedge")
        return {}

    try:
        df = con.execute(f"""
            SELECT ts_sec, close, high, low
            FROM read_parquet('{path}')
            ORDER BY ts_sec
        """).df()
        return {
            int(row.ts_sec): {
                "close": float(row.close),
                "high":  float(row.high),
                "low":   float(row.low),
            }
            for _, row in df.iterrows()
        }
    except Exception as e:
        print(f"  Could not load futures: {e}")
        return {}


# ─────────────────────────────────────
# Main backtest
# ─────────────────────────────────────
def run_options_backtest(underlying:       str,
                         date_str:         str,
                         lot_size:         int   = None,
                         min_premium:      float = 5.0,
                         max_premium:      float = 500.0,
                         max_delta:        float = 0.65,
                         max_inventory:    int   = 5,
                         spread_mult:      float = 1.5,
                         min_spread:       float = 1.0,
                         hedge_threshold:  float = None,
                         skip_open_mins:   int   = 0):

    if lot_size is None:
        lot_size = LOT_SIZES.get(underlying, 75)

    # Hedge threshold = half a lot
    if hedge_threshold is None:
        hedge_threshold = lot_size * 0.5

    print(f"\n{'='*55}")
    print(f"  Options Backtest — {underlying} | {date_str}")
    print(f"{'='*55}")
    print(f"  Lot size:         {lot_size}")
    print(f"  Premium range:    Rs.{min_premium} - Rs.{max_premium}")
    print(f"  Max |delta|:      {max_delta}")
    print(f"  Max inventory:    {max_inventory} lots per symbol")
    print(f"  Spread mult:      {spread_mult}x")
    print(f"  Min spread:       {min_spread} pts")
    print(f"  Skip open:        {skip_open_mins} minutes")
    print(f"  Hedge threshold:  {hedge_threshold} shares")
    print(f"{'='*55}\n")

    # Transaction costs
    tc_options = TransactionCosts(
        lot_size        = lot_size,
        instrument_type = "equity_options",
    )
    tc_futures = TransactionCosts(
        lot_size        = lot_size,
        instrument_type = "equity_futures",
    )

    # Delta hedger
    futures_symbol = FUTURES_SYMBOLS.get(underlying, "")
    hedger = DeltaHedger(
        futures_symbol  = futures_symbol,
        lot_size        = lot_size,
        hedge_threshold = hedge_threshold,
    )

    # ── Load futures prices for hedging ───────────
    print(f"Loading futures prices for delta hedging...")
    futures_prices = load_futures_prices(underlying, date_str)
    has_hedge = len(futures_prices) > 0
    print(f"  Futures bars: {len(futures_prices):,} "
          f"({'with hedge' if has_hedge else 'NO HEDGE — futures data missing'})\n")

    # ── Tracking ───────────────────────────────────
    total_fills       = 0
    buy_fills         = 0
    sell_fills        = 0
    hedge_fills       = 0
    realized_pnl      = 0.0
    options_costs     = 0.0
    futures_costs     = 0.0
    bars_processed    = 0
    bars_quoted       = 0
    bars_filtered     = 0
    bars_skipped_open = 0

    inventory:  dict[str, int]   = defaultdict(int)
    avg_price:  dict[str, float] = defaultdict(float)
    symbol_pnl: dict[str, float] = defaultdict(float)

    # Market open seconds (IST = UTC + 5:30 = UTC + 19800)
    MARKET_OPEN_UTC = 9 * 3600 + 15 * 60  # 09:15 IST in IST seconds
    skip_until_ist  = MARKET_OPEN_UTC + skip_open_mins * 60

    print(f"Loading options chain...")
    chains = list(iter_option_bars(
        underlying        = underlying,
        date_str          = date_str,
        market_hours_only = True,
        min_premium       = min_premium,
    ))

    if not chains:
        print(f"No data found for {underlying} {date_str}")
        return

    print(f"Loaded {len(chains):,} timestamps\n")
    print(f"Running backtest...")

    for ts_sec, chain in chains:
        bars_processed += 1

        # IST seconds since midnight
        ist_sec = (ts_sec + 19800) % 86400

        # Skip opening minutes
        if ist_sec < skip_until_ist:
            bars_skipped_open += 1
            continue

        # Get futures price for this timestamp
        fut_bar    = futures_prices.get(ts_sec, {})
        fut_price  = fut_bar.get("close", 0)

        for _, row in chain.iterrows():
            symbol  = str(row["symbol"])
            premium = float(row.get("close",       0) or 0)
            spread  = float(row.get("spread_mean", 2) or 2)
            delta   = float(row.get("delta",       0) or 0)
            gamma   = float(row.get("gamma",       0) or 0)
            iv      = float(row.get("iv",          0) or 0)
            bar_low = float(row.get("low",  premium * 0.995) or premium * 0.995)
            bar_high= float(row.get("high", premium * 1.005) or premium * 1.005)

            # ── Filters ───────────────────────────
            if premium < min_premium or premium > max_premium:
                bars_filtered += 1
                continue
            if abs(delta) > max_delta:
                bars_filtered += 1
                continue
            if iv <= 0:
                bars_filtered += 1
                continue
            if math.isnan(premium) or math.isnan(delta):
                bars_filtered += 1
                continue

            bars_quoted += 1
            curr_inv = inventory[symbol]

            # Update Greeks in hedger
            hedger.update_greeks(symbol, {
                "delta": delta,
                "gamma": gamma,
                "iv":    iv,
            })

            # ── Quote computation ──────────────────
            quote_spread = max(min_spread, spread * spread_mult)
            quote_spread = min(quote_spread, premium * 0.25)
            bid_price    = round(premium - quote_spread / 2, 1)
            ask_price    = round(premium + quote_spread / 2, 1)

            if bid_price <= 0:
                continue

            # ── BUY fill ───────────────────────────
            if curr_inv < max_inventory and bar_low <= bid_price:
                cost           = tc_options.compute(bid_price, 1, "BUY").total
                options_costs += cost
                total_fills   += 1
                buy_fills     += 1

                if curr_inv < 0:
                    pnl = (avg_price[symbol] - bid_price) \
                          * abs(curr_inv) * lot_size
                    realized_pnl       += pnl
                    symbol_pnl[symbol] += pnl
                    inventory[symbol]  += 1
                else:
                    total_units        = curr_inv * avg_price.get(symbol, 0)
                    inventory[symbol] += 1
                    avg_price[symbol]  = (total_units + bid_price) \
                                         / inventory[symbol]

                # Update hedger position
                hedger.update_option_position(symbol, delta, +1)

            # ── SELL fill ──────────────────────────
            if curr_inv > -max_inventory and bar_high >= ask_price:
                cost           = tc_options.compute(ask_price, 1, "SELL").total
                options_costs += cost
                total_fills   += 1
                sell_fills    += 1

                if curr_inv > 0:
                    pnl = (ask_price - avg_price[symbol]) \
                          * curr_inv * lot_size
                    realized_pnl       += pnl
                    symbol_pnl[symbol] += pnl
                    inventory[symbol]  -= 1
                else:
                    total_units        = abs(curr_inv) * avg_price.get(symbol, 0)
                    inventory[symbol] -= 1
                    avg_price[symbol]  = (total_units + ask_price) \
                                         / abs(inventory[symbol])

                # Update hedger position
                hedger.update_option_position(symbol, delta, -1)

        # ── Delta hedge check (once per timestamp) ─
        if has_hedge and fut_price > 0:
            action = hedger.compute_hedge()
            if action.needed:
                hedge_fills += 1
                hedge_cost   = tc_futures.compute(
                    fut_price, action.lots, action.side
                ).total
                futures_costs += hedge_cost

                # Simulate hedge fill at current futures price
                # Futures MM edge is tiny so approximate as market order
                if action.side == "SELL":
                    hedge_pnl = 0  # opening hedge, no PnL yet
                    hedger.update_futures_position(-action.lots)
                else:
                    hedge_pnl = 0
                    hedger.update_futures_position(+action.lots)

    # ── Results ───────────────────────────────────
    total_costs  = options_costs + futures_costs
    net_pnl      = realized_pnl - total_costs
    open_lots    = sum(abs(v) for v in inventory.values() if v != 0)
    open_symbols = sum(1 for v in inventory.values() if v != 0)
    hedge_status = hedger.status()
    top_symbols  = sorted(symbol_pnl.items(),
                          key=lambda x: x[1], reverse=True)[:5]

    avg_prem = (min_premium + max_premium) / 2
    rt = tc_options.round_trip(avg_prem, 1)
    be = tc_options.breakeven_spread(avg_prem, 1)

    print(f"\n{'─'*55}")
    print(f"  Options Backtest Results — {underlying}")
    print(f"  {date_str}")
    print(f"{'─'*55}")
    print(f"  Gross PnL:         Rs.{realized_pnl:>12,.2f}")
    print(f"  Options costs:    -Rs.{options_costs:>12,.2f}")
    print(f"  Futures costs:    -Rs.{futures_costs:>12,.2f}")
    print(f"  Total costs:      -Rs.{total_costs:>12,.2f}")
    print(f"  Net PnL:           Rs.{net_pnl:>12,.2f}")
    print(f"{'─'*55}")
    print(f"  Options fills:     {total_fills:>8,}  "
          f"(buy={buy_fills}, sell={sell_fills})")
    print(f"  Hedge trades:      {hedge_fills:>8,}")
    print(f"{'─'*55}")
    print(f"  Bars processed:    {bars_processed:>8,}")
    print(f"  Bars quoted:       {bars_quoted:>8,}")
    print(f"  Bars filtered:     {bars_filtered:>8,}")
    print(f"  Bars skipped open: {bars_skipped_open:>8,}")
    print(f"{'─'*55}")
    print(f"  Open option lots:  {open_lots:>8,}")
    print(f"  Open symbols:      {open_symbols:>8,}")
    print(f"  Net delta (shares):{hedge_status['net_delta_shares']:>8.1f}")
    print(f"  Futures hedge lots:{hedge_status['futures_lots']:>8,}")
    print(f"{'─'*55}")

    if top_symbols:
        print(f"\n  Top 5 symbols by PnL:")
        for sym, pnl in top_symbols:
            sign = "+" if pnl >= 0 else ""
            print(f"    {sym:<32} Rs.{sign}{pnl:,.2f}")

    print(f"\n  Cost structure (avg premium Rs.{avg_prem:.0f}):")
    print(f"    Options RT:       Rs.{rt:.2f}")
    print(f"    Breakeven spread: {be:.2f} pts")
    print(f"    Min spread quoted:{min_spread:.1f} pts")
    viable = "YES" if be < min_spread else "NO — widen spread"
    print(f"    Viable:           {viable}")
    print(f"    Delta hedged:     {'YES' if has_hedge else 'NO (no futures data)'}")

    if len(chains) < 200:
        print(f"\n  WARNING: Only {len(chains)} timestamps")
        print(f"  Not representative — need full trading day")
        print(f"  Run again after May 6 data is processed")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Options MM backtest with delta hedging"
    )
    parser.add_argument("--underlying",      default="BANKNIFTY")
    parser.add_argument("--date",            default="2026-05-05")
    parser.add_argument("--lot-size",        type=int,   default=None)
    parser.add_argument("--min-premium",     type=float, default=5.0)
    parser.add_argument("--max-premium",     type=float, default=500.0)
    parser.add_argument("--max-delta",       type=float, default=0.65)
    parser.add_argument("--max-inventory",   type=int,   default=5)
    parser.add_argument("--spread-mult",     type=float, default=1.5)
    parser.add_argument("--min-spread",      type=float, default=1.0)
    parser.add_argument("--skip-open-mins",  type=int,   default=0,
                        help="Skip first N minutes after open")
    args = parser.parse_args()

    run_options_backtest(
        underlying      = args.underlying,
        date_str        = args.date,
        lot_size        = args.lot_size,
        min_premium     = args.min_premium,
        max_premium     = args.max_premium,
        max_delta       = args.max_delta,
        max_inventory   = args.max_inventory,
        spread_mult     = args.spread_mult,
        min_spread      = args.min_spread,
        skip_open_mins  = args.skip_open_mins,
    )


if __name__ == "__main__":
    main()
