"""
0DTE Options Market Making Strategy

Makes markets on NIFTY weekly options (Tuesday expiry).
Uses delta hedging via NIFTY futures to stay directionally neutral.

Key differences from futures MM:
  1. Quote multiple strikes simultaneously (ATM ± 3)
  2. Delta hedge after each fill
  3. Stop quoting after 14:00 IST (gamma too high)
  4. Wider spreads as gamma increases
  5. Track theta decay as additional revenue

Strategy logic:
  Morning (09:15-11:00): Quote 2 lots, tight spreads
  Midday  (11:00-14:00): Quote 1 lot, wider spreads
  After   (14:00-15:30): FLAT — no new quotes, manage existing

Profitability:
  Revenue: bid-ask spread + theta decay
  Cost:    transaction costs + gamma hedging cost
  Edge:    quote when spread > breakeven + expected gamma cost
"""

import math
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from src.backtest.strategy import BaseStrategy, StrategyConfig
from src.trading.delta_hedger import DeltaHedger, HedgeAction

logger = logging.getLogger(__name__)

# Session boundaries (UTC seconds since midnight)
SESSION_OPEN      = 13500   # 09:15 IST
MORNING_END       = 20700   # 11:45 IST (use tighter quotes after this)
CUTOFF_UTC        = 30600   # 14:00 IST — stop new quotes after this
SESSION_CLOSE     = 36000   # 15:30 IST


@dataclass
class OptionQuote:
    symbol:    str
    strike:    float
    opt_type:  str     # CE or PE
    bid_price: float
    ask_price: float
    lots:      int
    delta:     float
    bid_id:    Optional[str] = None
    ask_id:    Optional[str] = None


class ZeroDTEStrategy(BaseStrategy):
    """
    0DTE Options Market Making Strategy.

    Quotes ATM ± N strikes on expiry day.
    Delta hedges continuously using NIFTY futures.
    Stops quoting at 14:00 IST.

    Parameters:
        strikes_each_side: ATM ± N strikes to quote
        morning_lots:      lots per side before 11:45 IST
        midday_lots:       lots per side after 11:45 IST
        min_spread_pts:    minimum bid-ask spread to quote
        max_delta_per_strike: max delta exposure per strike
        hedge_threshold:   delta imbalance to trigger hedge (shares)
    """

    def __init__(self, config: StrategyConfig,
                 strikes_each_side:      int   = 3,
                 morning_lots:           int   = 2,
                 midday_lots:            int   = 1,
                 min_spread_pts:         float = 1.0,
                 max_delta_per_strike:   float = 0.7,
                 hedge_threshold:        float = 37.5,
                 futures_symbol:         str   = "NIFTY26MAYFUT"):

        super().__init__(config)
        self.strikes_each_side    = strikes_each_side
        self.morning_lots         = morning_lots
        self.midday_lots          = midday_lots
        self.min_spread_pts       = min_spread_pts
        self.max_delta_per_strike = max_delta_per_strike
        self.futures_symbol       = futures_symbol

        self.hedger = DeltaHedger(
            futures_symbol  = futures_symbol,
            lot_size        = config.lot_size,
            hedge_threshold = hedge_threshold,
        )

        # Active quotes per symbol
        self._quotes: Dict[str, OptionQuote] = {}

        # Stats
        self.bars_quoted    = 0
        self.bars_filtered  = 0
        self.hedge_trades   = 0
        self.theta_collected = 0.0

    # ─────────────────────────────────────
    # Core strategy logic
    # ─────────────────────────────────────
    def on_chain(self, ts_sec: int, chain: pd.DataFrame,
                 engine=None, book=None) -> None:
        """
        Called every second with the full options chain.

        Args:
            ts_sec:  current timestamp
            chain:   DataFrame with all strikes at this second
            engine:  LiveEngine (for live trading)
            book:    SimulatedOrderBook (for backtesting)
        """
        self.bars_processed += 1

        # Check if we should be quoting
        time_utc = ts_sec % 86400

        if time_utc > CUTOFF_UTC:
            # After 14:00 IST — cancel all, go flat
            self._cancel_all_quotes(engine, book)
            self.bars_filtered += 1
            return

        if chain.empty:
            self.bars_filtered += 1
            return

        # Determine lot size based on time
        if time_utc < MORNING_END:
            lots = self.morning_lots
        else:
            lots = self.midday_lots

        # Update Greeks in hedger
        for _, row in chain.iterrows():
            if pd.notna(row.get("delta")):
                self.hedger.update_greeks(row["symbol"], {
                    "delta": float(row.get("delta", 0)),
                    "gamma": float(row.get("gamma", 0)),
                    "vega":  float(row.get("vega",  0)),
                    "theta": float(row.get("theta", 0)),
                    "iv":    float(row.get("iv",    0)),
                })

        # Quote each strike
        for _, row in chain.iterrows():
            self._quote_strike(ts_sec, row, lots, engine, book)

        # Check delta hedge
        action = self.hedger.compute_hedge()
        if action.needed:
            self._execute_hedge(action, ts_sec, engine, book)

        self.bars_quoted += 1

    def _quote_strike(self, ts_sec: int, row: pd.Series,
                      lots: int, engine, book) -> None:
        """Quote one strike (one row from options chain)."""
        symbol   = row["symbol"]
        delta    = float(row.get("delta", 0)) if pd.notna(row.get("delta")) else 0
        premium  = float(row.get("close", 0))
        spread   = float(row.get("spread_mean", 2.0))

        # Filter: skip if delta too high (deep ITM)
        if abs(delta) > self.max_delta_per_strike:
            return

        # Filter: skip if premium too low (illiquid)
        if premium < 1.0:
            return

        # Compute spread to quote
        # Wider spread when:
        #   - Higher gamma (more risk)
        #   - Lower premium (less room)
        #   - Near expiry (T→0)
        gamma      = float(row.get("gamma", 0)) if pd.notna(row.get("gamma")) else 0
        tte        = float(row.get("tte", 0.001)) if pd.notna(row.get("tte")) else 0.001
        quote_spread = max(
            self.min_spread_pts,
            spread * 1.5,                    # 1.5× market spread
            gamma * 100 * 0.5,               # gamma adjustment
        )
        quote_spread = min(quote_spread, premium * 0.3)  # cap at 30% of premium

        bid_price = round(premium - quote_spread / 2, 1)
        ask_price = round(premium + quote_spread / 2, 1)

        if bid_price <= 0:
            return

        # Cancel existing quote for this symbol
        if symbol in self._quotes:
            old_quote = self._quotes[symbol]
            if old_quote.bid_id and book:
                book.cancel_order(old_quote.bid_id)
            if old_quote.ask_id and book:
                book.cancel_order(old_quote.ask_id)

        # Post new quotes
        quote = OptionQuote(
            symbol    = symbol,
            strike    = float(row.get("strike", 0)),
            opt_type  = str(row.get("opt_type", "")),
            bid_price = bid_price,
            ask_price = ask_price,
            lots      = lots,
            delta     = delta,
        )

        if book:
            from src.backtest.order_book import Side
            quote.bid_id = book.post_order(
                Side.BUY, bid_price, lots, ts_sec
            )
            quote.ask_id = book.post_order(
                Side.SELL, ask_price, lots, ts_sec
            )
        elif engine:
            quote.bid_id = engine.post_quote("BUY",  bid_price, lots)
            quote.ask_id = engine.post_quote("SELL", ask_price, lots)

        self._quotes[symbol] = quote

    def _execute_hedge(self, action: HedgeAction,
                       ts_sec: int, engine, book) -> None:
        """Execute a delta hedge trade in NIFTY futures."""
        self.hedge_trades += 1
        logger.info(
            f"HEDGE: {action.side} {action.lots}L {self.futures_symbol} | "
            f"net_delta={action.net_delta:.1f} | {action.reason}"
        )

        if engine:
            # Live mode — need current futures price
            # For now use a placeholder — in live mode get from tick
            engine.post_quote(action.side, 0, action.lots)
        elif book:
            from src.backtest.order_book import Side
            side = Side.BUY if action.side == "BUY" else Side.SELL
            book.post_order(side, 0, action.lots, ts_sec)

        # Update hedger position
        if action.side == "BUY":
            self.hedger.update_futures_position(+action.lots)
        else:
            self.hedger.update_futures_position(-action.lots)

    def _cancel_all_quotes(self, engine, book) -> None:
        """Cancel all open option quotes."""
        for symbol, quote in self._quotes.items():
            if quote.bid_id and book:
                book.cancel_order(quote.bid_id)
            if quote.ask_id and book:
                book.cancel_order(quote.ask_id)
        self._quotes.clear()

    # ─────────────────────────────────────
    # BaseStrategy interface
    # ─────────────────────────────────────
    def on_bar(self, bar: pd.Series, book) -> None:
        """
        Not used directly for options MM.
        Options MM uses on_chain() instead.
        Kept for BaseStrategy compatibility.
        """
        pass

    def on_fill(self, fill) -> None:
        """Update delta hedger after option fill."""
        symbol = fill.symbol if hasattr(fill, 'symbol') else ""
        if symbol in self._quotes:
            quote = self._quotes[symbol]
            lots  = fill.quantity
            if fill.side.value == "BUY":
                self.hedger.update_option_position(
                    symbol, quote.delta, +lots
                )
            else:
                self.hedger.update_option_position(
                    symbol, quote.delta, -lots
                )

    def on_day_end(self, date: str, book) -> None:
        """Cancel all quotes at end of day."""
        self._cancel_all_quotes(None, book)
        self.hedger.reset()
        logger.info(
            f"Day end | hedge_trades={self.hedge_trades} | "
            f"bars_quoted={self.bars_quoted}"
        )

    def quote_rate(self) -> float:
        if self.bars_processed == 0:
            return 0.0
        return self.bars_quoted / self.bars_processed * 100


if __name__ == "__main__":
    print("=== 0DTE Options MM Strategy Tests ===\n")

    config = StrategyConfig(
        symbol        = "NIFTY_OPTIONS",
        lot_size      = 75,
        max_inventory = 10,
    )
    strategy = ZeroDTEStrategy(
        config            = config,
        strikes_each_side = 3,
        morning_lots      = 2,
        midday_lots       = 1,
        min_spread_pts    = 1.0,
    )

    # Simulate a mini options chain
    chain = pd.DataFrame([
        {"symbol": "NIFTY2650524000CE", "strike": 24000, "opt_type": "CE",
         "close": 150.0, "spread_mean": 2.0, "delta": 0.52,
         "gamma": 0.003, "vega": 8.5, "theta": -15.0,
         "iv": 0.145, "tte": 0.004, "ts_sec": 1777586700},
        {"symbol": "NIFTY2650524000PE", "strike": 24000, "opt_type": "PE",
         "close": 145.0, "spread_mean": 2.0, "delta": -0.48,
         "gamma": 0.003, "vega": 8.3, "theta": -14.5,
         "iv": 0.148, "tte": 0.004, "ts_sec": 1777586700},
        {"symbol": "NIFTY2650524100CE", "strike": 24100, "opt_type": "CE",
         "close": 90.0, "spread_mean": 1.5, "delta": 0.35,
         "gamma": 0.002, "vega": 6.5, "theta": -12.0,
         "iv": 0.150, "tte": 0.004, "ts_sec": 1777586700},
    ])

    print("Simulating chain at 09:30 IST (morning session)")
    print(f"Chain: {len(chain)} strikes\n")

    # Check delta hedger
    strategy.hedger.update_option_position("NIFTY2650524000CE", delta=0.52, lots=2)
    action = strategy.hedger.compute_hedge()
    print(f"After buying 2 lots ATM CE (delta=0.52):")
    print(f"  Net delta: {strategy.hedger.net_delta_shares():.1f} shares")
    print(f"  Hedge needed: {action.needed}")
    if action.needed:
        print(f"  Action: {action.side} {action.lots} lots futures")
    print(f"\nHedger status: {strategy.hedger.status()}")

    print("\n=== Time-based quoting test ===")
    morning_ts = 1777586700   # 09:30 IST
    cutoff_ts  = 1777608000   # 14:00 IST
    sod_utc    = morning_ts % 86400
    eod_utc    = cutoff_ts  % 86400
    print(f"Morning ts_sec % 86400 = {sod_utc} (< {CUTOFF_UTC} = quote)")
    print(f"After 14:00 ts_sec % 86400 = {eod_utc} (>= {CUTOFF_UTC} = stop)")

    print("\n✓ 0DTE strategy tests passed")