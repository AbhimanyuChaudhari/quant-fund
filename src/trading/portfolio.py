"""
Portfolio — live position and PnL tracking.

Single source of truth for:
  - Current positions (lots per symbol)
  - Realized PnL
  - Unrealized PnL (mark-to-market)
  - Gross exposure
  - Trade history

State is saved to disk every minute so we can recover from crashes.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_PATH = Path("logs/portfolio_state.json")


@dataclass
class Position:
    symbol:       str
    lots:         int    = 0      # net lots (+ long, - short)
    avg_price:    float  = 0.0   # average entry price
    realized_pnl: float  = 0.0   # realized PnL for this symbol
    lot_size:     int    = 75    # contract lot size


@dataclass
class Trade:
    timestamp:  str
    symbol:     str
    side:       str
    lots:       int
    price:      float
    pnl:        float   = 0.0   # realized PnL if closing trade
    order_id:   str     = ""
    paper:      bool    = True


class Portfolio:
    """
    Tracks all live positions and PnL.

    Usage:
        portfolio = Portfolio(lot_sizes={'NIFTY26MAYFUT': 75})
        portfolio.on_fill('NIFTY26MAYFUT', 'BUY', 1, 23880.0)
        portfolio.on_fill('NIFTY26MAYFUT', 'SELL', 1, 23885.0)
        print(portfolio.realized_pnl)   # ₹375
    """

    def __init__(self, lot_sizes: dict[str, int] = None,
                 paper_trading: bool = True):
        self.lot_sizes     = lot_sizes or {}
        self.paper_trading = paper_trading
        self.positions:    dict[str, Position] = {}
        self.trades:       list[Trade]          = []
        self.realized_pnl  = 0.0
        self.trade_date    = date.today()
        self._last_prices: dict[str, float]     = {}

        # Load state if exists (crash recovery)
        self._load_state()

    # ─────────────────────────────────────
    # Fill handling
    # ─────────────────────────────────────
    def on_fill(self, symbol: str, side: str, lots: int,
                price: float, order_id: str = "") -> float:
        """
        Record a fill. Updates position and computes realized PnL.

        Returns:
            Realized PnL from this fill (0 if opening position)
        """
        lot_size  = self.lot_sizes.get(symbol, 75)
        pos       = self._get_or_create(symbol, lot_size)
        pnl       = 0.0

        if side == "BUY":
            if pos.lots < 0:
                # Closing a short position
                close_lots = min(lots, abs(pos.lots))
                pnl        = (pos.avg_price - price) * close_lots * lot_size
                pos.realized_pnl += pnl
                self.realized_pnl += pnl
                pos.lots += close_lots

                # If more lots than short, open long with remainder
                remaining = lots - close_lots
                if remaining > 0:
                    pos.avg_price = price
                    pos.lots      = remaining
            else:
                # Adding to long or opening long
                total_cost    = pos.avg_price * pos.lots + price * lots
                pos.lots     += lots
                pos.avg_price = total_cost / pos.lots if pos.lots else 0

        else:  # SELL
            if pos.lots > 0:
                # Closing a long position
                close_lots = min(lots, pos.lots)
                pnl        = (price - pos.avg_price) * close_lots * lot_size
                pos.realized_pnl += pnl
                self.realized_pnl += pnl
                pos.lots -= close_lots

                # If more lots than long, open short with remainder
                remaining = lots - close_lots
                if remaining > 0:
                    pos.avg_price = price
                    pos.lots      = -remaining
            else:
                # Adding to short or opening short
                total_cost    = pos.avg_price * abs(pos.lots) + price * lots
                pos.lots     -= lots
                pos.avg_price = total_cost / abs(pos.lots) if pos.lots else 0

        # Update last price
        self._last_prices[symbol] = price

        # Record trade
        trade = Trade(
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol    = symbol,
            side      = side,
            lots      = lots,
            price     = price,
            pnl       = round(pnl, 2),
            order_id  = order_id,
            paper     = self.paper_trading,
        )
        self.trades.append(trade)

        logger.info(
            f"{'[PAPER] ' if self.paper_trading else ''}"
            f"FILL {side} {lots}L {symbol} @ {price:.2f} | "
            f"PnL: Rs.{pnl:+,.0f} | "
            f"Position: {pos.lots}L @ {pos.avg_price:.2f}"
        )

        self._save_state()
        return pnl

    # ─────────────────────────────────────
    # Mark to market
    # ─────────────────────────────────────
    def update_price(self, symbol: str, price: float):
        """Update last known price for unrealized PnL calculation."""
        self._last_prices[symbol] = price

    def unrealized_pnl(self, symbol: Optional[str] = None) -> float:
        """Unrealized PnL at current market prices."""
        if symbol:
            pos = self.positions.get(symbol)
            if not pos or pos.lots == 0:
                return 0.0
            last = self._last_prices.get(symbol, pos.avg_price)
            return (last - pos.avg_price) * pos.lots * pos.lot_size

        total = 0.0
        for sym, pos in self.positions.items():
            if pos.lots != 0:
                last   = self._last_prices.get(sym, pos.avg_price)
                total += (last - pos.avg_price) * pos.lots * pos.lot_size
        return total

    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl()

    # ─────────────────────────────────────
    # Position queries
    # ─────────────────────────────────────
    def get_position(self, symbol: str) -> int:
        """Net lots for a symbol (+ long, - short, 0 flat)."""
        pos = self.positions.get(symbol)
        return pos.lots if pos else 0

    def gross_exposure(self) -> int:
        """Total absolute lots across all symbols."""
        return sum(abs(p.lots) for p in self.positions.values())

    def is_flat(self, symbol: str) -> bool:
        return self.get_position(symbol) == 0

    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.lots != 0}

    # ─────────────────────────────────────
    # EOD
    # ─────────────────────────────────────
    def on_day_end(self):
        """Save daily trade log and reset for next day."""
        self._save_trade_log()
        logger.info(
            f"Day end | Realized: Rs.{self.realized_pnl:+,.0f} | "
            f"Unrealized: Rs.{self.unrealized_pnl():+,.0f} | "
            f"Total: Rs.{self.total_pnl():+,.0f} | "
            f"Trades: {len(self.trades)}"
        )

    # ─────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────
    def _get_or_create(self, symbol: str, lot_size: int) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, lot_size=lot_size
            )
        return self.positions[symbol]

    def _save_state(self):
        """Save portfolio state to disk for crash recovery."""
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "date":          str(self.trade_date),
                "realized_pnl":  self.realized_pnl,
                "positions":     {
                    s: asdict(p) for s, p in self.positions.items()
                },
            }
            STATE_PATH.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")

    def _load_state(self):
        if not STATE_PATH.exists():
            return
        try:
            state = json.loads(STATE_PATH.read_text())
            saved_date = state.get("date", "")
            if saved_date == str(date.today()):
                self.realized_pnl = state.get("realized_pnl", 0.0)
                for sym, pos_data in state.get("positions", {}).items():
                    pos = Position(**pos_data)
                    if pos.lots != 0:   # ← only restore open positions
                        self.positions[sym] = pos
                logger.info(
                    f"Portfolio state restored | "
                    f"PnL: Rs.{self.realized_pnl:+,.0f} | "
                    f"Open positions: {len(self.positions)}"
                )
            else:
                # Different day — start fresh, delete stale state
                STATE_PATH.unlink(missing_ok=True)
                logger.info("New trading day — portfolio state reset")
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")

    def _save_trade_log(self):
        """Save today's trades to CSV."""
        if not self.trades:
            return
        log_path = Path(f"logs/trades/{self.trade_date}.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        import csv
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "symbol", "side", "lots",
                "price", "pnl", "order_id", "paper"
            ])
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(asdict(trade))

        logger.info(f"Trade log saved: {log_path}")

    def summary(self) -> dict:
        return {
            "realized_pnl":   round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl(), 2),
            "total_pnl":      round(self.total_pnl(), 2),
            "open_positions": {s: p.lots
                               for s, p in self.open_positions().items()},
            "gross_exposure": self.gross_exposure(),
            "total_trades":   len(self.trades),
            "paper_trading":  self.paper_trading,
        }


if __name__ == "__main__":
    print("=== Portfolio Tests ===\n")

    portfolio = Portfolio(
        lot_sizes={"NIFTY26MAYFUT": 75},
        paper_trading=True
    )

    # Test 1: buy and sell — realize PnL
    portfolio.on_fill("NIFTY26MAYFUT", "BUY",  1, 23880.0)
    portfolio.on_fill("NIFTY26MAYFUT", "SELL", 1, 23885.0)
    print(f"Round trip PnL: Rs.{portfolio.realized_pnl:,.0f} "
          f"(expected Rs.375)")

    # Test 2: unrealized PnL
    portfolio.on_fill("NIFTY26MAYFUT", "BUY", 1, 23880.0)
    portfolio.update_price("NIFTY26MAYFUT", 23890.0)
    print(f"Unrealized PnL: Rs.{portfolio.unrealized_pnl():,.0f} "
          f"(expected Rs.750)")

    print(f"\nSummary: {portfolio.summary()}")
    print("\n✓ Portfolio tests passed")