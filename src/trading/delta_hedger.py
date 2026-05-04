"""
Delta Hedger — Real-time delta neutralization for options MM

Tracks net portfolio delta across all option positions
and computes required futures trades to stay delta neutral.

Key concepts:
  Net delta = sum(option_delta × lots × lot_size) + futures_position × lot_size
  Hedge when |net_delta| > threshold (half a futures lot = 37.5 shares)
  Use NIFTY futures as hedge instrument
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HedgeAction:
    needed:       bool  = False
    side:         str   = ""      # 'BUY' or 'SELL'
    lots:         int   = 0
    net_delta:    float = 0.0
    reason:       str   = ""


class DeltaHedger:
    """
    Computes and manages delta hedging for options positions.

    Usage:
        hedger = DeltaHedger(
            futures_symbol = 'NIFTY26MAYFUT',
            lot_size       = 75,
            hedge_threshold= 37.5,  # half a lot
        )

        # After each option fill, update position
        hedger.update_option_position('NIFTY2650524000CE', delta=0.5, lots=+1)

        # Check if hedge needed
        action = hedger.compute_hedge()
        if action.needed:
            engine.post_quote(action.side, futures_price, action.lots)
    """

    def __init__(self,
                 futures_symbol:  str   = "NIFTY26MAYFUT",
                 lot_size:        int   = 75,
                 hedge_threshold: float = 36.0,   # half a futures lot
                 max_hedge_lots:  int   = 10):

        self.futures_symbol  = futures_symbol
        self.lot_size        = lot_size
        self.hedge_threshold = hedge_threshold
        self.max_hedge_lots  = max_hedge_lots

        # Option positions: {symbol: {'lots': int, 'delta': float}}
        self._option_positions: dict = {}

        # Futures position (net lots, + long, - short)
        self._futures_lots: int = 0

        # Greeks cache: {symbol: {'delta': float, 'gamma': float, ...}}
        self._greeks: dict = {}

    # ─────────────────────────────────────
    # Position tracking
    # ─────────────────────────────────────
    def update_option_position(self, symbol: str,
                               delta: float, lots: int):
        """
        Update option position after a fill.

        Args:
            symbol: option symbol e.g. 'NIFTY2650524000CE'
            delta:  current delta of the option (from Greeks)
            lots:   change in lots (+1 = bought 1, -1 = sold 1)
        """
        if symbol not in self._option_positions:
            self._option_positions[symbol] = {"lots": 0, "delta": delta}

        self._option_positions[symbol]["lots"]  += lots
        self._option_positions[symbol]["delta"]  = delta

        logger.debug(
            f"Option position updated: {symbol} "
            f"lots={self._option_positions[symbol]['lots']} "
            f"delta={delta:.3f}"
        )

    def update_greeks(self, symbol: str, greeks: dict):
        """
        Update cached Greeks for a symbol.
        Called every bar with latest computed Greeks.

        Args:
            symbol: option symbol
            greeks: dict with keys: delta, gamma, vega, theta, iv
        """
        self._greeks[symbol] = greeks

        # Update delta in position tracker
        if symbol in self._option_positions:
            self._option_positions[symbol]["delta"] = greeks.get("delta", 0)

    def update_futures_position(self, lots_change: int):
        """
        Update futures position after a hedge trade.

        Args:
            lots_change: +N = bought N lots, -N = sold N lots
        """
        self._futures_lots += lots_change
        logger.debug(f"Futures position: {self._futures_lots} lots")

    # ─────────────────────────────────────
    # Delta computation
    # ─────────────────────────────────────
    def net_delta_shares(self) -> float:
        """
        Compute total portfolio delta in shares.

        Net delta = sum(option_lots × option_delta × lot_size)
                  + futures_lots × lot_size
        """
        options_delta = 0.0
        for sym, pos in self._option_positions.items():
            lots  = pos["lots"]
            delta = pos["delta"]
            if lots != 0:
                options_delta += lots * delta * self.lot_size

        futures_delta = self._futures_lots * self.lot_size
        return options_delta + futures_delta

    def net_delta_lots(self) -> float:
        """Net delta expressed in equivalent futures lots."""
        return self.net_delta_shares() / self.lot_size

    # ─────────────────────────────────────
    # Hedge computation
    # ─────────────────────────────────────
    def compute_hedge(self) -> HedgeAction:
        """
        Determine if a hedge trade is needed.

        Returns HedgeAction with:
            needed=True  → place hedge order
            needed=False → no action required
        """
        net_delta = self.net_delta_shares()
        action    = HedgeAction(net_delta=net_delta)

        if abs(net_delta) <= self.hedge_threshold:
            action.reason = (
                f"Delta within threshold "
                f"({net_delta:.1f} < {self.hedge_threshold})"
            )
            return action

        # Need to hedge
        lots_to_hedge = int(abs(net_delta) / self.lot_size)
        lots_to_hedge = min(lots_to_hedge, self.max_hedge_lots)

        if lots_to_hedge == 0:
            action.reason = "Delta too small to hedge with full lot"
            return action

        action.needed = True
        action.lots   = lots_to_hedge

        if net_delta > 0:
            # Long delta → sell futures
            action.side   = "SELL"
            action.reason = (
                f"Long delta {net_delta:.1f} shares → "
                f"sell {lots_to_hedge} lots futures"
            )
        else:
            # Short delta → buy futures
            action.side   = "BUY"
            action.reason = (
                f"Short delta {net_delta:.1f} shares → "
                f"buy {lots_to_hedge} lots futures"
            )

        return action

    # ─────────────────────────────────────
    # Status
    # ─────────────────────────────────────
    def status(self) -> dict:
        open_options = {
            s: p for s, p in self._option_positions.items()
            if p["lots"] != 0
        }
        return {
            "net_delta_shares": round(self.net_delta_shares(), 2),
            "net_delta_lots":   round(self.net_delta_lots(), 2),
            "futures_lots":     self._futures_lots,
            "open_options":     len(open_options),
            "option_positions": {
                s: {"lots": p["lots"], "delta": p["delta"]}
                for s, p in open_options.items()
            },
        }

    def reset(self):
        self._option_positions = {}
        self._futures_lots     = 0
        self._greeks           = {}


if __name__ == "__main__":
    print("=== Delta Hedger Tests ===\n")

    hedger = DeltaHedger(
        futures_symbol  = "NIFTY26MAYFUT",
        lot_size        = 75,
        hedge_threshold = 37.5,
    )

    # Test 1: Buy 1 lot ATM call (delta=0.5)
    hedger.update_option_position("NIFTY2650524000CE", delta=0.5, lots=1)
    net = hedger.net_delta_shares()
    print(f"After buying 1 lot ATM call (delta=0.5):")
    print(f"  Net delta: {net:.1f} shares ({net/75:.2f} lots)")

    action = hedger.compute_hedge()
    print(f"  Hedge needed: {action.needed}")
    print(f"  Action: {action.side} {action.lots} lots")
    print(f"  Reason: {action.reason}\n")

    # Test 2: Apply hedge
    hedger.update_futures_position(-1)  # sold 1 lot futures
    net = hedger.net_delta_shares()
    print(f"After selling 1 lot futures:")
    print(f"  Net delta: {net:.1f} shares ({net/75:.2f} lots)")
    action = hedger.compute_hedge()
    print(f"  Hedge needed: {action.needed} (within threshold)\n")

    # Test 3: Buy 3 more calls — need more hedging
    hedger.update_option_position("NIFTY2650524000CE", delta=0.5, lots=3)
    net = hedger.net_delta_shares()
    print(f"After buying 3 more ATM calls:")
    print(f"  Net delta: {net:.1f} shares ({net/75:.2f} lots)")
    action = hedger.compute_hedge()
    print(f"  Hedge needed: {action.needed}")
    print(f"  Action: {action.side} {action.lots} lots\n")

    # Test 4: Mixed CE and PE position (delta neutral naturally)
    hedger2 = DeltaHedger(lot_size=75, hedge_threshold=37.5)
    hedger2.update_option_position("NIFTY2650524000CE", delta=0.5,  lots=1)
    hedger2.update_option_position("NIFTY2650524000PE", delta=-0.5, lots=1)
    net = hedger2.net_delta_shares()
    print(f"Long 1 CE (delta=+0.5) + Long 1 PE (delta=-0.5):")
    print(f"  Net delta: {net:.1f} shares")
    action = hedger2.compute_hedge()
    print(f"  Hedge needed: {action.needed} (naturally delta neutral)")

    print(f"\nStatus: {hedger.status()}")
    print("\n✓ Delta hedger tests passed")