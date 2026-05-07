"""
Dynamic Options Subscription Manager

Resubscribes to new ATM strikes every 30 minutes
when underlying price moves significantly.

Problem solved:
  Collector subscribes at startup based on current price
  If market gaps overnight, ATM shifts
  We end up collecting far OTM data instead of ATM

Solution:
  Monitor underlying spot prices continuously
  When ATM shifts by > 1 strike interval:
    Unsubscribe old strikes far from new ATM
    Subscribe new strikes near new ATM

Integration:
  Called from collector.py every 30 minutes
  Uses existing KiteTicker websocket connection
"""

import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Strike intervals per underlying
STRIKE_INTERVALS = {
    "NIFTY":     50,
    "BANKNIFTY": 100,
    "FINNIFTY":  50,
}

# Spot tokens for real-time price
SPOT_TOKENS = {
    "NIFTY":     256265,
    "BANKNIFTY": 260105,
}


class DynamicSubscriptionManager:
    """
    Manages dynamic options subscriptions based on live spot price.

    Usage in collector.py:
        mgr = DynamicSubscriptionManager(kite, kws, instruments_df)
        
        # Call every 30 minutes during market hours
        mgr.rebalance_if_needed(current_spot_prices)
    """

    def __init__(self, kite, kws, instruments_df,
                 strikes_each_side: int = 10,
                 rebalance_interval: int = 1800,  # 30 minutes
                 min_shift_strikes:  int = 2):    # rebalance if ATM shifts 2+ strikes
        """
        Args:
            kite:               KiteConnect client
            kws:                KiteTicker websocket
            instruments_df:     pd.DataFrame of all NFO instruments
            strikes_each_side:  ATM ± N strikes to subscribe
            rebalance_interval: seconds between rebalance checks
            min_shift_strikes:  minimum strike shift to trigger rebalance
        """
        self.kite               = kite
        self.kws                = kws
        self.instruments        = instruments_df
        self.strikes_each_side  = strikes_each_side
        self.rebalance_interval = rebalance_interval
        self.min_shift_strikes  = min_shift_strikes

        # Track current subscriptions per underlying
        # {underlying: {'atm': float, 'tokens': set, 'expiry': str}}
        self._current: dict = {}

        # Last rebalance time
        self._last_rebalance: float = 0

    def should_rebalance(self) -> bool:
        """Check if enough time has passed since last rebalance."""
        return time.time() - self._last_rebalance > self.rebalance_interval

    def rebalance_if_needed(self, spot_prices: dict) -> dict:
        """
        Check if ATM has shifted and resubscribe if needed.

        Args:
            spot_prices: {underlying: current_price}
                e.g. {'NIFTY': 23850.0, 'BANKNIFTY': 54900.0}

        Returns:
            Summary of changes made
        """
        if not self.should_rebalance():
            return {}

        changes = {}
        for underlying, price in spot_prices.items():
            change = self._rebalance_underlying(underlying, price)
            if change:
                changes[underlying] = change

        self._last_rebalance = time.time()
        return changes

    def _rebalance_underlying(self, underlying: str,
                               current_price: float) -> Optional[dict]:
        """
        Rebalance subscriptions for one underlying.
        Returns change summary or None if no change needed.
        """
        import pandas as pd
        from datetime import date

        interval = STRIKE_INTERVALS.get(underlying, 50)
        new_atm  = round(current_price / interval) * interval

        # Check if ATM has shifted significantly
        current = self._current.get(underlying)
        if current:
            old_atm      = current["atm"]
            strike_shift = abs(new_atm - old_atm) / interval
            if strike_shift < self.min_shift_strikes:
                return None  # ATM hasn't moved enough

        logger.info(
            f"Rebalancing {underlying} options: "
            f"price={current_price:.2f} ATM={new_atm}"
        )

        # Get nearest expiry options
        today   = pd.Timestamp(date.today())
        opts    = self.instruments[
            (self.instruments["name"] == underlying) &
            (self.instruments["instrument_type"].isin(["CE", "PE"])) &
            (self.instruments["expiry"] >= today)
        ]

        if opts.empty:
            return None

        expiries    = sorted(opts["expiry"].unique())
        use_expiry  = expiries[0]  # nearest expiry
        exp_opts    = opts[opts["expiry"] == use_expiry]

        # Build new strike list ATM ± N
        new_strikes = set()
        new_tokens  = set()

        for i in range(-self.strikes_each_side,
                       self.strikes_each_side + 1):
            strike = new_atm + i * interval
            for opt_type in ["CE", "PE"]:
                match = exp_opts[
                    (abs(exp_opts["strike"] - strike) < 0.01) &
                    (exp_opts["instrument_type"] == opt_type)
                ]
                if not match.empty:
                    token = int(match.iloc[0]["instrument_token"])
                    new_tokens.add(token)
                    new_strikes.add(strike)

        if not new_tokens:
            return None

        # Find tokens to remove (old but not in new set)
        old_tokens  = current["tokens"] if current else set()
        to_remove   = old_tokens - new_tokens
        to_add      = new_tokens - old_tokens

        # Unsubscribe old strikes
        if to_remove:
            try:
                self.kws.unsubscribe(list(to_remove))
                logger.info(
                    f"{underlying}: unsubscribed "
                    f"{len(to_remove)} old strikes"
                )
            except Exception as e:
                logger.error(f"Unsubscribe error: {e}")

        # Subscribe new strikes
        if to_add:
            try:
                self.kws.subscribe(list(to_add))
                self.kws.set_mode(
                    self.kws.MODE_FULL, list(to_add)
                )
                logger.info(
                    f"{underlying}: subscribed "
                    f"{len(to_add)} new strikes "
                    f"(ATM={new_atm})"
                )
            except Exception as e:
                logger.error(f"Subscribe error: {e}")

        # Update tracking
        self._current[underlying] = {
            "atm":     new_atm,
            "tokens":  new_tokens,
            "expiry":  str(use_expiry.date()),
            "price":   current_price,
            "updated": datetime.now().strftime("%H:%M:%S"),
        }

        return {
            "old_atm":   current["atm"] if current else None,
            "new_atm":   new_atm,
            "added":     len(to_add),
            "removed":   len(to_remove),
            "total":     len(new_tokens),
        }

    def initialize(self, spot_prices: dict):
        """
        Initial subscription at startup.
        Same as rebalance but forces update regardless of time.
        """
        self._last_rebalance = 0  # force rebalance
        changes = self.rebalance_if_needed(spot_prices)
        logger.info(f"Initial subscription: {changes}")
        return changes

    def status(self) -> dict:
        return {
            underlying: {
                "atm":     info["atm"],
                "price":   info["price"],
                "tokens":  len(info["tokens"]),
                "expiry":  info["expiry"],
                "updated": info["updated"],
            }
            for underlying, info in self._current.items()
        }