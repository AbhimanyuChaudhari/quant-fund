"""
Indian Derivatives Transaction Costs
Supports: Equity Futures, Currency Futures, Equity Options, Currency Options

Rates effective April 1, 2026 (post Budget 2026)
Sources: Zerodha charges page, NSE circulars, SEBI regulations
"""

from dataclasses import dataclass


@dataclass
class CostBreakdown:
    stt:        float = 0.0
    exchange:   float = 0.0
    sebi:       float = 0.0
    gst:        float = 0.0
    brokerage:  float = 0.0
    stamp:      float = 0.0

    @property
    def total(self) -> float:
        return self.stt + self.exchange + self.sebi + \
               self.gst + self.brokerage + self.stamp


# ─────────────────────────────────────
# Rate tables per instrument type
# ─────────────────────────────────────
RATES = {
    "equity_futures": {
        "stt":        0.0005,    # 0.05% sell side (Budget 2026)
        "exchange":   0.0000173, # 0.00173% NSE
        "sebi":       0.000001,  # 0.0001%
        "stamp":      0.00002,   # 0.002% buy side
        "brokerage":  20.0,      # Zerodha flat
    },
    "currency_futures": {
        "stt":        0.0,       # ZERO — currency exempt from STT
        "exchange":   0.000009,  # 0.0009% NSE CDS
        "sebi":       0.000001,  # 0.0001%
        "stamp":      0.00001,   # 0.001% buy side
        "brokerage":  20.0,      # Zerodha flat
    },
    "equity_options": {
        "stt":        0.0005,    # 0.05% on premium, sell side
        "exchange":   0.0000353, # 0.00353% NSE
        "sebi":       0.000001,  # 0.0001%
        "stamp":      0.00003,   # 0.003% buy side
        "brokerage":  20.0,      # Zerodha flat
    },
    "currency_options": {
        "stt":        0.0,       # ZERO
        "exchange":   0.000009,  # 0.0009% NSE CDS
        "sebi":       0.000001,  # 0.0001%
        "stamp":      0.00001,   # 0.001% buy side
        "brokerage":  20.0,      # Zerodha flat
    },
}

GST_RATE = 0.18   # 18% on brokerage + exchange + sebi


class TransactionCosts:
    """
    All-in transaction costs for Indian derivatives via Zerodha.

    Usage:
        # Equity futures (NIFTY, BANKNIFTY, stocks)
        tc = TransactionCosts(lot_size=75, instrument_type='equity_futures')

        # Currency futures (USDINR) — zero STT!
        tc = TransactionCosts(lot_size=1000, instrument_type='currency_futures')

        cost = tc.compute(price=23880, lots=1, side='BUY')
        print(f'Total: Rs.{cost.total:.2f}')
        print(f'Breakeven: {tc.breakeven_spread(23880, 1):.4f} pts')
    """

    def __init__(self, lot_size:        int = 75,
                 instrument_type: str = "equity_futures"):
        if instrument_type not in RATES:
            raise ValueError(f"Unknown instrument_type: {instrument_type}. "
                             f"Choose from {list(RATES.keys())}")
        self.lot_size        = lot_size
        self.instrument_type = instrument_type
        self.rates           = RATES[instrument_type]

    def compute(self, price: float, lots: int, side: str) -> CostBreakdown:
        """
        Full cost breakdown for one fill.

        Args:
            price:  fill price
            lots:   number of lots
            side:   'BUY' or 'SELL'

        Returns:
            CostBreakdown with each component and .total in Rs.
        """
        trade_value = price * lots * self.lot_size
        cost        = CostBreakdown()

        cost.brokerage = self.rates["brokerage"]
        cost.exchange  = trade_value * self.rates["exchange"]
        cost.sebi      = trade_value * self.rates["sebi"]

        if side == "SELL":
            cost.stt = trade_value * self.rates["stt"]

        if side == "BUY":
            cost.stamp = trade_value * self.rates["stamp"]

        cost.gst = (cost.brokerage + cost.exchange + cost.sebi) * GST_RATE
        return cost

    def round_trip(self, price: float, lots: int) -> float:
        """Total cost for one complete round trip (buy + sell)."""
        return (self.compute(price, lots, "BUY").total +
                self.compute(price, lots, "SELL").total)

    def breakeven_spread(self, price: float, lots: int) -> float:
        """Minimum spread to capture per round trip to break even."""
        return self.round_trip(price, lots) / (lots * self.lot_size)


if __name__ == "__main__":
    print("=" * 55)
    print("Transaction Cost Comparison — All Instruments")
    print("=" * 55)

    instruments = [
        ("NIFTY Futures",    75,   "equity_futures",   23880),
        ("BANKNIFTY Futures",35,   "equity_futures",   51000),
        ("USDINR Futures",   1000, "currency_futures",    91),
        ("NIFTY Options",    75,   "equity_options",      50),  # ATM premium
        ("USDINR Options",   1000, "currency_options",   0.5),  # ATM premium
    ]

    print(f"\n{'Instrument':<22} {'Lot':>5} {'RT Cost':>10} "
          f"{'Breakeven':>12} {'STT':>10}")
    print("-" * 65)

    for name, lot_size, itype, price in instruments:
        tc = TransactionCosts(lot_size=lot_size, instrument_type=itype)
        rt = tc.round_trip(price, 1)
        be = tc.breakeven_spread(price, 1)
        stt_sell = tc.compute(price, 1, "SELL").stt
        print(f"{name:<22} {lot_size:>5} Rs.{rt:>8,.2f} "
              f"{be:>11.4f} pts  Rs.{stt_sell:>7,.2f}")

    print("\n=== USDINR Futures Detail ===")
    tc = TransactionCosts(lot_size=1000, instrument_type="currency_futures")
    for side in ["BUY", "SELL"]:
        cost = tc.compute(price=91.0, lots=1, side=side)
        print(f"\n{side}:")
        print(f"  Brokerage: Rs.{cost.brokerage:.2f}")
        print(f"  STT:       Rs.{cost.stt:.2f}  <- ZERO for currency!")
        print(f"  Exchange:  Rs.{cost.exchange:.4f}")
        print(f"  SEBI:      Rs.{cost.sebi:.4f}")
        print(f"  GST:       Rs.{cost.gst:.4f}")
        print(f"  Stamp:     Rs.{cost.stamp:.4f}")
        print(f"  TOTAL:     Rs.{cost.total:.2f}")

    rt = tc.round_trip(91.0, 1)
    be = tc.breakeven_spread(91.0, 1)
    print(f"\nRound trip:       Rs.{rt:.2f}")
    print(f"Breakeven spread: {be:.6f} paise")
    print(f"Typical spread:   0.25 paise")
    print(f"Viable:           {'YES' if be < 0.25 else 'NO'}")
