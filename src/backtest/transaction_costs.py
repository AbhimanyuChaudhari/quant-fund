"""
Indian F&O Transaction Costs for NIFTY Futures

Cost breakdown per trade (one side):
    STT:              0.01%  of trade value (futures, sell side only)
    Exchange charges: 0.0019% of trade value (NSE)
    SEBI charges:     0.0001% of trade value
    GST:              18% on (brokerage + exchange + SEBI charges)
    Brokerage:        Zerodha flat ₹20 per order (futures)
    Stamp duty:       0.002% on buy side only

Sources:
    https://zerodha.com/charges/
    NSE circular on transaction charges
"""

from dataclasses import dataclass


@dataclass
class CostBreakdown:
    stt:              float = 0.0
    exchange:         float = 0.0
    sebi:             float = 0.0
    gst:              float = 0.0
    brokerage:        float = 0.0
    stamp:            float = 0.0

    @property
    def total(self) -> float:
        return self.stt + self.exchange + self.sebi + self.gst + \
               self.brokerage + self.stamp


class TransactionCosts:
    """
    Computes all-in transaction costs for NIFTY futures trades.

    Usage:
        tc = TransactionCosts(lot_size=75)
        cost = tc.compute(price=23880, lots=1, side='BUY')
        print(cost.total)   # total cost in rupees
    """

    # Rate constants
    STT_RATE      = 0.0001    # 0.01% — sell side only for futures
    EXCHANGE_RATE = 0.0000019 # 0.00019% NSE exchange charges
    SEBI_RATE     = 0.000001  # 0.0001% SEBI turnover fee
    GST_RATE      = 0.18      # 18% GST on brokerage + exchange + SEBI
    BROKERAGE     = 20.0      # Zerodha flat ₹20 per executed order
    STAMP_RATE    = 0.00002   # 0.002% — buy side only

    def __init__(self, lot_size: int = 75):
        self.lot_size = lot_size

    def compute(self, price: float, lots: int, side: str) -> CostBreakdown:
        """
        Compute full cost breakdown for one fill.

        Args:
            price:  fill price in points
            lots:   number of lots traded
            side:   'BUY' or 'SELL'

        Returns:
            CostBreakdown with each component and .total
        """
        trade_value = price * lots * self.lot_size

        cost = CostBreakdown()
        cost.brokerage = self.BROKERAGE
        cost.exchange  = trade_value * self.EXCHANGE_RATE
        cost.sebi      = trade_value * self.SEBI_RATE

        # STT only on sell side for futures
        if side == "SELL":
            cost.stt = trade_value * self.STT_RATE

        # Stamp duty only on buy side
        if side == "BUY":
            cost.stamp = trade_value * self.STAMP_RATE

        # GST on brokerage + exchange + SEBI
        cost.gst = (cost.brokerage + cost.exchange + cost.sebi) * self.GST_RATE

        return cost

    def round_trip(self, price: float, lots: int) -> float:
        """
        Total cost for a complete round trip (one buy + one sell).
        Useful for quickly checking if a strategy's spread capture
        exceeds transaction costs.
        """
        buy  = self.compute(price, lots, "BUY")
        sell = self.compute(price, lots, "SELL")
        return buy.total + sell.total


if __name__ == "__main__":
    tc = TransactionCosts(lot_size=75)

    print("=== NIFTY Futures Transaction Costs ===")
    print(f"Trade: 1 lot @ 23880 (value = ₹{23880*75:,.0f})\n")

    for side in ["BUY", "SELL"]:
        cost = tc.compute(price=23880, lots=1, side=side)
        print(f"{side}:")
        print(f"  Brokerage:  ₹{cost.brokerage:.2f}")
        print(f"  STT:        ₹{cost.stt:.2f}")
        print(f"  Exchange:   ₹{cost.exchange:.2f}")
        print(f"  SEBI:       ₹{cost.sebi:.2f}")
        print(f"  GST:        ₹{cost.gst:.2f}")
        print(f"  Stamp:      ₹{cost.stamp:.2f}")
        print(f"  TOTAL:      ₹{cost.total:.2f}")
        print()

    rt = tc.round_trip(23880, 1)
    spread_to_breakeven = rt / 75  # cost per share = breakeven spread in pts
    print(f"Round trip cost (1 lot):     ₹{rt:.2f}")
    print(f"Breakeven spread:            {spread_to_breakeven:.4f} pts")
    print(f"Our min spread:              0.5 pts")
    print(f"Profitable if spread >       {spread_to_breakeven:.4f} pts  ", end="")
    print("✓" if 0.5 > spread_to_breakeven else "✗ — tighten costs or widen spread")