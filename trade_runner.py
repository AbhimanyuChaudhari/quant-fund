"""
Live Trading CLI Entry Point

Usage:
    # Paper trading on historical data (offline, no market needed)
    python trade_runner.py --symbol NIFTY26MAYFUT --paper

    # Paper trading on LIVE ticks (best for validation, run during market hours)
    python trade_runner.py --symbol NIFTY26MAYFUT --paper-live

    # Live trading (real money — careful)
    python trade_runner.py --symbol NIFTY26MAYFUT --live

    # Custom parameters
    python trade_runner.py --symbol NIFTY26MAYFUT --paper-live \
        --gamma 0.1 --loss-limit 10000 --max-lots 2
"""

import argparse
import pathlib
from src.trading.engine import LiveEngine
from src.trading.risk import RiskConfig
from src.backtest.models.v1_avellaneda_stoikov.strategy import StrategyConfig
from src.backtest.mm_strategy import AvellanedaStoikovStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trading engine — paper, paper-live, or live mode"
    )

    # Required
    parser.add_argument("--symbol", required=True,
                        help="e.g. NIFTY26MAYFUT or USDINR26508FUT")

    # Mode — must pick exactly one
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper",      action="store_true",
                      help="Historical GCS data, simulated fills (offline)")
    mode.add_argument("--paper-live", action="store_true",
                      help="Live WebSocket ticks, simulated fills (during market hours)")
    mode.add_argument("--live",       action="store_true",
                      help="Live WebSocket ticks, real Zerodha orders (real money)")

    # Strategy params
    parser.add_argument("--gamma",      type=float, default=0.1,
                        help="Risk aversion parameter (default 0.1)")
    parser.add_argument("--kappa",      type=float, default=1.5,
                        help="Order arrival rate (default 1.5)")
    parser.add_argument("--min-spread", type=float, default=0.5,
                        help="Min spread in pts (default 0.5)")
    parser.add_argument("--max-spread", type=float, default=10.0,
                        help="Max spread in pts (default 10.0)")

    # Risk params
    parser.add_argument("--loss-limit", type=float, default=10_000,
                        help="Daily loss limit in Rs. (default 10000)")
    parser.add_argument("--max-lots",   type=int,   default=2,
                        help="Max position in lots (default 2)")
    parser.add_argument("--lot-size",   type=int,   default=75,
                        help="Contract lot size (NIFTY=75, BANKNIFTY=35, USDINR=1000)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create log directories
    pathlib.Path("logs/orders").mkdir(parents=True, exist_ok=True)
    pathlib.Path("logs/trades").mkdir(parents=True, exist_ok=True)

    paper_live = args.paper_live
    live       = args.live
    paper      = args.paper

    # Confirm live trading
    if live:
        print(f"\nWARNING: LIVE TRADING MODE")
        print(f"Symbol:      {args.symbol}")
        print(f"Loss limit:  Rs.{args.loss_limit:,.0f}")
        print(f"Max lots:    {args.max_lots}")
        confirm = input("\nType 'YES' to confirm real orders will be placed: ")
        if confirm != "YES":
            print("Aborted.")
            return

    # Mode label
    if live:
        mode_label = "LIVE"
    elif paper_live:
        mode_label = "PAPER-LIVE (real ticks, simulated fills)"
    else:
        mode_label = "PAPER (historical data)"

    # Build strategy
    strat_config = StrategyConfig(
        symbol        = args.symbol,
        lot_size      = args.lot_size,
        max_inventory = args.max_lots,
    )
    strategy = AvellanedaStoikovStrategy(
        config     = strat_config,
        gamma      = args.gamma,
        kappa      = args.kappa,
        min_spread = args.min_spread,
        max_spread = args.max_spread,
    )

    # Build risk config
    risk_config = RiskConfig(
        daily_loss_limit  = args.loss_limit,
        max_position_lots = args.max_lots,
        max_order_lots    = args.max_lots,
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Mode:        {mode_label}")
    print(f"  Symbol:      {args.symbol}")
    print(f"  Strategy:    Avellaneda-Stoikov MM")
    print(f"  Gamma:       {args.gamma}")
    print(f"  Loss limit:  Rs.{args.loss_limit:,.0f}")
    print(f"  Max lots:    {args.max_lots}")
    print(f"{'='*50}\n")

    # Run
    engine = LiveEngine(
        strategy    = strategy,
        symbol      = args.symbol,
        paper_live  = paper_live,
        live        = live,
        lot_size    = args.lot_size,
        risk_config = risk_config,
    )
    engine.run()


if __name__ == "__main__":
    main()