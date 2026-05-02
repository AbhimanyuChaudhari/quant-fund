"""
Backtest CLI entry point.

Usage:
    # Simple simulation (fills every bar that crosses price)
    python backtest.py --symbol NIFTY26MAYFUT --start 2026-04-30 --end 2026-04-30

    # Realistic simulation (queue + volume filter)
    python backtest.py --symbol NIFTY26MAYFUT --start 2026-04-30 --end 2026-04-30 --realistic

    # Custom parameters
    python backtest.py --symbol NIFTY26MAYFUT --start 2026-04-30 --end 2026-04-30 \
        --realistic --gamma 0.1 --max-inventory 5 --queue-aggression 0.3
"""

import argparse
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.strategy import StrategyConfig
from src.backtest.mm_strategy import AvellanedaStoikovStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest market making strategies on NIFTY/BANKNIFTY/USDINR"
    )

    # Required
    parser.add_argument("--symbol",   required=True, help="e.g. NIFTY26MAYFUT")
    parser.add_argument("--start",    required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      required=True, help="End date YYYY-MM-DD")

    # Fill simulation
    parser.add_argument("--realistic", action="store_true",
                        help="Use realistic fill simulation (L1+L2+L3)")
    parser.add_argument("--queue-aggression", type=float, default=0.3,
                        help="Queue aggression 0-1 (default 0.3, higher=more fills)")

    # Strategy params
    parser.add_argument("--gamma",         type=float, default=0.1)
    parser.add_argument("--kappa",         type=float, default=1.5)
    parser.add_argument("--min-spread",    type=float, default=0.5)
    parser.add_argument("--max-spread",    type=float, default=10.0)

    # Risk params
    parser.add_argument("--max-inventory", type=int,   default=5)
    parser.add_argument("--lot-size",      type=int,   default=75)

    return parser.parse_args()


def main():
    args = parse_args()

    strat_config = StrategyConfig(
        symbol        = args.symbol,
        lot_size      = args.lot_size,
        max_inventory = args.max_inventory,
    )
    strategy = AvellanedaStoikovStrategy(
        config     = strat_config,
        gamma      = args.gamma,
        kappa      = args.kappa,
        min_spread = args.min_spread,
        max_spread = args.max_spread,
    )

    bt_config = BacktestConfig(
        symbol        = args.symbol,
        start         = args.start,
        end           = args.end,
        lot_size      = args.lot_size,
        max_inventory = args.max_inventory,
    )

    # Choose order book type
    if args.realistic:
        from src.trading.fill_simulator import RealisticOrderBook
        from src.backtest.order_book import SimulatedOrderBook
        print(f"\nUsing REALISTIC fill simulation "
              f"(queue_aggression={args.queue_aggression})")
        order_book = RealisticOrderBook(
            max_inventory    = args.max_inventory,
            queue_aggression = args.queue_aggression,
        )
    else:
        from src.backtest.order_book import SimulatedOrderBook
        print("\nUsing SIMPLE fill simulation (fills on price crossing)")
        order_book = None  # engine uses default

    engine  = BacktestEngine(bt_config, strategy,
                             order_book=order_book if args.realistic else None)
    metrics = engine.run()

    if args.realistic and hasattr(engine, 'book'):
        print(f"\nFill rate: {engine.book.fill_rate():.1%} "
              f"({engine.book.fill_successes:,} / "
              f"{engine.book.fill_attempts:,} attempts)")

    exit(0 if metrics.net_pnl > 0 else 1)


if __name__ == "__main__":
    main()