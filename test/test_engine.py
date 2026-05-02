import traceback
import sys

print("Step 1: importing risk...", flush=True)
from src.trading.risk import RiskManager, RiskConfig
print("Step 2: importing portfolio...", flush=True)
from src.trading.portfolio import Portfolio
print("Step 3: importing order_manager...", flush=True)
from src.trading.order_manager import OrderManager
print("Step 4: importing feature_engine...", flush=True)
from src.trading.feature_engine import FeatureEngine
print("Step 5: importing strategy...", flush=True)
from src.backtest.strategy import StrategyConfig
from src.backtest.mm_strategy import AvellanedaStoikovStrategy
print("Step 6: importing engine...", flush=True)
from src.trading.engine import LiveEngine
print("Step 7: all imports done", flush=True)

try:
    strat_config = StrategyConfig(
        symbol='NIFTY26MAYFUT', lot_size=75, max_inventory=2
    )
    strategy = AvellanedaStoikovStrategy(strat_config)
    print("Step 8: strategy created", flush=True)

    engine = LiveEngine(
        strategy=strategy,
        symbol='NIFTY26MAYFUT',
        paper_trading=True
    )
    print("Step 9: engine created", flush=True)
    engine.run()

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()