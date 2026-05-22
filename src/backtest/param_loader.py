import json
from pathlib import Path

def load_optimal_params(model: str = 'v1') -> dict:
    path = Path(f'research/findings/{model}_optimal_params.json')
    if path.exists():
        return json.load(open(path))
    return {}

def get_symbol_params(symbol: str, model: str = 'v1') -> dict:
    params = load_optimal_params(model)
    if symbol in params:
        p = params[symbol]
        return {
            'gamma':      p.get('gamma', 0.001),
            'kappa':      p.get('kappa', 1.5),
            'min_spread': p.get('min_spread', 0.10),
            'open_mult':  p.get('open_mult', 2.0),
        }
    # Default params if symbol not found
    return {'gamma': 0.001, 'kappa': 1.5, 'min_spread': 0.10, 'open_mult': 2.0}