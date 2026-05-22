# V2 — Ricci Hawkes-Alpha Market Making Strategy

## Reference
Ricci, J. (2014). *Applied Stochastic Control in High Frequency and Algorithmic Trading*.
PhD Thesis, University of Toronto. Chapter 3: Market Making in a Single Asset.

## Overview

V2 upgrades V1 (Avellaneda-Stoikov) with three major improvements from Ricci Chapter 3:

1. **Self-exciting order arrivals** (Hawkes process) — replaces Poisson assumption
2. **Short-term alpha** — predictable drift adjustment for adverse selection
3. **Stochastic fill rate** — κ±t adapts to LOB state instead of constant κ

## Files

| File | Purpose | Ricci Section |
|---|---|---|
| `strategy.py` | Main strategy — wires all components | 3.5 |
| `hawkes_process.py` | λ±t self-exciting dynamics + stability | 3.2.2, Lemma 3.2.2 |
| `short_term_alpha.py` | αt stochastic drift estimation | 3.4 |
| `fill_rate.py` | κ±t stochastic fill probability shape | 3.3 |
| `market_classifier.py` | P(influential \| MO) estimation | 3.2.2 |
| `calibrator.py` | MLE parameter fitting (after 30 days) | Appendix E.1 |

## Mathematical Model

### Price Dynamics (Section 3.2.1)
```
dSt = αt dt + σ dWt
```
αt = predictable short-term drift (zero in A-S, stochastic here)

### Hawkes Order Arrivals (Section 3.2.2, Assumption 3.2.1)
```
dλ⁻t = β(θ - λ⁻t)dt + η dM⁺t + ν dM⁻t
dλ⁺t = β(θ - λ⁺t)dt + η dM⁻t + ν dM⁺t
```
- λ±t = buy/sell MO intensity
- β = mean reversion rate
- θ = long-run intensity
- η = self-excitation (same side, η > ν)
- ν = cross-excitation (opposite side)
- ρ = P(influential | MO arrives)

**Stability condition (Lemma 3.2.2):** β > ρ(η + ν)
If violated → market is explosive → halt trading.

### Short-Term Alpha (Section 3.4, Assumption 3.4.1)
```
dαt = -ζ αt dt + σα dBt + ε⁺ dM⁺t - ε⁻ dM⁻t
```
- ζ = mean reversion rate
- ε± = jump size on influential buy/sell MO

### Stochastic Fill Rate (Section 3.3, Assumption 3.3.2)
```
h±(δ; κt) = e^(-κ±t · δ)    [exponential class]

dκ⁻t = βκ(θκ - κ⁻t)dt + ηκ dM⁺t + νκ dM⁻t
dκ⁺t = βκ(θκ - κ⁺t)dt + νκ dM⁺t + ηκ dM⁻t
```

Key result (Example 3.5.5): h±(δ0±; κ) = e⁻¹ ≈ 0.368 always
→ Optimal depth δ0± = 1/κ±t

### Optimal Quotes (Corollary 3.5.3)
```
δ±* = δ0± + B * {∓ Et[∫ᵀ αu du] + φ · bφ}
```

Where:
- δ0± = 1/κ±t (risk-neutral depth)
- B = 1/(2 + κ±t) (scaling coefficient)
- Et[∫αu du] = αt/ζ * (1 - e^(-ζT)) (alpha integral, Eq 3.19a)
- bφ = 2h(λ⁺t - λ⁻t)*[(T/β̂) - (1-e^(-β̂T))/β̂²] (lambda adjustment, Eq 3.19b)
- β̂ = β - ρ(η - ν) (effective decay rate)

### Quote Interpretation
```
Positive α (price rising):
    → widen ask (+alpha_integral): avoid sell-side adverse selection
    → tighten bid (-alpha_integral): capture upward move
    
Negative α (price falling):
    → widen bid (+alpha_integral): avoid buy-side adverse selection
    → tighten ask (-alpha_integral): capture downward move

Long inventory (q > 0):
    → ask closer to mid (inv_adj_ask smaller): easier to sell
    → bid further from mid (inv_adj_bid larger): avoid adding more longs

Buy-heavy flow (λ⁺ > λ⁻, lambda_adj > 0):
    → both spreads widen: compensate for increased adverse selection risk
```

## Parameters

### Phase 1 (Current — Proxy Implementation)
```python
params = V2Params(
    # Hawkes — reasonable defaults from Ricci's simulations
    beta=1.0, theta=2.0, eta=0.5, nu=0.2, rho=0.30,
    
    # Alpha — calibrated to price point scale
    zeta=0.5, epsilon_plus=0.002, epsilon_minus=0.002,
    
    # Fill rate — θκ matches V1 kappa
    beta_kappa=0.5, theta_kappa=1.5, eta_kappa=0.3, nu_kappa=0.1,
    
    # Risk
    phi=0.001, min_spread=0.10, max_spread=10.0,
)
```

### Phase 2 (After 30 days — MLE calibrated per symbol)
Run `calibrator.py --symbol CHOLAFIN26MAYFUT --start 2026-05-13 --end 2026-06-13`

## Comparison vs V1

| Feature | V1 (A-S) | V2 (Ricci) |
|---|---|---|
| Order arrivals | Poisson (constant κ) | Hawkes (self-exciting λ±t) |
| Price drift | Zero (μ=0) | Stochastic αt (OU + jumps) |
| Fill rate | Constant κ=1.5 | Stochastic κ±t |
| Quote symmetry | Symmetric around r | Asymmetric (α adjusts δ±) |
| Adverse selection | None | Explicit (alpha term) |
| Stability check | None | Lemma 3.2.2 |
| Parameters | 3 (γ, κ, min_spread) | 12 (calibrated per symbol) |

## Backtest Results (to be updated)

| Date | Symbols | V1 Net PnL | V2 Net PnL | Improvement |
|---|---|---|---|---|
| TBD | Core 7 | Rs.+17,89,911 | TBD | TBD |

## Usage

```python
from src.backtest.models.v2_ricci_hawkes_alpha.strategy import (
    RicciHawkesAlphaStrategy, V2Params
)
from src.backtest.strategy import StrategyConfig

config = StrategyConfig(
    symbol='CHOLAFIN26MAYFUT',
    lot_size=625,
    max_inventory=5,
)

params = V2Params(
    beta=1.0, theta=2.0, eta=0.5, nu=0.2, rho=0.30,
    phi=0.001, min_spread=0.10,
)

strategy = RicciHawkesAlphaStrategy(config, params)

# Use exactly like V1 in backtest engine
engine = BacktestEngine(backtest_config, strategy)
metrics = engine.run()
```

## Known Limitations (Phase 1)

1. **α proxy** — uses price_mom_60s as αt estimate, not calibrated OU process
2. **ρ not calibrated** — uses classifier heuristic, not MLE-fitted ρ
3. **No news events** — Zt± terms from Assumption 3.2.1 not implemented (μ̃=0)
4. **κ blended** — 70% model + 30% LOB observation (not pure model)

All four will be fixed in Phase 2 after calibrator.py is run on 30 days data.