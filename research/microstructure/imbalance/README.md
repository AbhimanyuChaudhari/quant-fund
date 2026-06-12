# Order Flow Imbalance Research

## Objective
Build the best possible imbalance signal for NSE stock futures
that predicts short-term price direction at 1s-60s horizon.

## Key Papers
1. Cont, Kukanov, Stoikov (2014) — "The Price Impact of Order Book Events"
   Core finding: OFI (order flow imbalance) is linearly related to price changes
   
2. Easley, Lopez de Prado, O'Hara (2012) — "Flow Toxicity and Liquidity"
   VPIN metric — volume-synchronized probability of informed trading
   
3. Gould, Porter, Williams et al (2013) — "Limit Order Books"
   Comprehensive LOB dynamics review
   
4. Stoikov (2018) — "The micro-price: a high frequency estimator of future prices"
   Weighted mid price as better price estimate than simple mid

## Research Questions
1. Which imbalance measure best predicts next-bar price move on NSE?
   - Simple book imbalance (bid_q - ask_q) / (bid_q + ask_q)
   - OFI (order flow imbalance from Cont et al)
   - Weighted OFI across LOB levels
   - VPIN approximation
   
2. At what timescale does imbalance signal decay?
   - 1s horizon
   - 5s horizon  
   - 30s horizon
   - 60s horizon

3. Does signal strength vary by:
   - Time of day (open vs midday vs close)
   - Volatility regime (high vol vs low vol)
   - Symbol (liquid vs illiquid)
   
4. What's the optimal lookback for imbalance features?
   - Instantaneous (current bar only)
   - 10s rolling
   - 30s rolling
   - 60s rolling

## Available Features (from pipeline.py)
- imbalance_last      : book imbalance at last tick of bar
- imbalance_mean      : mean imbalance over bar
- imbalance_std       : std of imbalance over bar
- imbalance_ma_10s    : 10s rolling mean
- imbalance_ma_30s    : 30s rolling mean
- imbalance_ma_60s    : 60s rolling mean
- total_bid_qty       : total bid quantity
- total_ask_qty       : total ask quantity
- volume_delta        : signed volume (buy - sell proxy)
- tick_count          : number of ticks in bar

## Missing Features (need to add to pipeline)
- bid_p1-5, ask_p1-5  : LOB depth prices (5 levels)
- bid_q1-5, ask_q1-5  : LOB depth quantities (5 levels)
- ofi_weighted         : weighted OFI across levels
- trade_imbalance      : buyer vs seller initiated trades

## Notebooks
- nb_imb_01_eda.py      : Exploratory analysis — signal properties
- nb_imb_02_signals.py  : Signal construction + predictive power
- nb_imb_03_model.py    : Transformer training + validation

## Findings
(populated as research progresses)