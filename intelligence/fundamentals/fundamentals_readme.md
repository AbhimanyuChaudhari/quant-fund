# Fundamentals Engine — Developer Guide

## Overview

This module builds a fundamental analysis layer that feeds into the quantitative
market making system. The goal is simple — answer one question for each symbol
every morning:

> **"Should we trade more aggressively, normally, cautiously, or not at all today?"**

The output is a `FundamentalSignal` per symbol that adjusts:
- `spread_multiplier` — widen or tighten quotes
- `alpha_prior` — directional bias for the Hawkes model
- `mm_recommendation` — aggressive / normal / cautious / avoid

---

## How It Fits Into the Trading System

```
                    ┌─────────────────────┐
                    │  Fundamental Engine  │  ← your work
                    │  (runs at 8:30am)   │
                    └──────────┬──────────┘
                               │ FundamentalSignal per symbol
                               ▼
                    ┌─────────────────────┐
                    │   MM Strategy        │
                    │   V1 (A-S)          │
                    │   V2 (Ricci Hawkes) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Order Execution   │
                    │   (Zerodha/Finvasia)│
                    └─────────────────────┘
```

Three integration points:

1. **Symbol selection** (weekly) — which 85 symbols to trade
2. **Parameter adjustment** (daily at 8:30am) — spread multiplier per symbol
3. **Real-time signals** (intraday) — earnings beats, rating changes, block deals

---

## File Structure

```
src/
  fundamentals/
    __init__.py
    engine.py              ← FundamentalEngine (main class)
    signal.py              ← FundamentalSignal dataclass
    scrapers/
      __init__.py
      nse_scraper.py        ← NSE corporate actions, filings
      screener_scraper.py   ← financial ratios from screener.in
      earnings_scraper.py   ← earnings calendar, estimates
      news_scraper.py       ← corporate announcements
    models/
      __init__.py
      quality_score.py      ← fundamental quality scoring
      earnings_predictor.py ← earnings surprise prediction
      sector_rotation.py    ← sector strength tracker
    dashboard/
      app.py                ← Streamlit dashboard
    data/
      earnings_calendar.csv ← manually maintained calendar
      sector_map.json        ← symbol → sector mapping
      quality_scores.json    ← latest quality scores

research/
  notebooks/
    08_fundamental_quality_vs_mm_performance.ipynb
    09_earnings_filter_backtest.ipynb
    10_sector_rotation_signals.ipynb
```

---

## Phase 1 — Setup and Data Sources (Week 1)

### Task 1.1 — Install Dependencies

```bash
pip install requests beautifulsoup4 pandas numpy
pip install streamlit plotly sqlalchemy
pip install yfinance nsepy  # fallback data sources
```

### Task 1.2 — Data Sources

Start with free sources. Build the pipeline first, upgrade to paid later.

**Free sources (start here):**

| Source | Data | URL |
|---|---|---|
| NSE website | Bhavcopy, corporate actions, filings | https://nseindia.com |
| BSE website | Quarterly results, announcements | https://bseindia.com |
| Screener.in | Financial ratios, 10-year history | https://screener.in |
| Tijori Finance | India-specific fundamental data | https://tijorifinance.com |
| Tickertape | Earnings calendar, analyst estimates | https://tickertape.in |
| Moneycontrol | News, analyst ratings | https://moneycontrol.com |

**Paid sources (when budget allows):**

| Source | Cost | Best For |
|---|---|---|
| Ace Equity | Rs.20,000/month | India financials, deep history |
| Capitaline | Rs.15,000/month | Good India coverage |
| Refinitiv Eikon | Rs.50,000+/month | Institutional grade |
| Bloomberg | Rs.1,50,000+/month | Industry standard |

### Task 1.3 — Data to Collect Per Symbol

**Quarterly (every 3 months):**

```python
# Financial results
revenue          # total revenue
net_profit       # profit after tax
ebitda           # earnings before interest, tax, depreciation
debt             # total debt
cash             # cash and equivalents

# Per share
eps_actual       # actual earnings per share
eps_estimate     # analyst consensus estimate
eps_surprise_pct # (actual - estimate) / estimate × 100

# Growth
revenue_growth_yoy  # year-on-year revenue growth %
profit_growth_yoy   # year-on-year profit growth %

# Quality ratios
roe              # return on equity
roce             # return on capital employed
debt_to_equity   # leverage ratio
current_ratio    # liquidity ratio
```

**Annual:**

```python
capex            # capital expenditure
free_cash_flow   # fcf = operating cash - capex
promoter_holding # % shares held by promoters
fii_holding      # % held by foreign institutions
dii_holding      # % held by domestic institutions
promoter_pledge  # % of promoter shares pledged (RED FLAG if > 30%)
```

**Real-time / Event-driven:**

```python
next_earnings_date    # next quarterly result announcement
dividend_ex_date      # ex-dividend date (stock drops by dividend amount)
bonus_record_date     # bonus issue record date
rights_issue_date     # rights issue date
board_meeting_date    # upcoming board meetings
```

### Task 1.4 — NSE Scraper (Priority)

NSE provides free data. Build this first:

```python
# src/fundamentals/scrapers/nse_scraper.py

import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://www.nseindia.com/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com",
}

class NSEScraper:
    
    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """
        Returns upcoming corporate actions:
        dividends, bonus, splits, rights issues.
        """
        url = f"{BASE_URL}/corporates-corporateActions"
        params = {"index": "equities", "symbol": symbol}
        # Returns ex-date, action type, value
        ...
    
    def get_quarterly_results(self, symbol: str) -> pd.DataFrame:
        """
        Returns last 8 quarters of financial results.
        """
        ...
    
    def get_announcements(self, symbol: str,
                          days_back: int = 30) -> list:
        """
        Returns recent corporate announcements.
        Filter for: earnings, board meetings, pledging changes.
        """
        ...
    
    def get_bulk_deals(self, date: str = None) -> pd.DataFrame:
        """
        Returns bulk/block deals — large institutional transactions.
        These are influential market orders in the Hawkes model.
        """
        ...
```

---

## Phase 2 — Fundamental Scoring Model (Week 2-3)

### Task 2.1 — Quality Score Formula

Score each symbol 0-100. Higher = better MM candidate.

```python
# src/fundamentals/models/quality_score.py

def compute_quality_score(data: dict) -> float:
    """
    Score 0-100 measuring fundamental quality.
    
    HIGH score (>70):  strong business, trade aggressively
    MID score (40-70): average, trade normally
    LOW score (<40):   weak, widen spreads
    """
    score = 0.0
    
    # ── Profitability (30 points) ──────────────────────────────────────
    roe = data.get('roe', 0)
    if roe > 20:    score += 10
    elif roe > 15:  score += 7
    elif roe > 10:  score += 4
    
    roce = data.get('roce', 0)
    if roce > 15:   score += 10
    elif roce > 12: score += 7
    elif roce > 8:  score += 4
    
    margin = data.get('net_profit_margin', 0)
    if margin > 15:  score += 10
    elif margin > 8: score += 7
    elif margin > 4: score += 4
    
    # ── Growth (25 points) ────────────────────────────────────────────
    rev_growth = data.get('revenue_growth_yoy', 0)
    if rev_growth > 20:  score += 10
    elif rev_growth > 10: score += 7
    elif rev_growth > 5:  score += 4
    
    eps_growth = data.get('eps_growth_yoy', 0)
    if eps_growth > 25:  score += 15
    elif eps_growth > 15: score += 10
    elif eps_growth > 5:  score += 5
    
    # ── Balance Sheet (25 points) ─────────────────────────────────────
    dte = data.get('debt_to_equity', 999)
    if dte < 0.3:   score += 15
    elif dte < 0.7: score += 10
    elif dte < 1.0: score += 5
    elif dte < 2.0: score += 2
    # > 2.0: 0 points — high leverage is a red flag for MM
    
    cr = data.get('current_ratio', 0)
    if cr > 2.0:   score += 10
    elif cr > 1.5: score += 7
    elif cr > 1.0: score += 4
    
    # ── Earnings Quality (20 points) ──────────────────────────────────
    fcf = data.get('free_cash_flow', 0)
    net_profit = data.get('net_profit', 1)
    cash_conversion = fcf / max(abs(net_profit), 1)
    if cash_conversion > 0.9:   score += 10
    elif cash_conversion > 0.7: score += 7
    elif cash_conversion > 0.5: score += 4
    
    promoter_holding = data.get('promoter_holding', 0)
    if promoter_holding > 60: score += 10
    elif promoter_holding > 50: score += 7
    elif promoter_holding > 40: score += 4
    
    # ── Penalties ─────────────────────────────────────────────────────
    # Promoter pledging is a major red flag
    pledge_pct = data.get('promoter_pledge_pct', 0)
    if pledge_pct > 50: score -= 20
    elif pledge_pct > 30: score -= 10
    elif pledge_pct > 10: score -= 5
    
    # Audit qualifications
    if data.get('audit_qualification', False):
        score -= 15
    
    return max(0.0, min(100.0, score))
```

### Task 2.2 — Earnings Surprise Tracker

```python
# src/fundamentals/models/earnings_predictor.py

class EarningsSurpriseTracker:
    """
    Tracks historical earnings surprises per symbol.
    
    Key insight: companies that consistently beat estimates
    have positive alpha tendency — feeds into V2 Hawkes model.
    """
    
    def compute_beat_rate(self, symbol: str,
                          quarters: int = 8) -> dict:
        """
        Returns:
          beat_rate:        fraction of quarters that beat estimates
          avg_surprise_pct: average surprise magnitude
          trend:            'improving', 'declining', 'stable'
        """
        history = self.db.get_earnings_history(symbol, quarters)
        
        surprises = [
            (q.eps_actual - q.eps_estimate) / abs(q.eps_estimate) * 100
            for q in history if q.eps_estimate != 0
        ]
        
        if not surprises:
            return {'beat_rate': 0.5, 'avg_surprise_pct': 0, 'trend': 'stable'}
        
        beat_rate = sum(1 for s in surprises if s > 0) / len(surprises)
        avg_surprise = sum(surprises) / len(surprises)
        
        # Trend: compare last 2 quarters vs previous
        if len(surprises) >= 4:
            recent = sum(surprises[-2:]) / 2
            older  = sum(surprises[-4:-2]) / 2
            if recent > older + 2:   trend = 'improving'
            elif recent < older - 2: trend = 'declining'
            else:                    trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'beat_rate':        round(beat_rate, 3),
            'avg_surprise_pct': round(avg_surprise, 2),
            'trend':            trend,
        }
    
    def alpha_prior_from_earnings(self, symbol: str) -> float:
        """
        Convert earnings history to alpha prior for V2 model.
        
        Consistent beaters → positive alpha prior
        Consistent missers → negative alpha prior
        
        Returns value in range [-0.005, +0.005]
        """
        stats = self.compute_beat_rate(symbol)
        
        # Scale: 50% beat rate = 0 alpha, 100% = +0.005, 0% = -0.005
        raw = (stats['beat_rate'] - 0.5) * 0.01
        return max(-0.005, min(0.005, raw))
```

### Task 2.3 — Sector Rotation Tracker

```python
# src/fundamentals/models/sector_rotation.py

SECTOR_MAP = {
    # Financial Services
    'CHOLAFIN26MAYFUT':   'nbfc',
    'BAJFINANCE26MAYFUT': 'nbfc',
    'INDUSINDBK26MAYFUT': 'banking',
    'ICICIBANK26MAYFUT':  'banking',
    'AXISBANK26MAYFUT':   'banking',
    'HDFCBANK26MAYFUT':   'banking',
    
    # Metals
    'JSWSTEEL26MAYFUT':   'metals',
    'HINDALCO26MAYFUT':   'metals',
    'SAIL26MAYFUT':       'metals',
    'TATASTEEL26MAYFUT':  'metals',
    
    # Real Estate
    'OBEROIRLTY26MAYFUT': 'realty',
    'GODREJPROP26MAYFUT': 'realty',
    'PRESTIGE26MAYFUT':   'realty',
    'PHOENIXLTD26MAYFUT': 'realty',
    'DLF26MAYFUT':        'realty',
    
    # Technology
    'TECHM26MAYFUT':      'it',
    'INFY26MAYFUT':       'it',
    'HCLTECH26MAYFUT':    'it',
    'TCS26MAYFUT':        'it',
    'WIPRO26MAYFUT':      'it',
    
    # Pharma
    'SUNPHARMA26MAYFUT':  'pharma',
    'CIPLA26MAYFUT':      'pharma',
    'DRREDDY26MAYFUT':    'pharma',
    'LUPIN26MAYFUT':      'pharma',
    
    # FMCG
    'HINDUNILVR26MAYFUT': 'fmcg',
    'NESTLEIND26MAYFUT':  'fmcg',
    'BRITANNIA26MAYFUT':  'fmcg',
    
    # Industrials
    'HAVELLS26MAYFUT':    'industrials',
    'VOLTAS26MAYFUT':     'industrials',
    'BDL26MAYFUT':        'defence',
    
    # Energy
    'BPCL26MAYFUT':       'energy',
    'ONGC26MAYFUT':       'energy',
    'RELIANCE26MAYFUT':   'conglomerate',
}

# Peer groups for contagion signals
PEER_GROUPS = {
    'nbfc':    ['CHOLAFIN26MAYFUT', 'BAJFINANCE26MAYFUT', 'INDUSINDBK26MAYFUT'],
    'metals':  ['JSWSTEEL26MAYFUT', 'HINDALCO26MAYFUT', 'SAIL26MAYFUT'],
    'realty':  ['OBEROIRLTY26MAYFUT', 'GODREJPROP26MAYFUT', 'PRESTIGE26MAYFUT'],
    'it':      ['TECHM26MAYFUT', 'INFY26MAYFUT', 'HCLTECH26MAYFUT'],
    'pharma':  ['SUNPHARMA26MAYFUT', 'CIPLA26MAYFUT', 'DRREDDY26MAYFUT'],
}
```

---

## Phase 3 — Core Signal (Week 3-4)

### Task 3.1 — FundamentalSignal Dataclass

```python
# src/fundamentals/signal.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class FundamentalSignal:
    """
    Daily fundamental signal per symbol.
    Generated at 8:30am, used throughout the trading day.
    
    PRIMARY OUTPUTS (what the trading system actually uses):
      mm_recommendation  → aggressive / normal / cautious / avoid
      spread_multiplier  → multiply base spread by this value
      alpha_prior        → directional bias for Hawkes model
    
    Everything else supports computing those three values.
    """
    
    symbol:    str
    timestamp: datetime
    
    # ── Quality ───────────────────────────────────────────────────────
    quality_score:    float   # 0-100 (higher = better MM candidate)
    
    # ── Earnings ──────────────────────────────────────────────────────
    next_earnings:         Optional[datetime] = None
    days_to_earnings:      int   = 999      # 999 = unknown
    earnings_surprise_avg: float = 0.0      # avg surprise last 4Q (%)
    beat_rate:             float = 0.5      # fraction of Q that beat
    eps_trend:             str   = 'stable' # improving/declining/stable
    
    # ── Growth ────────────────────────────────────────────────────────
    revenue_growth: float = 0.0   # YoY %
    eps_growth:     float = 0.0   # YoY %
    
    # ── Risk Flags ────────────────────────────────────────────────────
    high_debt:        bool  = False   # debt_to_equity > 2
    promoter_pledge:  float = 0.0    # % pledged (>30% = warning)
    audit_issues:     bool  = False   # any audit qualifications
    
    # ── Sector ────────────────────────────────────────────────────────
    sector:       str   = 'unknown'
    sector_score: float = 50.0      # sector strength 0-100
    
    # ── PRIMARY OUTPUTS ───────────────────────────────────────────────
    mm_recommendation: str   = 'normal'  # aggressive/normal/cautious/avoid
    spread_multiplier: float = 1.0       # multiply base spread by this
    alpha_prior:       float = 0.0       # directional bias (-0.005 to +0.005)
    
    # ── Metadata ──────────────────────────────────────────────────────
    data_freshness:    str  = 'stale'    # fresh/stale/missing
    confidence:        float = 0.5       # 0-1 confidence in signal
```

### Task 3.2 — FundamentalEngine

```python
# src/fundamentals/engine.py

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class FundamentalEngine:
    """
    Main engine. Run update_all_symbols() every morning at 8:30am.
    
    Usage:
        engine  = FundamentalEngine()
        signals = engine.update_all_symbols()
        
        # In trading strategy:
        fund = signals.get('CHOLAFIN26MAYFUT')
        base_spread *= fund.spread_multiplier
    """
    
    CACHE_PATH = Path('src/fundamentals/data/signals_cache.json')
    
    def __init__(self):
        from src.fundamentals.scrapers.nse_scraper import NSEScraper
        from src.fundamentals.scrapers.screener_scraper import ScreenerScraper
        from src.fundamentals.scrapers.earnings_scraper import EarningsScraper
        from src.fundamentals.models.quality_score import compute_quality_score
        from src.fundamentals.models.earnings_predictor import EarningsSurpriseTracker
        from src.fundamentals.models.sector_rotation import SECTOR_MAP, PEER_GROUPS
        
        self.nse       = NSEScraper()
        self.screener  = ScreenerScraper()
        self.earnings  = EarningsScraper()
        self.quality   = compute_quality_score
        self.tracker   = EarningsSurpriseTracker()
        self.sector_map = SECTOR_MAP
        self._cache    = {}
    
    def update_all_symbols(self,
                            symbols: list = None) -> dict:
        """
        Run at 8:30am before market open.
        Returns dict of symbol → FundamentalSignal.
        Takes ~5 minutes for 85 symbols.
        """
        from src.backtest.data_loader import TRADING_UNIVERSE
        symbols = symbols or list(TRADING_UNIVERSE)
        
        signals = {}
        for i, symbol in enumerate(symbols):
            try:
                signals[symbol] = self._compute_signal(symbol)
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(symbols)}")
            except Exception as e:
                logger.warning(f"Failed {symbol}: {e}")
                signals[symbol] = self._default_signal(symbol)
        
        # Cache to disk for dashboard and recovery
        self._save_cache(signals)
        return signals
    
    def _compute_signal(self, symbol: str) -> FundamentalSignal:
        from src.fundamentals.signal import FundamentalSignal
        
        # Strip expiry suffix: CHOLAFIN26MAYFUT → CHOLAFIN
        name = symbol.replace('26MAYFUT', '').replace('26JUNFUT', '')
        
        # Get data
        financial_data   = self.screener.get_ratios(name)
        earnings_history = self.tracker.compute_beat_rate(name)
        next_earnings    = self.earnings.get_next_date(name)
        
        # Compute quality
        quality = self.quality(financial_data)
        
        # Days to earnings
        days_to_earnings = 999
        if next_earnings:
            days_to_earnings = (next_earnings - datetime.now()).days
        
        # Sector
        sector = self.sector_map.get(symbol, 'unknown')
        
        # Alpha prior
        alpha_prior = self.tracker.alpha_prior_from_earnings(name)
        
        # Compute recommendation and spread multiplier
        rec, spread_mult = self._compute_recommendation(
            quality          = quality,
            days_to_earnings = days_to_earnings,
            high_debt        = financial_data.get('debt_to_equity', 0) > 2,
            promoter_pledge  = financial_data.get('promoter_pledge_pct', 0),
            audit_issues     = financial_data.get('audit_qualification', False),
        )
        
        return FundamentalSignal(
            symbol               = symbol,
            timestamp            = datetime.now(),
            quality_score        = quality,
            next_earnings        = next_earnings,
            days_to_earnings     = days_to_earnings,
            earnings_surprise_avg = earnings_history['avg_surprise_pct'],
            beat_rate            = earnings_history['beat_rate'],
            eps_trend            = earnings_history['trend'],
            revenue_growth       = financial_data.get('revenue_growth_yoy', 0),
            eps_growth           = financial_data.get('eps_growth_yoy', 0),
            high_debt            = financial_data.get('debt_to_equity', 0) > 2,
            promoter_pledge      = financial_data.get('promoter_pledge_pct', 0),
            audit_issues         = financial_data.get('audit_qualification', False),
            sector               = sector,
            mm_recommendation    = rec,
            spread_multiplier    = spread_mult,
            alpha_prior          = alpha_prior,
            data_freshness       = 'fresh',
            confidence           = 0.8 if financial_data else 0.3,
        )
    
    def _compute_recommendation(self,
                                  quality:          float,
                                  days_to_earnings: int,
                                  high_debt:        bool,
                                  promoter_pledge:  float,
                                  audit_issues:     bool
                                  ) -> tuple[str, float]:
        """
        Returns (recommendation, spread_multiplier).
        
        Spread multiplier guide:
          0.8 = tighten 20% (be more aggressive — better fills)
          1.0 = no change
          1.5 = widen 50% (be more cautious)
          2.0 = widen 100% (very cautious)
          0.0 = avoid (don't trade)
        """
        
        # Hard stops
        if audit_issues:
            return 'avoid', 0.0
        
        if promoter_pledge > 50:
            return 'avoid', 0.0
        
        if high_debt and quality < 30:
            return 'avoid', 0.0
        
        # Earnings proximity — most important signal
        if days_to_earnings <= 1:
            return 'cautious', 2.0    # earnings today/tomorrow — very wide
        
        if days_to_earnings <= 3:
            return 'cautious', 1.7    # earnings this week — wide
        
        if days_to_earnings <= 7:
            return 'cautious', 1.3    # earnings next week — slightly wide
        
        # Quality-based recommendation
        if quality >= 70:
            if high_debt:
                return 'normal', 1.0
            return 'aggressive', 0.85   # strong fundamentals → tighten
        
        if quality >= 50:
            return 'normal', 1.0
        
        if quality >= 30:
            return 'cautious', 1.2
        
        # Poor fundamentals
        if promoter_pledge > 30:
            return 'cautious', 1.5
        
        return 'cautious', 1.3
    
    def _default_signal(self, symbol: str):
        """Return neutral signal when data unavailable."""
        from src.fundamentals.signal import FundamentalSignal
        return FundamentalSignal(
            symbol            = symbol,
            timestamp         = datetime.now(),
            quality_score     = 50.0,
            mm_recommendation = 'normal',
            spread_multiplier = 1.0,
            alpha_prior       = 0.0,
            data_freshness    = 'missing',
            confidence        = 0.0,
        )
    
    def _save_cache(self, signals: dict):
        """Save to disk for dashboard and recovery."""
        cache = {}
        for sym, sig in signals.items():
            cache[sym] = {
                'quality_score':     sig.quality_score,
                'days_to_earnings':  sig.days_to_earnings,
                'mm_recommendation': sig.mm_recommendation,
                'spread_multiplier': sig.spread_multiplier,
                'alpha_prior':       sig.alpha_prior,
                'beat_rate':         sig.beat_rate,
                'timestamp':         sig.timestamp.isoformat(),
            }
        self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
    
    def load_from_cache(self) -> dict:
        """Load yesterday's signals if engine hasn't run yet today."""
        if not self.CACHE_PATH.exists():
            return {}
        with open(self.CACHE_PATH) as f:
            return json.load(f)
```

### Task 3.3 — Integration with Trading Strategy

In `src/backtest/mm_strategy.py` add these two methods:

```python
# Add to AvellanedaStoikovStrategy:

def on_day_start(self, date: str) -> None:
    """Called at 9:00am — load fundamental signals."""
    try:
        from src.fundamentals.engine import FundamentalEngine
        engine = FundamentalEngine()
        self._fund_signal = engine.load_from_cache().get(
            self.config.symbol, None
        )
    except Exception:
        self._fund_signal = None

# In on_bar(), after computing base_spread, add:
#
#   if hasattr(self, '_fund_signal') and self._fund_signal:
#       fund = self._fund_signal
#
#       # Hard stop
#       if fund['mm_recommendation'] == 'avoid':
#           book.cancel_all()
#           return
#
#       # Adjust spread
#       base_spread *= fund['spread_multiplier']
#
#   spread = base_spread  # continue normally
```

---

## Phase 4 — Streamlit Dashboard (Month 2)

### Task 4.1 — Build the Dashboard

```python
# src/fundamentals/dashboard/app.py
# Run with: streamlit run src/fundamentals/dashboard/app.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Fundamental Signals", layout="wide")
st.title("Fundamental Signal Dashboard")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Load signals
cache_path = Path('src/fundamentals/data/signals_cache.json')
if cache_path.exists():
    with open(cache_path) as f:
        signals = json.load(f)
else:
    st.error("No signals found. Run FundamentalEngine.update_all_symbols() first.")
    st.stop()

# Build dataframe
rows = []
for sym, sig in signals.items():
    rows.append({
        'Symbol':           sym.replace('26MAYFUT', ''),
        'Quality':          sig['quality_score'],
        'Days to Earnings': sig['days_to_earnings'],
        'Beat Rate':        f"{sig['beat_rate']:.0%}",
        'Recommendation':   sig['mm_recommendation'],
        'Spread Mult':      sig['spread_multiplier'],
        'Alpha Prior':      sig['alpha_prior'],
    })

df = pd.DataFrame(rows).sort_values('Quality', ascending=False)

# Color code by recommendation
def color_rec(val):
    colors = {
        'aggressive': 'background-color: #90EE90',   # green
        'normal':     '',
        'cautious':   'background-color: #FFD700',   # yellow
        'avoid':      'background-color: #FF6B6B',   # red
    }
    return colors.get(val, '')

# Display
st.dataframe(
    df.style.applymap(color_rec, subset=['Recommendation']),
    use_container_width=True,
    height=600,
)

# Alerts
st.subheader("Alerts")
earnings_soon = [r for r in rows if r['Days to Earnings'] <= 3
                 and r['Days to Earnings'] >= 0]
if earnings_soon:
    st.warning(f"Earnings in next 3 days: "
               f"{', '.join(r['Symbol'] for r in earnings_soon)}")

avoid_list = [r for r in rows if r['Recommendation'] == 'avoid']
if avoid_list:
    st.error(f"AVOID today: "
             f"{', '.join(r['Symbol'] for r in avoid_list)}")
```

---

## Phase 5 — Backtesting Fundamental Signals (Month 3)

### Task 5.1 — Does Quality Score Predict MM Performance?

Run this analysis after building quality scores for all 85 symbols:

```python
# research/notebooks/08_fundamental_quality_vs_mm_performance.ipynb

import json
import pandas as pd
import matplotlib.pyplot as plt

# Load quality scores
with open('src/fundamentals/data/signals_cache.json') as f:
    signals = json.load(f)

# Load 8-day backtest results
with open('research/findings/v1_optimal_params.json') as f:
    backtest = json.load(f)

# Build comparison dataframe
rows = []
for sym in set(signals) & set(backtest):
    rows.append({
        'symbol':      sym,
        'quality':     signals[sym]['quality_score'],
        'test_sharpe': backtest[sym]['test_sharpe'],
        'test_pnl':    backtest[sym]['test_pnl'],
        'profitable':  backtest[sym]['test_pnl'] > 0,
    })

df = pd.DataFrame(rows)

# Plot quality vs sharpe
plt.scatter(df['quality'], df['test_sharpe'])
plt.xlabel('Quality Score')
plt.ylabel('OOS Sharpe')
plt.title('Does Fundamental Quality Predict MM Performance?')
plt.savefig('research/findings/quality_vs_sharpe.png')

# Expected: positive correlation
# If no correlation: fundamentals don't help symbol selection
# If positive correlation: use quality score to filter symbols
print(df.corr()[['quality', 'test_sharpe', 'profitable']])
```

### Task 5.2 — Does Earnings Filter Improve Performance?

```python
# research/notebooks/09_earnings_filter_backtest.ipynb

# Hypothesis: avoid trading 2 days before and after earnings
# Test: compare backtest WITH and WITHOUT earnings filter

# Without filter: current results (already have)
# With filter: re-run backtest skipping earnings dates

# Load earnings dates from your calendar
earnings_dates = pd.read_csv('src/fundamentals/data/earnings_calendar.csv')

# For each symbol, identify earnings dates in backtest period
# Re-run fast backtest with those dates removed

# Compare:
#   Sharpe with filter vs without
#   PnL with filter vs without
#   Max drawdown with filter vs without

# Expected: filter reduces total PnL but improves Sharpe and reduces drawdown
```

---

## Earnings Calendar — Manual Maintenance

This is the most important file to keep updated.
Update every week by checking NSE/BSE announcements.

```csv
# src/fundamentals/data/earnings_calendar.csv
symbol,earnings_date,quarter,eps_estimate,eps_actual,surprise_pct
CHOLAFIN26MAYFUT,2026-04-25,Q4FY26,18.5,,
JSWSTEEL26MAYFUT,2026-05-02,Q4FY26,12.2,,
OBEROIRLTY26MAYFUT,2026-05-08,Q4FY26,8.1,,
GODREJPROP26MAYFUT,2026-05-10,Q4FY26,15.3,,
HAVELLS26MAYFUT,2026-05-12,Q4FY26,6.8,,
```

Fill in `eps_actual` and `surprise_pct` after each announcement.

---

## Key Rules for Your Brother

```
1. EARNINGS CALENDAR IS SACRED
   Update it every Monday morning.
   Wrong earnings dates = wrong spread multipliers = losses.

2. NEVER GO LIVE WITHOUT TESTING
   Every signal must be backtested before wiring into strategy.
   Use research/notebooks/ for all analysis.

3. DEFAULT TO NEUTRAL
   When data is missing or stale → spread_multiplier = 1.0
   Never let missing data cause aggressive trading.

4. QUALITY SCORE IS A GUIDE NOT A RULE
   BAJFINANCE has good fundamentals but still loses in MM.
   Fundamentals help at the margin — they don't override
   the microstructure signals from the Hawkes model.

5. LOG EVERYTHING
   Every signal generated should be logged with timestamp.
   We need to audit why signals were given on any given day.
```

---

## Questions to Answer in Research Notebooks

Before building anything, answer these with data:

```
1. Do high quality score symbols have better MM Sharpe? (NB08)
2. Does avoiding earnings days improve risk-adjusted returns? (NB09)
3. Which sectors have the best/worst MM performance? (NB10)
4. Does promoter pledging predict bad MM days? (NB11)
5. Does peer earnings contagion signal help? (NB12)
```

Start with question 1. If quality score doesn't predict MM performance,
questions 2-5 may not be worth building.

---

## Summary — Week by Week

```
WEEK 1:
  □ Set up NSE/BSE scrapers
  □ Build earnings calendar for 85 symbols (PRIORITY)
  □ Collect last 8 quarters financial data
  □ Build quality_score.py

WEEK 2:
  □ Build EarningsSurpriseTracker
  □ Build sector map and peer groups
  □ Implement FundamentalSignal dataclass
  □ Test quality scores — do they make sense?

WEEK 3:
  □ Build FundamentalEngine.update_all_symbols()
  □ Validate earnings calendar accuracy
  □ Wire spread_multiplier into mm_strategy.py
  □ Test: does it break anything?

WEEK 4:
  □ NB08: quality score vs MM performance
  □ NB09: earnings filter backtest
  □ Build Streamlit dashboard
  □ Set up daily 8:30am automated run

MONTH 2:
  □ Build earnings surprise ML model
  □ Build peer contagion signals
  □ Build promoter pledging alerts
  □ Wire alpha_prior into V2 strategy

MONTH 3:
  □ Full backtesting of all signals
  □ Integrate with news module
  □ Combined fundamental + news + Hawkes signal
  □ Final validation before live trading
```

---

## Contact / Questions

share this after each week:
1. Quality scores for top 20 symbols (does it make sense?)
2. Earnings calendar (is it complete and accurate?)
3. Notebook results (does quality predict MM performance?)

The earnings calendar is the most important deliverable.
Everything else can wait if needed.
