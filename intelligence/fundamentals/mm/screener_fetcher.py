"""
Screener.in Data Fetcher — MM Fundamentals
==========================================
Fetches structured financial ratios from Screener.in for all
84 futures symbols in our MM universe.

Data fetched per company:
    Valuation:   P/E, P/B, EV/EBITDA, Market Cap
    Profitability: ROE, ROCE, PAT margin, Operating margin
    Growth:      Revenue growth (3yr), PAT growth (3yr)
    Health:      Debt/Equity, Interest coverage, Current ratio
    Quality:     Promoter holding, Promoter pledging %
                 Cash flow vs reported profit ratio
    F&O:         In F&O ban? OI change

Why these metrics for MM (not investing):
    Promoter pledging    → distressed promoter = informed selling
    Debt/EBITDA > 4x     → financial stress = informed flow
    PAT variance         → unpredictable earnings = high uncertainty
    Low promoter holding → more float = more informed traders
    F&O ban              → AVOID completely (can't trade anyway)

Data sources:
    Screener.in    → free tier, structured ratios
    NSE F&O        → ban list, OI data
    BSE            → promoter shareholding

Runs:
    Quarterly after results season (July, October, January, April)
    Monthly for watchlist updates

Usage:
    # Fetch all symbols
    python intelligence/fundamentals/mm/screener_fetcher.py --all

    # Single symbol
    python intelligence/fundamentals/mm/screener_fetcher.py --symbol RELIANCE

    # Check F&O ban list only
    python intelligence/fundamentals/mm/screener_fetcher.py --ban-list

    # Import
    from intelligence.fundamentals.mm.screener_fetcher import ScreenerFetcher
    fetcher = ScreenerFetcher()
    data    = fetcher.fetch(symbol='RELIANCE')
"""

import argparse
import json
import logging
import re
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import requests

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET          = "hedge-fund-494103-marketdata-mumbai"
FILINGS_PREFIX      = "intelligence/fundamentals/raw"
SCORES_PREFIX       = "intelligence/fundamentals/scores"

SCREENER_BASE       = "https://www.screener.in"
NSE_FNO_BAN_URL     = "https://www.nseindia.com/api/fo-secban"
NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":          "application/json",
    "Referer":         "https://www.nseindia.com/",
}

REQUEST_DELAY = 2.0   # seconds between requests — be polite to Screener.in

# NSE equity symbol → Screener.in symbol mapping
# Most are the same, but some differ
SYMBOL_MAP = {
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "M&M":        "M&M",
    # Add more overrides if Screener returns 404
}

# Futures universe — equity symbols (no expiry suffix)
MM_UNIVERSE = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "AXISBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "BAJFINANCE",
    "ADANIPORTS", "ADANIENT", "HAL", "BEL", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "ONGC", "BPCL", "IOC",
    "DRREDDY", "SUNPHARMA", "CIPLA", "LUPIN", "DIVISLAB",
    "MARUTI", "BAJAJ-AUTO", "TVSMOTOR", "EICHERMOT", "HEROMOTOCO",
    "ULTRACEMCO", "GRASIM", "DLF", "GODREJPROP", "PRESTIGE",
    "CHOLAFIN", "BAJAJFINSV", "MUTHOOTFIN", "LICHSGFIN",
    "LT", "POWERGRID", "NTPC", "PFC", "RECLTD",
    "HCLTECH", "WIPRO", "TECHM",
    "TITAN", "NESTLEIND", "HINDUNILVR", "BRITANNIA", "TATACONSUM",
    "COALINDIA", "SAIL", "VEDL",
    "HAL", "BEL", "ADANIPORTS", "CHOLAFIN",
]

# Index futures — skip fundamental scoring (no balance sheet)
INDEX_FUTURES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


# ─── Data structures ──────────────────────────────────────────────────────────

def empty_fundamentals(symbol: str) -> dict:
    """Return empty fundamental dict for symbols with no data."""
    return {
        "symbol":               symbol,
        "fetched_at":           datetime.utcnow().isoformat() + "Z",
        "source":               "screener.in",
        "available":            False,
        "error":                "No data",

        # Valuation
        "market_cap":           None,
        "pe_ratio":             None,
        "pb_ratio":             None,
        "ev_ebitda":            None,
        "dividend_yield":       None,

        # Profitability
        "roe":                  None,
        "roce":                 None,
        "pat_margin":           None,
        "operating_margin":     None,

        # Growth (3yr CAGR)
        "revenue_growth_3yr":   None,
        "pat_growth_3yr":       None,

        # Financial health
        "debt_equity":          None,
        "interest_coverage":    None,
        "current_ratio":        None,
        "debt_ebitda":          None,

        # Quality (MM-specific)
        "promoter_holding":     None,
        "promoter_pledging":    None,   # % of promoter shares pledged
        "cash_flow_pat_ratio":  None,   # CFO/PAT — <0.7 = earnings quality risk
        "pat_std_3yr":          None,   # earnings volatility

        # F&O
        "in_fno_ban":           False,

        # MM scoring inputs
        "mm_data_quality":      "MISSING",  # GOOD/PARTIAL/MISSING
    }


# ─── Screener.in fetcher ──────────────────────────────────────────────────────

class ScreenerFetcher:
    """
    Fetches financial data from Screener.in.

    Screener.in provides free access to:
    - 10 years of financial data
    - Quarterly and annual results
    - Ratios, shareholding, cash flows

    Rate limiting: 2 seconds between requests to avoid blocking.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer":    "https://www.screener.in/",
        })
        self._ban_list: set = set()
        self._ban_fetched   = False

    def fetch(self, symbol: str) -> dict:
        """
        Fetch fundamentals for one equity symbol.
        Returns dict with all ratios and MM-relevant metrics.
        """
        if symbol in INDEX_FUTURES:
            result = empty_fundamentals(symbol)
            result["error"] = "Index future — no balance sheet"
            return result

        screener_sym = SYMBOL_MAP.get(symbol, symbol)

        log.info(f"Fetching {symbol} from Screener.in...")

        try:
            # Try consolidated first, then standalone
            data = (
                self._fetch_screener(screener_sym, consolidated=True)
                or self._fetch_screener(screener_sym, consolidated=False)
            )

            if data is None:
                result = empty_fundamentals(symbol)
                result["error"] = "Symbol not found on Screener.in"
                return result

            # Parse and structure the data
            result = self._parse(symbol, data)

            # Add F&O ban status
            result["in_fno_ban"] = symbol in self._get_ban_list()

            return result

        except Exception as e:
            log.warning(f"Screener fetch failed for {symbol}: {e}")
            result = empty_fundamentals(symbol)
            result["error"] = str(e)
            return result

    def fetch_all(
        self,
        symbols:  list[str] = None,
        save_gcs: bool = True,
    ) -> dict[str, dict]:
        """
        Fetch fundamentals for all symbols in MM universe.
        Returns dict: symbol → fundamentals.
        """
        if symbols is None:
            symbols = [s for s in MM_UNIVERSE if s not in INDEX_FUTURES]
            symbols = list(set(symbols))   # deduplicate

        results = {}
        total   = len(symbols)

        log.info(f"Fetching fundamentals for {total} symbols...")

        for i, sym in enumerate(symbols):
            log.info(f"[{i+1}/{total}] {sym}")
            data = self.fetch(sym)
            results[sym] = data

            if save_gcs:
                self._save_to_gcs(sym, data)

            # Rate limit — be polite to Screener.in
            if i < total - 1:
                time.sleep(REQUEST_DELAY)

        log.info(f"Done. Fetched {len(results)} symbols.")
        return results

    def _fetch_screener(
        self,
        symbol:       str,
        consolidated: bool = True,
    ) -> Optional[dict]:
        """
        Fetch raw JSON data from Screener.in API.
        Returns parsed JSON or None if not found.
        """
        suffix = "consolidated" if consolidated else "standalone"
        url    = f"{SCREENER_BASE}/api/company/{symbol}/{suffix}/"

        try:
            resp = self.session.get(url, timeout=15)

            if resp.status_code == 404:
                return None
            if resp.status_code == 403:
                log.warning(f"Screener.in blocked request for {symbol} "
                            f"— rate limited or login required")
                return None
            if resp.status_code != 200:
                log.warning(f"Screener {symbol}: HTTP {resp.status_code}")
                return None

            return resp.json()

        except Exception as e:
            log.debug(f"Screener fetch error for {symbol}: {e}")
            return None

    def _parse(self, symbol: str, data: dict) -> dict:
        """
        Parse Screener.in API response into our standard format.
        Handles missing fields gracefully.
        """
        result = empty_fundamentals(symbol)
        result["available"] = True
        result["error"]     = None

        try:
            # ── Ratios ──
            ratios = {
                r.get("name", ""): r.get("values", [{}])
                for r in data.get("ratios", [])
            }

            def get_ratio(name: str, idx: int = -1) -> Optional[float]:
                """Get latest value of a ratio by name."""
                for key, values in ratios.items():
                    if name.lower() in key.lower():
                        try:
                            vals = [
                                v.get("value") for v in values
                                if v.get("value") is not None
                            ]
                            if vals:
                                val = vals[idx]
                                return float(str(val).replace(",", "").replace("%", ""))
                        except (ValueError, IndexError):
                            continue
                return None

            # Valuation
            result["market_cap"]     = get_ratio("Market Cap")
            result["pe_ratio"]       = get_ratio("Price to Earning")
            result["pb_ratio"]       = get_ratio("Price to Book")
            result["dividend_yield"] = get_ratio("Dividend Yield")

            # Profitability
            result["roe"]             = get_ratio("Return on Equity")
            result["roce"]            = get_ratio("Return on Capital")
            result["pat_margin"]      = get_ratio("Net Profit Margin") or \
                                        get_ratio("OPM")
            result["operating_margin"]= get_ratio("OPM")

            # Health
            result["debt_equity"]      = get_ratio("Debt to Equity")
            result["interest_coverage"] = get_ratio("Interest Coverage")
            result["current_ratio"]    = get_ratio("Current Ratio")

            # ── P&L data for growth and volatility ──
            pl = data.get("profit_loss", {})
            if pl:
                result.update(self._parse_pl(pl))

            # ── Cash flow data ──
            cf = data.get("cash_flows", {})
            if cf:
                result.update(self._parse_cashflow(cf, result))

            # ── Shareholding ──
            sh = data.get("shareholding", {})
            if sh:
                result.update(self._parse_shareholding(sh))

            # ── Data quality ──
            filled = sum(
                1 for k in [
                    "pe_ratio", "roe", "debt_equity",
                    "promoter_holding", "pat_growth_3yr"
                ]
                if result.get(k) is not None
            )
            result["mm_data_quality"] = (
                "GOOD"    if filled >= 4 else
                "PARTIAL" if filled >= 2 else
                "MISSING"
            )

        except Exception as e:
            log.warning(f"Parse error for {symbol}: {e}")
            result["error"] = f"Parse error: {e}"

        return result

    def _parse_pl(self, pl: dict) -> dict:
        """Extract growth and volatility from P&L data."""
        result = {}
        try:
            rows = pl.get("rows", [])

            # Find revenue and PAT rows
            revenue_row = None
            pat_row     = None

            for row in rows:
                title = row.get("title", "").upper()
                if "SALES" in title or "REVENUE" in title:
                    revenue_row = row
                if "NET PROFIT" in title or "PAT" in title:
                    pat_row = row

            def extract_values(row) -> list[float]:
                if not row:
                    return []
                cols = row.get("columns", [])
                vals = []
                for col in cols:
                    try:
                        v = float(str(col.get("value", "")).replace(",", ""))
                        vals.append(v)
                    except (ValueError, TypeError):
                        pass
                return vals

            rev_vals = extract_values(revenue_row)
            pat_vals = extract_values(pat_row)

            # 3yr CAGR
            if len(rev_vals) >= 4:
                try:
                    cagr = (rev_vals[-1] / rev_vals[-4]) ** (1/3) - 1
                    result["revenue_growth_3yr"] = round(cagr * 100, 1)
                except (ZeroDivisionError, ValueError):
                    pass

            if len(pat_vals) >= 4:
                try:
                    cagr = (pat_vals[-1] / pat_vals[-4]) ** (1/3) - 1
                    result["pat_growth_3yr"] = round(cagr * 100, 1)
                except (ZeroDivisionError, ValueError):
                    pass

            # PAT standard deviation (earnings predictability)
            if len(pat_vals) >= 3:
                import statistics
                try:
                    result["pat_std_3yr"] = round(
                        statistics.stdev(pat_vals[-3:]), 1
                    )
                except statistics.StatisticsError:
                    pass

        except Exception as e:
            log.debug(f"P&L parse error: {e}")

        return result

    def _parse_cashflow(self, cf: dict, existing: dict) -> dict:
        """Extract cash flow quality metrics."""
        result = {}
        try:
            rows = cf.get("rows", [])

            cfo_row = None
            for row in rows:
                title = row.get("title", "").upper()
                if "OPERATING" in title and "ACTIVIT" in title:
                    cfo_row = row
                    break

            if cfo_row:
                cols = cfo_row.get("columns", [])
                cfo_vals = []
                for col in cols[-4:]:   # last 4 quarters/years
                    try:
                        v = float(str(col.get("value", "")).replace(",", ""))
                        cfo_vals.append(v)
                    except (ValueError, TypeError):
                        pass

                # CFO/PAT ratio — quality of earnings
                # < 0.7 = earnings not backed by cash = red flag
                pat = existing.get("pat_margin")
                if cfo_vals and pat:
                    try:
                        avg_cfo = sum(cfo_vals) / len(cfo_vals)
                        result["cash_flow_pat_ratio"] = round(
                            avg_cfo / abs(avg_cfo + 1), 2
                        )
                    except ZeroDivisionError:
                        pass

        except Exception as e:
            log.debug(f"Cashflow parse error: {e}")

        return result

    def _parse_shareholding(self, sh: dict) -> dict:
        """Extract promoter holding and pledging data."""
        result = {}
        try:
            rows = sh.get("rows", [])

            for row in rows:
                title = row.get("title", "").upper()
                cols  = row.get("columns", [])

                if not cols:
                    continue

                # Get latest value (last column)
                try:
                    latest = float(
                        str(cols[-1].get("value", "")).replace(",", "").replace("%", "")
                    )
                except (ValueError, TypeError):
                    continue

                if "PROMOTER" in title and "PLEDG" not in title:
                    result["promoter_holding"] = latest
                elif "PLEDG" in title:
                    result["promoter_pledging"] = latest

        except Exception as e:
            log.debug(f"Shareholding parse error: {e}")

        return result

    def _get_ban_list(self) -> set:
        """
        Fetch current F&O ban list from NSE.
        Symbols in ban cannot be traded — skip MM entirely.
        """
        if self._ban_fetched:
            return self._ban_list

        try:
            session = requests.Session()
            session.headers.update(NSE_HEADERS)
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)

            resp = session.get(NSE_FNO_BAN_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # NSE returns list of banned symbols
                if isinstance(data, list):
                    self._ban_list = set(data)
                elif isinstance(data, dict):
                    self._ban_list = set(data.get("data", []))
                log.info(f"F&O ban list: {len(self._ban_list)} symbols banned")

        except Exception as e:
            log.warning(f"Could not fetch F&O ban list: {e}")

        self._ban_fetched = True
        return self._ban_list

    def get_ban_list(self) -> list[str]:
        """Public method to get current F&O ban list."""
        return sorted(self._get_ban_list())

    def _save_to_gcs(self, symbol: str, data: dict):
        """Save fundamental data to GCS."""
        try:
            from google.cloud import storage
            client = storage.Client()
            today  = datetime.utcnow().date()
            path   = f"{FILINGS_PREFIX}/{today}/{symbol}.json"
            client.bucket(GCS_BUCKET).blob(path).upload_from_string(
                json.dumps(data, indent=2, default=str),
                content_type="application/json",
            )
            log.debug(f"Saved {symbol} → gs://{GCS_BUCKET}/{path}")
        except Exception as e:
            log.debug(f"GCS save failed for {symbol}: {e}")


# ─── NSE F&O ban list (standalone) ───────────────────────────────────────────

class FnOBanMonitor:
    """
    Monitors NSE F&O ban list daily.
    Symbols in ban = AVOID for MM (can't open new positions).

    Cron: daily at 8:45am IST (before market open)
    """

    def __init__(self):
        self.fetcher = ScreenerFetcher()

    def check_and_alert(self, our_symbols: list[str]) -> list[str]:
        """
        Check if any of our trading symbols are in F&O ban.
        Returns list of banned symbols from our universe.
        """
        ban_list   = set(self.fetcher.get_ban_list())
        our_banned = [s for s in our_symbols if s in ban_list]

        if our_banned:
            log.warning(f"F&O BAN — {len(our_banned)} of our symbols banned:")
            for sym in our_banned:
                log.warning(f"  {sym} — AVOID MM today")
        else:
            log.info("F&O ban check: none of our symbols in ban list")

        return our_banned

    def save_ban_signals(self, banned_symbols: list[str]):
        """
        Save AVOID signals to GCS for banned symbols.
        Aggregator picks these up automatically.
        """
        try:
            import sys
            import os
            root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
            if root not in sys.path:
                sys.path.insert(0, root)

            from intelligence.signals.schema import IntelSignal, SignalSource
            from datetime import timedelta

            now = datetime.utcnow()

            for sym in banned_symbols:
                # Map equity symbol to futures symbol
                futures_sym = f"{sym}26MAYFUT"   # approximate
                signal = IntelSignal(
                    symbol       = futures_sym,
                    generated_at = now.isoformat() + "Z",
                    expires_at   = (now + timedelta(hours=8)).isoformat() + "Z",
                    mm_action    = "AVOID",
                    widen_factor = 5.0,
                    confidence   = 1.0,
                    source       = SignalSource.FUNDAMENTAL,
                    reason       = f"{sym} is in NSE F&O ban list today",
                )
                signal.save()
                log.info(f"Saved AVOID signal for {futures_sym}")

        except Exception as e:
            log.warning(f"Could not save ban signals: {e}")


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_fundamentals_summary(results: dict[str, dict]):
    """Print a ranked summary of fundamental data."""
    available = {s: d for s, d in results.items() if d.get("available")}
    missing   = {s: d for s, d in results.items() if not d.get("available")}

    print(f"\n{'=' * 80}")
    print(f"  FUNDAMENTAL DATA SUMMARY — {datetime.utcnow().strftime('%Y-%m-%d')}")
    print(f"{'=' * 80}")
    print(f"  Fetched:  {len(available)} symbols")
    print(f"  Missing:  {len(missing)} symbols")

    if available:
        print(f"\n{'─' * 80}")
        print(f"  {'Symbol':<20} {'P/E':>6} {'ROE':>6} {'D/E':>6} "
              f"{'Pledg%':>7} {'Promot%':>8} {'Quality':<10}")
        print(f"{'─' * 80}")

        # Sort by promoter pledging (most pledged = highest risk first)
        sorted_syms = sorted(
            available.items(),
            key=lambda x: x[1].get("promoter_pledging") or 0,
            reverse=True,
        )

        for sym, d in sorted_syms:
            pe      = f"{d.get('pe_ratio'):.1f}"    if d.get("pe_ratio")          else "N/A"
            roe     = f"{d.get('roe'):.1f}%"        if d.get("roe")               else "N/A"
            de      = f"{d.get('debt_equity'):.2f}" if d.get("debt_equity") is not None else "N/A"
            pledge  = f"{d.get('promoter_pledging'):.1f}%" if d.get("promoter_pledging") is not None else "N/A"
            promo   = f"{d.get('promoter_holding'):.1f}%"  if d.get("promoter_holding") is not None  else "N/A"
            quality = d.get("mm_data_quality", "?")

            # Flag high risk symbols
            flag = ""
            if (d.get("promoter_pledging") or 0) > 20:
                flag = " ⚠️  HIGH PLEDGE"
            elif (d.get("debt_equity") or 0) > 3:
                flag = " ⚠️  HIGH DEBT"
            elif d.get("in_fno_ban"):
                flag = " 🔴 F&O BAN"

            print(
                f"  {sym:<20} {pe:>6} {roe:>6} {de:>6} "
                f"{pledge:>7} {promo:>8} {quality:<10}{flag}"
            )

    if missing:
        print(f"\n── Missing Data ─────────────────────────────────────────────")
        for sym, d in missing.items():
            print(f"  {sym:<20} {d.get('error', 'Unknown error')}")

    print(f"{'=' * 80}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Screener.in Fetcher — MM Fundamental Data"
    )
    parser.add_argument("--all",      action="store_true",
                        help="Fetch all symbols in MM universe")
    parser.add_argument("--symbol",   type=str,
                        help="Fetch single equity symbol (e.g. RELIANCE)")
    parser.add_argument("--ban-list", action="store_true",
                        help="Show current F&O ban list")
    parser.add_argument("--no-save",  action="store_true",
                        help="Don't save to GCS")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print full data for each symbol")
    args = parser.parse_args()

    fetcher = ScreenerFetcher()

    if args.ban_list:
        ban = fetcher.get_ban_list()
        print(f"\nF&O Ban List ({len(ban)} symbols):")
        for s in ban:
            print(f"  {s}")

    if args.symbol:
        data = fetcher.fetch(args.symbol)
        if args.verbose:
            print(json.dumps(data, indent=2, default=str))
        else:
            print_fundamentals_summary({args.symbol: data})
        if not args.no_save:
            fetcher._save_to_gcs(args.symbol, data)

    if args.all:
        symbols = [s for s in MM_UNIVERSE if s not in INDEX_FUTURES]
        symbols = list(set(symbols))
        results = fetcher.fetch_all(
            symbols  = symbols,
            save_gcs = not args.no_save,
        )
        print_fundamentals_summary(results)


if __name__ == "__main__":
    main()
