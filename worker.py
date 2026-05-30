"""
Headless signal worker — runs every 5 minutes via cron.

Does everything the web app's auto-log section does, but without a browser:
  1. Fetch live BTC price
  2. Compute vol / drift / tail dof from 1-hour candles
  3. Load KXBTCD contracts for the next eligible hour (≥45 min left)
  4. Evaluate with the quant model (same thresholds as web app)
  5. Log the highest-edge signal that passes all filters
  6. Resolve any expired trades in the log

Run manually:   python worker.py
Cron (every 5 min): */5 * * * * /home/ubuntu/BTC-Binary/.venv/bin/python /home/ubuntu/BTC-Binary/worker.py >> /home/ubuntu/BTC-Binary/worker.log 2>&1
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Logging — stdout so cron captures it in worker.log
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s ET  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = lambda *_: datetime.now(ZoneInfo("America/New_York")).timetuple()
log = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Constants — keep in sync with web_app.py
# ---------------------------------------------------------------------------

SYMBOL      = "BTC"
HORIZON     = 60        # minutes
VOL_WINDOW  = 60        # 1-hour candles
BANKROLL    = 1000
KELLY       = 0.25
MAX_POS_PCT = 0.10
MIN_EDGE    = 0.15      # must match web_app TradeParams min_edge_pct
MIN_MINUTES = 45.0      # must match trade_log.MIN_LOG_MINUTES_LEFT

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Step 1 — Live BTC price
# ---------------------------------------------------------------------------

def fetch_price(symbol: str) -> float:
    try:
        from stock_agent.robinhood_crypto import RobinhoodCryptoClient, fetch as rh_fetch
        from stock_agent.market_state import Horizon
        client = RobinhoodCryptoClient()
        data = rh_fetch(symbol, horizon=Horizon.DAY, client=client)
        log.info(f"Price via Robinhood: ${data.current_price:,.2f}")
        return data.current_price
    except Exception as e:
        log.warning(f"Robinhood failed ({e}), falling back to yfinance")
        import yfinance as yf
        df = yf.download(f"{symbol}-USD", period="1d", interval="1m",
                         auto_adjust=True, progress=False)
        close = df["Close"].squeeze().dropna()
        price = float(close.iloc[-1])
        log.info(f"Price via yfinance: ${price:,.2f}")
        return price


# ---------------------------------------------------------------------------
# Step 2 — Vol / drift / tail dof
# ---------------------------------------------------------------------------

def fetch_vol_drift(symbol: str) -> tuple[float, float, float]:
    from stock_agent.prediction_market import (
        blended_vol_annual, estimate_chart_signal,
        estimate_tail_dof, fetch_intraday,
    )
    df = fetch_intraday(symbol, lookback_hours=720)
    annual_vol = blended_vol_annual(df, window=VOL_WINDOW).annual_vol
    annual_drift, _ = estimate_chart_signal(df)
    tail_dof = estimate_tail_dof(df)
    log.info(f"Vol={annual_vol:.1%}  drift={annual_drift:+.1%}  dof={tail_dof:.1f}")
    return annual_vol, annual_drift, tail_dof


# ---------------------------------------------------------------------------
# Step 3 — KXBTCD contracts (try next hour, then hour+2 if too close to expiry)
# ---------------------------------------------------------------------------

def load_contracts(current_price: float) -> list[dict]:
    from stock_agent.kalshi_market import find_kxbtcd_atm_markets

    now_et = datetime.now(ET)
    candidates: list[dict] = []

    for offset in (1, 2):
        hour_et = (now_et.hour + offset) % 24
        try:
            markets = find_kxbtcd_atm_markets(current_price, hour_et=hour_et)
        except Exception as e:
            log.warning(f"Could not load hour+{offset} markets: {e}")
            continue

        live = [m for m in markets
                if m.get("minutes_left") is not None
                and float(m["minutes_left"]) >= MIN_MINUTES
                and m.get("display_price_cents") is not None]
        candidates.extend(live)
        if live:
            log.info(f"Loaded {len(live)} live contracts for hour+{offset} ({hour_et:02d}:00 ET)")
            break   # first hour with eligible contracts is enough

    return candidates


# ---------------------------------------------------------------------------
# Step 4 — Evaluate contracts with quant model
# ---------------------------------------------------------------------------

def evaluate_contracts(
    markets: list[dict],
    current_price: float,
    annual_vol: float,
    annual_drift: float,
    tail_dof: float,
) -> list[dict]:
    from stock_agent.prediction_market import evaluate_range_contract
    from stock_agent.trading import TradeParams, Signal

    tp = TradeParams(
        bankroll=BANKROLL,
        kelly_fraction=KELLY,
        max_position_pct=MAX_POS_PCT,
        min_edge_pct=MIN_EDGE,
        transaction_cost_pct=0.02,
    )

    evaluated = []
    for m in markets:
        try:
            price_c = float(m["display_price_cents"])
            d = evaluate_range_contract(
                SYMBOL,
                m.get("floor_strike"),
                m.get("cap_strike"),
                price_c,
                current_price,
                annual_vol,
                HORIZON,
                tp,
                annual_drift,
                tail_dof,
            )
            action = (
                "BUY YES" if d.signal == Signal.BUY else
                "BUY NO"  if d.signal == Signal.SELL else
                "HOLD"
            )
            evaluated.append({
                "ticker":       m["ticker"],
                "floor":        m.get("floor_strike"),
                "close_time":   m.get("close_time", ""),
                "minutes_left": m.get("minutes_left"),
                "kalshi_c":     price_c,
                "fair_pct":     d.fair_prob,
                "edge_pct":     d.net_edge,
                "action":       action,
                "side":         "YES" if action == "BUY YES" else ("NO" if action == "BUY NO" else None),
            })
        except Exception as e:
            log.debug(f"Skipped {m.get('ticker')}: {e}")

    actionable = [r for r in evaluated if r["action"] in ("BUY YES", "BUY NO")]
    log.info(f"Evaluated {len(evaluated)} contracts, {len(actionable)} actionable")
    return evaluated


# ---------------------------------------------------------------------------
# Step 5 — Log the best signal
# ---------------------------------------------------------------------------

def log_best_signal(evaluated: list[dict]) -> None:
    from stock_agent.trade_log import is_contract_loggable, log_recommendation

    actionable = sorted(
        [r for r in evaluated if r["action"] in ("BUY YES", "BUY NO")],
        key=lambda r: r["edge_pct"],
        reverse=True,
    )

    if not actionable:
        log.info("No actionable signal — nothing logged")
        return

    best = actionable[0]
    ml = best.get("minutes_left")
    try:
        ml = float(ml) if ml not in (None, "") else None
    except (TypeError, ValueError):
        ml = None

    ok, reason = is_contract_loggable(best["close_time"], ml)
    if not ok:
        log.info(f"Best contract not loggable: {reason}  ({best['ticker']})")
        return

    try:
        row_id = log_recommendation(
            ticker=best["ticker"],
            floor_strike=best.get("floor"),
            close_time=best["close_time"],
            kalshi_price_c=best["kalshi_c"],
            fair_prob=best["fair_pct"],
            edge=best["edge_pct"],
            ai_action=best["action"],
            ai_confidence="worker",
            ai_bias="quant",
            minutes_left=ml,
            btc_price=_current_price_global,
            side=best["side"],
        )
        log.info(
            f"Logged {row_id}  {best['action']} {best['ticker']}"
            f"  edge={best['edge_pct']:+.1%}  min_left={ml}"
        )
    except ValueError as e:
        log.info(f"Already logged or skipped: {e}")


# ---------------------------------------------------------------------------
# Step 6 — Resolve expired trades
# ---------------------------------------------------------------------------

def resolve_outcomes() -> None:
    from stock_agent.trade_log import check_and_mark_outcomes
    updated = check_and_mark_outcomes()
    if updated:
        log.info(f"Resolved {updated} trade(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_current_price_global: float = 0.0

def main() -> None:
    global _current_price_global
    log.info("=== worker start ===")

    try:
        resolve_outcomes()
    except Exception as e:
        log.warning(f"resolve_outcomes failed: {e}")

    try:
        price = fetch_price(SYMBOL)
        _current_price_global = price
    except Exception as e:
        log.error(f"Price fetch failed, aborting: {e}")
        return

    try:
        annual_vol, annual_drift, tail_dof = fetch_vol_drift(SYMBOL)
    except Exception as e:
        log.error(f"Vol/drift fetch failed, aborting: {e}")
        return

    try:
        markets = load_contracts(price)
    except Exception as e:
        log.error(f"Contract load failed: {e}")
        return

    if not markets:
        log.info("No contracts with ≥45 min left — nothing to evaluate")
        log.info("=== worker done ===")
        return

    try:
        evaluated = evaluate_contracts(markets, price, annual_vol, annual_drift, tail_dof)
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        return

    try:
        log_best_signal(evaluated)
    except Exception as e:
        log.error(f"Logging failed: {e}")

    log.info("=== worker done ===")


if __name__ == "__main__":
    main()
