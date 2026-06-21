"""
Headless signal worker — called every 5 minutes by run_bot.py (or cron).

Does everything the web app's auto-log section does, headlessly:
  1. Fetch live BTC price
  2. Compute vol / drift / tail dof + funding-rate blend (matches web app)
  3. Build probability calibration from intraday history
  4. Load KXBTCD contracts for the next eligible hour
  5. Evaluate with the quant model (same thresholds as web app)
  6. Ask the AI analyst (Groq) — must agree with quant side to log
  7. Log the highest-edge approved signal
  8. Resolve any expired trades in the log

Run manually:   python worker.py
Run 24/7:       python run_bot.py
Cron (every 5 min): */5 * * * * /path/to/.venv/bin/python /path/to/worker.py >> worker.log 2>&1
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
MIN_EDGE    = 0.08      # must match web_app TradeParams min_edge_pct

CHART_DRIFT_WEIGHT   = 0.75   # matches web_app: annual_drift * 0.75 + funding_drift * 0.25
FUNDING_DRIFT_WEIGHT = 0.25


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MIN_MINUTES = _env_float("MIN_LOG_MINUTES_LEFT", 20.0)

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
# Step 2 — Market data: vol, drift, tail dof, chart signal, intraday df
# ---------------------------------------------------------------------------

def fetch_market_data(symbol: str):
    """
    Returns (df, annual_vol, annual_drift, tail_dof, signal_details).
    annual_drift is the chart-only signal — blend with funding drift in main().
    """
    import pandas as pd
    from stock_agent.prediction_market import (
        blended_vol_annual, estimate_chart_signal,
        estimate_tail_dof, fetch_intraday,
    )
    df = fetch_intraday(symbol, lookback_hours=720)
    vol_est = blended_vol_annual(df, window=VOL_WINDOW)
    annual_drift, signal_details = estimate_chart_signal(df)
    tail_dof = estimate_tail_dof(df)
    log.info(
        f"Vol={vol_est.annual_vol:.1%} ({vol_est.source})"
        f"  chart_drift={annual_drift:+.1f}  dof={tail_dof:.1f}"
        f"  ema={signal_details['ema_cross']}  pos={signal_details['price_pos']}"
        f"  vol_factor={signal_details['vol_factor']:.2f}x"
    )
    return df, vol_est.annual_vol, annual_drift, tail_dof, signal_details


# ---------------------------------------------------------------------------
# Step 2b — Funding rate (Binance perp) — contrarian drift component
# ---------------------------------------------------------------------------

def fetch_funding(symbol: str) -> tuple[float, float]:
    """Returns (raw_rate, annualized funding drift nudge)."""
    from stock_agent.prediction_market import fetch_funding_rate, funding_rate_to_drift
    try:
        rate, status = fetch_funding_rate(symbol)
        drift = funding_rate_to_drift(rate)
        log.info(f"Funding rate={rate*100:.4f}%  funding_drift={drift:+.1f}  ({status})")
        return rate, drift
    except Exception as e:
        log.warning(f"Funding rate unavailable: {e}")
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Step 3 — KXBTCD contracts (try next hour, then hour+2, hour+3)
# ---------------------------------------------------------------------------

def load_contracts(current_price: float) -> list[dict]:
    from stock_agent.kalshi_market import find_kxbtcd_atm_markets

    now_et = datetime.now(ET)
    candidates: list[dict] = []

    for offset in (1, 2, 3):
        hour_et = (now_et.hour + offset) % 24
        try:
            markets = find_kxbtcd_atm_markets(current_price, hour_et=hour_et)
        except Exception as e:
            log.warning(f"Could not load hour+{offset} markets: {e}")
            continue

        quoted = [m for m in markets if m.get("display_price_cents") is not None]
        live = [
            m for m in quoted
            if m.get("minutes_left") is not None
            and float(m["minutes_left"]) >= MIN_MINUTES
        ]
        if markets:
            minute_values = [
                float(m["minutes_left"]) for m in markets
                if m.get("minutes_left") is not None
            ]
            min_left = min(minute_values) if minute_values else None
            max_left = max(minute_values) if minute_values else None
            time_text = (
                f"{min_left:.1f}-{max_left:.1f} min"
                if min_left is not None and max_left is not None
                else "unknown minutes"
            )
            log.info(
                f"hour+{offset} {hour_et:02d}:00 ET: "
                f"{len(markets)} markets, {len(quoted)} quoted, "
                f"{len(live)} eligible >= {MIN_MINUTES:g} min ({time_text})"
            )
        else:
            log.info(f"hour+{offset} {hour_et:02d}:00 ET: no markets found")

        candidates.extend(live)
        if live:
            log.info(f"Loaded {len(live)} live contracts for hour+{offset} ({hour_et:02d}:00 ET)")
            break

    return candidates


# ---------------------------------------------------------------------------
# Step 4 — Evaluate contracts with quant model + calibration
# ---------------------------------------------------------------------------

def evaluate_contracts(
    markets: list[dict],
    current_price: float,
    annual_vol: float,
    annual_drift: float,
    tail_dof: float,
    calibration=None,
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
                calibration,
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
    if evaluated:
        best = max(evaluated, key=lambda r: r["edge_pct"])
        log.info(
            f"Best evaluated: {best['action']} {best['ticker']} "
            f"edge={best['edge_pct']:+.1%} fair={best['fair_pct']:.1%} "
            f"kalshi={best['kalshi_c']:.1f}c min_left={best['minutes_left']}"
        )
    return evaluated


# ---------------------------------------------------------------------------
# Step 5 — AI analyst gate (Groq)
# ---------------------------------------------------------------------------

def _ai_trade_side(contract_action: str | None) -> str | None:
    """Parse the AI's contract_action into YES / NO / SKIP / None."""
    action = (contract_action or "").upper()
    if action.startswith("SKIP"):
        return "SKIP"
    if "BUY YES" in action:
        return "YES"
    if "BUY NO" in action:
        return "NO"
    return None


def _build_contracts_context(evaluated: list[dict]) -> str:
    if not evaluated:
        return ""
    lines = [
        "--- KXBTCD contract table (model vs Kalshi market price) ---",
        f"{'Floor Strike':>12}  {'Kalshi':>7}  {'Fair':>7}  {'Edge':>7}  {'Min Left':>8}  Action",
    ]
    for r in evaluated:
        lines.append(
            f"  ${r.get('floor') or 0:>10,.0f}  {r['kalshi_c']:>5.1f}c"
            f"  {r['fair_pct']:>6.1%}  {r['edge_pct']:>+6.1%}"
            f"  {str(r.get('minutes_left', '?')):>8}  {r['action']}"
        )
    lines.append("--- End of contract table ---")
    return "\n".join(lines)


def run_ai_analyst(
    price: float,
    annual_vol: float,
    annual_drift: float,
    signal_details: dict,
    funding_rate: float,
    tail_dof: float,
    evaluated: list[dict],
    df,
    price_1h_ago: float | None,
    price_2h_ago: float | None,
):
    """
    Call Groq AI analyst and return (report, contracts_context).
    Returns (None, contracts_context) if the call fails.
    """
    from stock_agent.ai_analyst import analyse_btc

    contracts_context = _build_contracts_context(evaluated)

    history_context = ""
    try:
        from stock_agent.trade_log import build_history_context
        history_context = build_history_context(n=25)
    except Exception as e:
        log.warning(f"History context unavailable: {e}")

    # Use the minutes_left of the most urgent actionable contract for time urgency
    actionable = [r for r in evaluated if r["action"] in ("BUY YES", "BUY NO")]
    minutes_left: float | None = None
    if actionable:
        ml_vals = []
        for r in actionable:
            try:
                ml_vals.append(float(r["minutes_left"]))
            except (TypeError, ValueError):
                pass
        if ml_vals:
            minutes_left = min(ml_vals)

    report, _ = analyse_btc(
        symbol=SYMBOL,
        current_price=price,
        annual_vol=annual_vol,
        annual_drift=annual_drift,
        ema_cross=signal_details["ema_cross"],
        price_pos=signal_details["price_pos"],
        vol_factor=signal_details["vol_factor"],
        funding_rate=funding_rate,
        tail_dof=tail_dof,
        horizon_minutes=HORIZON,
        minutes_left=minutes_left,
        price_1h_ago=price_1h_ago,
        price_2h_ago=price_2h_ago,
        contracts_context=contracts_context,
        history_context=history_context,
    )
    log.info(
        f"AI: bias={report.drift_bias}  conf={report.confidence}"
        f"  action={report.contract_action or 'none'}"
    )
    return report, contracts_context


# ---------------------------------------------------------------------------
# Step 6 — Log the best AI-approved signal
# ---------------------------------------------------------------------------

def log_best_signal(evaluated: list[dict], price: float, ai_report=None) -> None:
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

    # AI gate: if analyst ran, it must agree with the quant's side
    if ai_report is not None:
        quant_side = best["side"]
        ai_side = _ai_trade_side(ai_report.contract_action)
        if ai_side == "SKIP":
            log.info(f"AI vetoed with SKIP — not logging  ({best['ticker']})")
            return
        if ai_side is None:
            log.info(f"AI gave no clear direction — not logging  ({best['ticker']})")
            return
        if ai_side != quant_side:
            log.info(
                f"AI disagreed: AI={ai_side} quant={quant_side} — not logging  ({best['ticker']})"
            )
            return
        log.info(f"AI approved {quant_side}  ({best['ticker']})")

    ml = best.get("minutes_left")
    try:
        ml = float(ml) if ml not in (None, "") else None
    except (TypeError, ValueError):
        ml = None

    ok, reason = is_contract_loggable(best["close_time"], ml)
    if not ok:
        log.info(f"Best contract not loggable: {reason}  ({best['ticker']})")
        return

    ai_action     = (ai_report.contract_action or best["action"]) if ai_report else best["action"]
    ai_confidence = ai_report.confidence if ai_report else "worker"
    ai_bias       = ai_report.drift_bias if ai_report else "quant"

    try:
        row_id = log_recommendation(
            ticker=best["ticker"],
            floor_strike=best.get("floor"),
            close_time=best["close_time"],
            kalshi_price_c=best["kalshi_c"],
            fair_prob=best["fair_pct"],
            edge=best["edge_pct"],
            ai_action=ai_action,
            ai_confidence=ai_confidence,
            ai_bias=ai_bias,
            minutes_left=ml,
            btc_price=price,
            side=best["side"],
        )
        log.info(
            f"Logged {row_id}  {best['action']} {best['ticker']}"
            f"  edge={best['edge_pct']:+.1%}  min_left={ml}"
        )
    except ValueError as e:
        log.info(f"Already logged or skipped: {e}")


# ---------------------------------------------------------------------------
# Step 7 — Resolve expired trades
# ---------------------------------------------------------------------------

def resolve_outcomes() -> None:
    from stock_agent.trade_log import check_and_mark_outcomes
    updated = check_and_mark_outcomes()
    if updated:
        log.info(f"Resolved {updated} trade(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== worker start ===")

    try:
        resolve_outcomes()
    except Exception as e:
        log.warning(f"resolve_outcomes failed: {e}")

    # --- Price ---
    try:
        price = fetch_price(SYMBOL)
    except Exception as e:
        log.error(f"Price fetch failed, aborting: {e}")
        return

    # --- Market data (df, vol, chart drift, tail dof, signal details) ---
    try:
        df, annual_vol, annual_drift_chart, tail_dof, signal_details = fetch_market_data(SYMBOL)
    except Exception as e:
        log.error(f"Market data fetch failed, aborting: {e}")
        return

    # --- Funding rate (contrarian drift nudge) ---
    funding_rate, funding_drift = 0.0, 0.0
    try:
        funding_rate, funding_drift = fetch_funding(SYMBOL)
    except Exception as e:
        log.warning(f"Funding fetch failed (using 0): {e}")

    # Blend chart + funding drift — identical to web_app.py
    annual_drift = annual_drift_chart * CHART_DRIFT_WEIGHT + funding_drift * FUNDING_DRIFT_WEIGHT
    log.info(
        f"Blended drift={annual_drift:+.2f}"
        f" (chart={annual_drift_chart:+.2f}×{CHART_DRIFT_WEIGHT}"
        f" + funding={funding_drift:+.2f}×{FUNDING_DRIFT_WEIGHT})"
    )

    # --- Probability calibration (optional — skip if it fails) ---
    calibration = None
    try:
        from stock_agent.prediction_market import build_probability_calibration
        calibration = build_probability_calibration(df, horizon_minutes=HORIZON, vol_window=VOL_WINDOW)
        log.info(f"Calibration: {calibration.samples} samples  Brier {calibration.brier:.3f}")
    except Exception as e:
        log.warning(f"Calibration failed (using uncalibrated probs): {e}")

    # --- Contracts ---
    try:
        markets = load_contracts(price)
    except Exception as e:
        log.error(f"Contract load failed: {e}")
        return

    if not markets:
        log.info(f"No contracts with >= {MIN_MINUTES:g} min left — nothing to evaluate")
        log.info("=== worker done ===")
        return

    # --- Evaluate ---
    try:
        evaluated = evaluate_contracts(markets, price, annual_vol, annual_drift, tail_dof, calibration)
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        return

    # --- AI analyst gate (only if there are actionable signals to approve) ---
    ai_report = None
    actionable = [r for r in evaluated if r["action"] in ("BUY YES", "BUY NO")]
    if actionable:
        price_1h_ago = float(df["Close"].squeeze().iloc[-2]) if len(df) >= 2 else None
        price_2h_ago = float(df["Close"].squeeze().iloc[-3]) if len(df) >= 3 else None
        try:
            ai_report, _ = run_ai_analyst(
                price, annual_vol, annual_drift, signal_details,
                funding_rate, tail_dof, evaluated, df,
                price_1h_ago, price_2h_ago,
            )
        except Exception as e:
            log.warning(f"AI analyst failed (logging without veto): {e}")
    else:
        log.info("No actionable signals — skipping AI analyst call")

    # --- Log ---
    try:
        log_best_signal(evaluated, price, ai_report)
    except Exception as e:
        log.error(f"Logging failed: {e}")

    log.info("=== worker done ===")


if __name__ == "__main__":
    main()
