"""
Trade log — records AI recommendations and tracks outcomes after expiry.

Storage: a single CSV file (trade_log.csv) in the project root.
Outcome resolution: polls the Kalshi API for the market result field.

Workflow:
  1. User clicks "Log Recommendation" in the web app.
  2. log_recommendation() appends a row with resolved=False.
  3. On each refresh, check_and_mark_outcomes() polls Kalshi for unresolved
     markets and fills in resolved_yes and ai_correct when the result arrives.
  4. build_history_context() formats the last N completed trades for the LLM.
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "trade_log.csv"

COLUMNS = [
    "id", "logged_at", "ticker", "floor_strike", "close_time",
    "kalshi_price_c", "fair_prob", "edge", "ai_action", "side", "log_key", "ai_confidence",
    "ai_bias", "minutes_left", "btc_price",
    "resolved", "resolved_yes", "ai_correct",
]

MIN_LOG_MINUTES_LEFT = 45.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_file() -> None:
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        return

    with open(LOG_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if all(col in fieldnames for col in COLUMNS):
            return
        rows = [_normalize_row(r) for r in reader]

    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_all() -> list[dict]:
    _ensure_file()
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return [_normalize_row(r) for r in rows]


def _write_all(rows: list[dict]) -> None:
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _parse_bool(val: str) -> bool | None:
    if val in ("True", "true", "1"):
        return True
    if val in ("False", "false", "0"):
        return False
    return None


def _normalize_row(row: dict) -> dict:
    normalized = {col: row.get(col, "") for col in COLUMNS}
    if not normalized.get("side"):
        normalized["side"] = _infer_side(normalized.get("ai_action", "")) or ""
    if not normalized.get("log_key") and normalized.get("ticker") and normalized.get("side"):
        normalized["log_key"] = _make_log_key(normalized["ticker"], normalized["side"])
    return normalized


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _infer_side(ai_action: str | None) -> str | None:
    action = (ai_action or "").upper()
    if "BUY YES" in action:
        return "YES"
    if "BUY NO" in action:
        return "NO"
    return None


def _make_log_key(ticker: str, side: str) -> str:
    return f"{ticker}|{side.upper()}"


def _is_stale_logged_row(row: dict) -> bool:
    logged_at = _parse_dt(row.get("logged_at"))
    close_time = _parse_dt(row.get("close_time"))
    return bool(logged_at and close_time and logged_at >= close_time)


def is_contract_loggable(
    close_time: str,
    minutes_left: float | None,
    min_minutes_left: float = MIN_LOG_MINUTES_LEFT,
) -> tuple[bool, str]:
    """Return whether a contract is still live enough to record a recommendation."""
    close_dt = _parse_dt(close_time)
    now = datetime.now(timezone.utc)
    if close_dt is None:
        return False, "missing or invalid close_time"
    if close_dt <= now:
        return False, "contract is already closed"
    if minutes_left is not None and minutes_left < min_minutes_left:
        return False, f"less than {min_minutes_left:g} minute left"
    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_recommendation(
    ticker: str,
    floor_strike: float | None,
    close_time: str,
    kalshi_price_c: float,
    fair_prob: float,
    edge: float,
    ai_action: str,
    ai_confidence: str,
    ai_bias: str,
    minutes_left: float | None,
    btc_price: float,
    side: str | None = None,
) -> str:
    """Append a new recommendation row. Returns the new row id."""
    _ensure_file()
    is_loggable, reason = is_contract_loggable(close_time, minutes_left)
    if not is_loggable:
        raise ValueError(f"not logging stale contract: {reason}")

    side = (side or _infer_side(ai_action) or "").upper()
    if side not in ("YES", "NO"):
        raise ValueError("not logging non-directional recommendation")

    log_key = _make_log_key(ticker, side)
    rows = _read_all()
    for existing in rows:
        if existing.get("resolved") in ("True", "true"):
            continue
        if existing.get("log_key") == log_key:
            return existing.get("id", "")

    row_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    row = {
        "id":             row_id,
        "logged_at":      datetime.now(timezone.utc).isoformat(),
        "ticker":         ticker,
        "floor_strike":   floor_strike or "",
        "close_time":     close_time,
        "kalshi_price_c": round(kalshi_price_c, 2),
        "fair_prob":      round(fair_prob, 4),
        "edge":           round(edge, 4),
        "ai_action":      ai_action,
        "side":           side,
        "log_key":        log_key,
        "ai_confidence":  ai_confidence,
        "ai_bias":        ai_bias,
        "minutes_left":   round(minutes_left, 1) if minutes_left is not None else "",
        "btc_price":      round(btc_price, 2),
        "resolved":       False,
        "resolved_yes":   "",
        "ai_correct":     "",
    }
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore").writerow(row)
    return row_id


def check_and_mark_outcomes() -> int:
    """
    Poll Kalshi for unresolved markets and fill in outcomes.
    Returns number of newly resolved rows.
    """
    from .kalshi_market import get_market

    rows = _read_all()
    updated = 0
    for row in rows:
        if row.get("resolved") in ("True", "true"):
            continue
        ticker = row.get("ticker", "")
        if not ticker:
            continue
        close_time = _parse_dt(row.get("close_time"))
        if close_time and close_time > datetime.now(timezone.utc):
            continue
        try:
            market = get_market(ticker)
            result = market.get("result")  # "yes" or "no" when resolved
            if result in ("yes", "no"):
                resolved_yes = result == "yes"
                side = row.get("side") or _infer_side(row.get("ai_action", ""))
                if side == "YES":
                    ai_correct = resolved_yes
                elif side == "NO":
                    ai_correct = not resolved_yes
                else:
                    ai_correct = None   # SKIP/HOLD — no directional bet

                row["resolved"]     = True
                row["resolved_yes"] = resolved_yes
                row["ai_correct"]   = ai_correct
                updated += 1
        except Exception:
            continue

    if updated:
        _write_all(rows)
    return updated


def get_recent_history(n: int = 15, valid_only: bool = False) -> list[dict]:
    """Return the last n rows, newest first."""
    rows = _read_all()
    if valid_only:
        rows = [r for r in rows if not _is_stale_logged_row(r)]
    return list(reversed(rows))[:n]


def accuracy_stats() -> dict:
    """Compute win rate and edge stats over all resolved directional trades."""
    rows = _read_all()
    resolved = [
        r for r in rows
        if r.get("resolved") in ("True", "true")
        and r.get("ai_correct") not in ("", None)
        and not _is_stale_logged_row(r)
    ]
    if not resolved:
        return {"total": 0, "correct": 0, "win_rate": None,
                "avg_edge_correct": None, "avg_edge_wrong": None}

    correct = [r for r in resolved if _parse_bool(r["ai_correct"]) is True]
    wrong   = [r for r in resolved if _parse_bool(r["ai_correct"]) is False]

    def avg_edge(subset: list[dict]) -> float | None:
        edges = [float(r["edge"]) for r in subset if r.get("edge")]
        return sum(edges) / len(edges) if edges else None

    return {
        "total":            len(resolved),
        "correct":          len(correct),
        "win_rate":         len(correct) / len(resolved),
        "avg_edge_correct": avg_edge(correct),
        "avg_edge_wrong":   avg_edge(wrong),
    }


def build_history_context(n: int = 10) -> str:
    """
    Format recent resolved trades for injection into the LLM prompt.
    Shows the AI its own track record so it can calibrate confidence.
    """
    rows = _read_all()
    resolved = [
        r for r in rows
        if r.get("resolved") in ("True", "true")
        and r.get("ai_correct") not in ("", None)
        and not _is_stale_logged_row(r)
    ]
    recent = list(reversed(resolved))[:n]
    if not recent:
        return ""

    stats = accuracy_stats()
    lines = [
        f"--- AI TRACK RECORD (last {len(recent)} completed trades) ---",
        f"Win rate: {stats['win_rate']:.0%} ({stats['correct']}/{stats['total']})  |  "
        f"Avg edge correct: {stats['avg_edge_correct']:+.1%}  |  "
        f"Avg edge wrong: {stats['avg_edge_wrong']:+.1%}"
        if stats["avg_edge_correct"] is not None
        else f"Win rate: {stats['win_rate']:.0%} ({stats['correct']}/{stats['total']})",
    ]
    lines.append(f"{'Date':>16}  {'Action':<10}  {'Edge':>6}  {'Min':>4}  {'Result':>8}  Correct")
    for r in recent:
        try:
            logged = _parse_dt(r.get("logged_at", "")).astimezone(_ET).strftime("%m-%d %H:%M ET")
        except Exception:
            logged = r.get("logged_at", "")[:16].replace("T", " ")
        action = r.get("ai_action", "")[:9]
        edge   = f"{float(r['edge']):+.1%}" if r.get("edge") else "N/A"
        mins   = r.get("minutes_left", "?")
        result = "YES" if _parse_bool(r.get("resolved_yes")) else "NO"
        correct = "YES" if _parse_bool(r.get("ai_correct")) else "NO"
        lines.append(f"  {logged:>16}  {action:<10}  {edge:>6}  {mins:>4}  {result:>8}  {correct}")
    lines.append("--- End track record ---")
    lines.append(
        "Use this history to calibrate your confidence. "
        "If win rate is low, be more conservative. "
        "If wrong trades had high edge, be skeptical of high-edge calls."
    )
    return "\n".join(lines)
