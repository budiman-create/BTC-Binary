"""
Kalshi public market-data adapter.

Uses unauthenticated endpoints for market metadata and order books. Trading
still stays outside this app.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import requests


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class KalshiQuote:
    ticker: str
    title: str
    strike: float | None
    floor_strike: float | None
    cap_strike: float | None
    strike_type: str | None
    yes_bid_cents: float | None
    yes_ask_cents: float | None
    yes_mid_cents: float | None
    no_bid_cents: float | None
    no_ask_cents: float | None

    @property
    def display_price_cents(self) -> float | None:
        return self.yes_mid_cents or self.yes_ask_cents or self.yes_bid_cents


def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(BASE_URL + path, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_market(ticker: str) -> dict:
    data = _get(f"/markets/{ticker.upper()}")
    return data.get("market", data)


def get_orderbook(ticker: str, depth: int = 1) -> dict:
    data = _get(f"/markets/{ticker.upper()}/orderbook", params={"depth": depth})
    return data.get("orderbook_fp") or data.get("orderbook") or {}


def get_orderbooks(tickers: list[str]) -> dict[str, dict]:
    """Fetch multiple orderbooks using Kalshi's batch orderbook endpoint."""
    if not tickers:
        return {}
    data = _get(
        "/markets/orderbooks",
        params=[("tickers", t.upper()) for t in tickers],
    )
    books: dict[str, dict] = {}
    for item in data.get("orderbooks", []):
        ticker = item.get("ticker")
        if ticker:
            books[ticker] = item.get("orderbook_fp") or item.get("orderbook") or {}
    return books


def get_markets(params: dict | None = None, max_pages: int = 5) -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    base_params = dict(params or {})

    for _ in range(max_pages):
        query = dict(base_params)
        if cursor:
            query["cursor"] = cursor
        data = _get("/markets", params=query)
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets


def get_event_markets(event_ticker: str) -> list[dict]:
    """Fetch all markets belonging to a specific Kalshi event."""
    data = _get("/markets", params={"event_ticker": event_ticker.upper(), "limit": 200})
    return data.get("markets", [])


def kxbtcd_event_ticker(dt: datetime.datetime | None = None, hour_et: int = 11) -> str:
    """
    Build a KXBTCD event ticker for a given date and hour (ET).

    Kalshi format: KXBTCD-{YY}{MON}{DD}{HH}
    e.g. today 11AM ET = KXBTCD-26MAY2511
    """
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    if dt is None:
        dt = datetime.datetime.now(_ET)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    # If the target hour has already passed today, the contract is for tomorrow.
    if hour_et < dt.hour:
        dt = dt + datetime.timedelta(days=1)
    yy  = dt.strftime("%y")
    mon = dt.strftime("%b").upper()
    dd  = dt.strftime("%d")
    return f"KXBTCD-{yy}{mon}{dd}{hour_et:02d}"


def find_kxbtcd_atm_markets(
    current_price: float,
    hour_et: int = 11,
    n_strikes: int = 12,
    dt: datetime.datetime | None = None,
) -> list[dict]:
    """
    Find the KXBTCD near-ATM contracts for a specific hour today.

    Returns up to n_strikes markets sorted closest to current_price first,
    with live orderbook prices attached.
    """
    event = kxbtcd_event_ticker(dt, hour_et)
    markets = get_event_markets(event)
    if not markets:
        return []

    # Sort by proximity to current price
    markets.sort(key=lambda m: abs((m.get("floor_strike") or 0) - current_price))
    near = markets[:n_strikes]

    quoted = attach_orderbook_quotes(near)
    for q in quoted:
        q["event_ticker"] = event
        q["floor_strike"] = _parse_float(q.get("floor_strike"))
        q["close_time"]   = q.get("close_time") or ""
        close_dt = _parse_close_time(q)
        if close_dt:
            now = datetime.datetime.now(datetime.timezone.utc)
            q["minutes_left"] = round((close_dt - now).total_seconds() / 60, 1)
    return quoted


def _parse_close_time(market: dict) -> datetime.datetime | None:
    raw = market.get("close_time") or market.get("expiration_time") or ""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S+00:00"):
        try:
            return datetime.datetime.strptime(raw, fmt).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


def find_btc_markets(
    limit: int = 25,
    max_expiry_hours: float | None = None,
    min_expiry_minutes: float = 5,
) -> list[dict]:
    """
    Return open Kalshi BTC/Bitcoin markets.

    max_expiry_hours: if set, only return markets closing within this many
                      hours (e.g. 2.0 for 1-hour trading).
    min_expiry_minutes: skip markets expiring in fewer than this many minutes
                        (already nearly expired).
    """
    markets: list[dict] = []
    seen_tickers: set[str] = set()

    # Kalshi uses several series tickers for BTC — cast a wide net
    queries = [
        {"status": "open", "series_ticker": "KXBTC",    "limit": 200},
        {"status": "open", "series_ticker": "KXBTCD",   "limit": 200},
        {"status": "open", "series_ticker": "KXBTCUSD", "limit": 200},
        {"status": "open", "series_ticker": "BTCUSD",   "limit": 200},
        {"status": "open", "series_ticker": "KXBTC1H",  "limit": 200},
        {"status": "open", "limit": 200},
    ]
    for query in queries:
        for market in get_markets(query):
            ticker = str(market.get("ticker", ""))
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            markets.append(market)

    now = datetime.datetime.now(datetime.timezone.utc)
    matches = []
    for market in markets:
        haystack = " ".join(
            str(market.get(key, ""))
            for key in ("ticker", "event_ticker", "title", "subtitle", "category")
        ).lower()
        if "btc" not in haystack and "bitcoin" not in haystack:
            continue

        close_dt = _parse_close_time(market)
        if close_dt is not None:
            minutes_left = (close_dt - now).total_seconds() / 60
            if minutes_left < min_expiry_minutes:
                continue
            if max_expiry_hours is not None and minutes_left > max_expiry_hours * 60:
                continue
        elif max_expiry_hours is not None:
            # Can't determine expiry — skip when filtering by time
            continue

        matches.append({
            "ticker":     market.get("ticker", ""),
            "title":      market.get("title") or market.get("subtitle") or "",
            "strike":     _infer_strike(market),
            "floor_strike": _parse_float(market.get("floor_strike")),
            "cap_strike":   _parse_float(market.get("cap_strike")),
            "close_time": market.get("close_time") or market.get("expiration_time") or "",
            "minutes_left": round((close_dt - now).total_seconds() / 60, 1) if close_dt else None,
        })

    # Sort by soonest expiry first — most relevant for 1-hour trading
    matches.sort(key=lambda m: (m["minutes_left"] is None, m["minutes_left"] or 0))
    return matches[:limit]


def attach_orderbook_quotes(markets: list[dict]) -> list[dict]:
    tickers = [str(m.get("ticker", "")) for m in markets if m.get("ticker")]
    books = get_orderbooks(tickers)
    quoted = []
    for market in markets:
        book = books.get(str(market.get("ticker", "")), {})
        yes_bid = _best_level_cents(book.get("yes_dollars") or book.get("yes"))
        no_bid = _best_level_cents(book.get("no_dollars") or book.get("no"))
        yes_ask = None if no_bid is None else 100.0 - no_bid
        if yes_bid is not None and yes_ask is not None:
            yes_mid = (yes_bid + yes_ask) / 2.0
        else:
            yes_mid = yes_ask or yes_bid
        quoted.append({
            **market,
            "yes_bid_cents": yes_bid,
            "yes_ask_cents": yes_ask,
            "yes_mid_cents": yes_mid,
            "display_price_cents": yes_mid,
        })
    return quoted


def _best_level_cents(levels: list | None) -> float | None:
    if not levels:
        return None
    value = max(float(level[0]) for level in levels)
    return value * 100.0 if value <= 1.0 else value


def _parse_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def _infer_strike(market: dict) -> float | None:
    for key in ("strike", "floor_strike", "cap_strike", "functional_strike"):
        parsed = _parse_float(market.get(key))
        if parsed is not None:
            return parsed

    functional = market.get("custom_strike")
    if functional is not None:
        return _parse_float(functional)
    return None


def get_quote(ticker: str) -> KalshiQuote:
    ticker = ticker.upper().strip()
    market = get_market(ticker)
    book = get_orderbook(ticker, depth=1)

    yes_bid = _best_level_cents(book.get("yes_dollars") or book.get("yes"))
    no_bid = _best_level_cents(book.get("no_dollars") or book.get("no"))

    yes_ask = None if no_bid is None else 100.0 - no_bid
    no_ask = None if yes_bid is None else 100.0 - yes_bid

    if yes_bid is not None and yes_ask is not None:
        yes_mid = (yes_bid + yes_ask) / 2.0
    else:
        yes_mid = yes_ask or yes_bid

    return KalshiQuote(
        ticker=ticker,
        title=market.get("title") or market.get("subtitle") or ticker,
        strike=_infer_strike(market),
        floor_strike=_parse_float(market.get("floor_strike")),
        cap_strike=_parse_float(market.get("cap_strike")),
        strike_type=market.get("strike_type"),
        yes_bid_cents=yes_bid,
        yes_ask_cents=yes_ask,
        yes_mid_cents=yes_mid,
        no_bid_cents=no_bid,
        no_ask_cents=no_ask,
    )
