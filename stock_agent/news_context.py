"""
Real-time news + sentiment context for the AI analyst.

Sources (all free, no paid subscription needed):
  1. Alternative.me Fear & Greed Index — no key required
  2. CryptoPanic headlines — requires free key (CRYPTOPANIC_API_KEY in .env)
     Get one at: cryptopanic.com -> Developers -> API

The combined string is injected into the LLM prompt as extra_context so
the model can reason about things the quant signals cannot see.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests


def fetch_fear_greed(limit: int = 3) -> dict:
    """
    Fetch the Alternative.me Crypto Fear & Greed Index.
    Returns dict with 'value', 'classification', and recent history.
    No API key needed.
    """
    try:
        r = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return {}
        latest = data[0]
        return {
            "value": int(latest["value"]),
            "classification": latest["value_classification"],
            "history": [
                {"value": int(d["value"]), "label": d["value_classification"]}
                for d in data
            ],
        }
    except Exception:
        return {}


def fetch_crypto_news(
    symbol: str = "BTC",
    api_key: str | None = None,
    limit: int = 8,
) -> list[dict]:
    """
    Fetch recent CryptoPanic headlines for a symbol.
    Returns list of {title, kind, votes_positive, votes_negative, published_at}.
    Requires CRYPTOPANIC_API_KEY in .env (free at cryptopanic.com).
    """
    key = api_key or os.environ.get("CRYPTOPANIC_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://cryptopanic.com/api/free/v1/posts/",
            params={
                "auth_token": key,
                "currencies": symbol.upper(),
                "kind": "news",
                "public": "true",
            },
            timeout=6,
        )
        r.raise_for_status()
        results = r.json().get("results", [])[:limit]
        return [
            {
                "title": item.get("title", ""),
                "kind": item.get("kind", "news"),
                "votes_positive": item.get("votes", {}).get("positive", 0),
                "votes_negative": item.get("votes", {}).get("negative", 0),
                "published_at": item.get("published_at", ""),
            }
            for item in results
        ]
    except Exception:
        return []


def build_extra_context(
    symbol: str = "BTC",
    news_api_key: str | None = None,
) -> tuple[str, dict]:
    """
    Build the extra_context string to inject into the LLM prompt.
    Also returns raw data dict for display in the web app.

    Returns (context_string, raw_data).
    """
    fg = fetch_fear_greed()
    news = fetch_crypto_news(symbol, api_key=news_api_key)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [f"--- Live market context as of {now_str} ---"]

    # Fear & greed
    if fg:
        history_str = " -> ".join(
            f"{h['value']} ({h['label']})" for h in fg.get("history", [])
        )
        lines.append(
            f"Fear & Greed Index: {fg['value']} ({fg['classification']})  "
            f"[recent trend: {history_str}]"
        )
    else:
        lines.append("Fear & Greed Index: unavailable")

    # News headlines
    if news:
        lines.append(f"\nRecent {symbol} news headlines (CryptoPanic, sorted by recency):")
        for item in news:
            sentiment = ""
            if item["votes_positive"] > item["votes_negative"] + 2:
                sentiment = " [bullish sentiment]"
            elif item["votes_negative"] > item["votes_positive"] + 2:
                sentiment = " [bearish sentiment]"
            lines.append(f"  • {item['title']}{sentiment}")
    else:
        lines.append(
            "\nNo live news available (add CRYPTOPANIC_API_KEY to .env for headlines)."
        )

    lines.append("--- End of live context ---")

    raw = {"fear_greed": fg, "news": news}
    return "\n".join(lines), raw
