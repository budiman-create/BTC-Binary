"""
AI analyst layer — uses Groq (free tier) + live news context.

Free tier: 1,000 requests/day, no credit card needed.
Keys needed in .env:
    GROQ_API_KEY=gsk_...              (required — console.groq.com)
    CRYPTOPANIC_API_KEY=...           (optional — cryptopanic.com, free)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from groq import Groq  # type: ignore

from .market_state import StockState


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

DriftBias = Literal["strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"]

DRIFT_NUDGE: dict[DriftBias, float] = {
    "strongly_bullish": +0.06,
    "bullish":          +0.03,
    "neutral":           0.00,
    "bearish":          -0.03,
    "strongly_bearish": -0.06,
}


@dataclass
class AnalystReport:
    ticker: str
    fundamental_summary: str
    macro_summary: str
    key_risks: list[str]
    key_catalysts: list[str]
    drift_bias: DriftBias
    drift_nudge: float
    confidence: str
    raw_response: str

    def print_report(self) -> None:
        print(f"\n{'='*70}")
        print(f"  AI ANALYST REPORT - {self.ticker}")
        print(f"{'='*70}")
        print(f"  Fundamental: {self.fundamental_summary}")
        print(f"  Macro:       {self.macro_summary}")
        print(f"  Bias:        {self.drift_bias}  (drift nudge: {self.drift_nudge:+.1%} p.a.)")
        print(f"  Confidence:  {self.confidence}")
        if self.key_catalysts:
            print(f"  Catalysts:   " + "; ".join(self.key_catalysts))
        if self.key_risks:
            print(f"  Risks:       " + "; ".join(self.key_risks))
        print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(state: StockState, extra_context: str = "") -> str:
    return f"""You are a senior crypto/equity analyst. Analyse the asset below and return a structured assessment.

ASSET: {state.ticker}
Current price: ${state.current_price:.4f}
Horizon: {state.horizon.value}
Technical trend: {state.trend_regime}
Volatility regime: {state.vol_regime}
RSI: {state.rsi:.1f}
Distance from 200-day MA: {state.pct_from_200ma:+.1%}
{extra_context}

Return your answer in EXACTLY this format (no extra text before or after):

FUNDAMENTAL: <2-3 sentence summary of asset quality and recent trend>
MACRO: <1-2 sentence macro/sector context>
CATALYSTS: <bullet 1> | <bullet 2> | <bullet 3>
RISKS: <bullet 1> | <bullet 2> | <bullet 3>
BIAS: <one of: strongly_bullish / bullish / neutral / bearish / strongly_bearish>
CONFIDENCE: <one of: high / medium / low>
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_response(ticker: str, text: str) -> AnalystReport:
    lines = {
        k.strip(): v.strip()
        for line in text.strip().splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }

    def get(key: str, default: str = "") -> str:
        return lines.get(key.upper(), default)

    bias_raw = get("BIAS", "neutral").lower().replace(" ", "_")
    bias: DriftBias = bias_raw if bias_raw in DRIFT_NUDGE else "neutral"  # type: ignore

    return AnalystReport(
        ticker=ticker,
        fundamental_summary=get("FUNDAMENTAL"),
        macro_summary=get("MACRO"),
        key_catalysts=[c.strip() for c in get("CATALYSTS").split("|") if c.strip()],
        key_risks=[r.strip() for r in get("RISKS").split("|") if r.strip()],
        drift_bias=bias,
        drift_nudge=DRIFT_NUDGE[bias],
        confidence=get("CONFIDENCE", "medium"),
        raw_response=text,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse(
    state: StockState,
    extra_context: str = "",
    api_key: str | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> AnalystReport:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set GROQ_API_KEY in your .env file.\n"
            "Get a free key at: console.groq.com -> API Keys"
        )

    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise, data-driven crypto and equity analyst. "
                    "Follow the exact output format requested. No markdown, no preamble."
                ),
            },
            {"role": "user", "content": _build_prompt(state, extra_context)},
        ],
    )
    raw = response.choices[0].message.content
    return _parse_response(state.ticker, raw)


def _build_btc_prompt(
    symbol: str,
    current_price: float,
    annual_vol: float,
    annual_drift: float,
    ema_cross: str,
    price_pos: str,
    vol_factor: float,
    funding_rate: float,
    tail_dof: float,
    horizon_minutes: int,
    extra_context: str = "",
) -> str:
    funding_bias = (
        "overcrowded long (bearish)" if funding_rate > 0.0002
        else "overcrowded short (bullish)" if funding_rate < 0
        else "neutral"
    )
    return f"""You are a senior crypto analyst specialising in short-term BTC binary contracts.
Analyse the data below and return a structured assessment for a {horizon_minutes}-minute horizon.

ASSET: {symbol}
Current price: ${current_price:,.2f}
Horizon: {horizon_minutes} minutes
Annualised vol: {annual_vol:.1%}
Blended drift signal: {annual_drift:+.2f} annualised
EMA cross: {ema_cross}  |  Price position: {price_pos}  |  Vol factor: {vol_factor:.2f}x
Funding rate (8h): {funding_rate*100:.4f}%  →  {funding_bias}
Tail fatness (dof): {tail_dof:.1f}  ({'fat tails' if tail_dof < 15 else 'near-normal'})

{extra_context}

Return your answer in EXACTLY this format (no extra text before or after):

FUNDAMENTAL: <2-3 sentences on BTC short-term price action and trend quality>
MACRO: <1-2 sentences on macro/sentiment context from the live data above>
CATALYSTS: <bullet 1> | <bullet 2> | <bullet 3>
RISKS: <bullet 1> | <bullet 2> | <bullet 3>
BIAS: <one of: strongly_bullish / bullish / neutral / bearish / strongly_bearish>
CONFIDENCE: <one of: high / medium / low>
"""


def analyse_btc(
    symbol: str,
    current_price: float,
    annual_vol: float,
    annual_drift: float,
    ema_cross: str,
    price_pos: str,
    vol_factor: float,
    funding_rate: float,
    tail_dof: float,
    horizon_minutes: int = 60,
    groq_api_key: str | None = None,
    news_api_key: str | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> tuple[AnalystReport, dict]:
    """
    Analyse a BTC prediction market contract using live quant signals + news context.

    Fetches Fear & Greed Index and CryptoPanic headlines automatically,
    injects them into the LLM prompt, and returns (AnalystReport, raw_news_data).
    """
    from .news_context import build_extra_context

    extra_context, raw_news = build_extra_context(symbol, news_api_key=news_api_key)

    key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set GROQ_API_KEY in your .env file.\n"
            "Get a free key at: console.groq.com -> API Keys"
        )

    prompt = _build_btc_prompt(
        symbol, current_price, annual_vol, annual_drift,
        ema_cross, price_pos, vol_factor, funding_rate,
        tail_dof, horizon_minutes, extra_context,
    )

    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise, data-driven crypto analyst specialising in "
                    "short-term binary contract pricing. Follow the exact output format. "
                    "No markdown, no preamble. Be specific about what the live news implies."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content
    return _parse_response(symbol, raw), raw_news
