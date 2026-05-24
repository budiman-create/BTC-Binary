"""
AI analyst layer — uses Groq (free tier).

Free tier: 1,000 requests/day, no credit card needed.
Get a free key at: console.groq.com -> API Keys

Set in .env:
    GROQ_API_KEY=gsk_...
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
