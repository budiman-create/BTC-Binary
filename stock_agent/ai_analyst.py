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
    contract_action: str      # specific contract recommendation when table is provided
    raw_response: str

    def print_report(self) -> None:
        print(f"\n{'='*70}")
        print(f"  AI ANALYST REPORT - {self.ticker}")
        print(f"{'='*70}")
        print(f"  Fundamental: {self.fundamental_summary}")
        print(f"  Macro:       {self.macro_summary}")
        print(f"  Bias:        {self.drift_bias}  (drift nudge: {self.drift_nudge:+.1%} p.a.)")
        print(f"  Confidence:  {self.confidence}")
        if self.contract_action:
            print(f"  Action:      {self.contract_action}")
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
        contract_action=get("ACTION", ""),
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


def _time_urgency(minutes_left: float) -> str:
    if minutes_left <= 10:
        return "FINAL MINUTES — price is essentially locked in; only a sudden spike/dump changes outcome"
    if minutes_left <= 20:
        return "VERY SHORT — momentum and current price position dominate; drift is irrelevant"
    if minutes_left <= 35:
        return "SHORT — price action and vol matter most; light macro influence"
    if minutes_left <= 50:
        return "MODERATE — balanced mix of momentum, vol, and sentiment"
    return "STANDARD — full drift + vol + sentiment all relevant"


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
    minutes_left: float | None,
    price_1h_ago: float | None,
    price_2h_ago: float | None,
    extra_context: str = "",
    contracts_context: str = "",
    history_context: str = "",
) -> str:
    import math

    funding_bias = (
        "overcrowded long (bearish)" if funding_rate > 0.0002
        else "overcrowded short (bullish)" if funding_rate < 0
        else "neutral"
    )

    # Time-on-clock block
    effective_min = minutes_left if minutes_left is not None else float(horizon_minutes)
    urgency = _time_urgency(effective_min)
    # sigma scaled to actual time remaining
    minutes_per_year = 525_600
    sigma_to_expiry = annual_vol * math.sqrt(effective_min / minutes_per_year)

    time_block = (
        f"\nTIME ON CLOCK:\n"
        f"  Minutes to expiry : {effective_min:.0f} min\n"
        f"  Urgency regime    : {urgency}\n"
        f"  Sigma to expiry   : {sigma_to_expiry:.3%}  "
        f"(1-sigma move = ${current_price * sigma_to_expiry:,.0f})\n"
    )

    # Price action block
    pa_lines = ["\nPRICE ACTION (1h candles):"]
    pa_lines.append(f"  Now       : ${current_price:,.2f}")
    if price_1h_ago:
        chg_1h = (current_price - price_1h_ago) / price_1h_ago
        direction = "RISING" if chg_1h > 0.001 else ("FALLING" if chg_1h < -0.001 else "FLAT")
        pa_lines.append(f"  1h ago    : ${price_1h_ago:,.2f}  ({chg_1h:+.3%} -> {direction})")
    if price_2h_ago and price_1h_ago:
        chg_prev = (price_1h_ago - price_2h_ago) / price_2h_ago
        accel = chg_1h - chg_prev  # type: ignore[possibly-unbound]
        accel_str = "ACCELERATING" if accel > 0.001 else ("DECELERATING" if accel < -0.001 else "STEADY")
        pa_lines.append(f"  2h ago    : ${price_2h_ago:,.2f}  (prev hour {chg_prev:+.3%} -> momentum {accel_str})")
    price_action_block = "\n".join(pa_lines)

    history_block = f"\n{history_context}\n" if history_context else ""
    contract_block = ""
    action_field = ""
    if contracts_context:
        contract_block = f"\n{contracts_context}\n"
        action_field = (
            "\nACTION: <given the TIME REMAINING and PRICE ACTION DIRECTION, pick the single best "
            "contract or say SKIP. State whether BTC is heading toward or away from the strike. "
            "Format: 'BUY YES $X — BTC heading [toward/away], N min left, edge holds because ...' "
            "or 'SKIP — reason'>"
        )

    return f"""You are a senior crypto analyst specialising in short-term BTC binary contracts.
Your analysis must be tightly focused on the 1-HOUR window. Ignore multi-day fundamentals.
Price action and time remaining are the PRIMARY inputs. Drift and macro are secondary.

ASSET: {symbol}
Current price  : ${current_price:,.2f}
Annualised vol : {annual_vol:.1%}
Blended drift  : {annual_drift:+.2f} annualised
EMA cross      : {ema_cross}  |  Price position: {price_pos}  |  Vol factor: {vol_factor:.2f}x
Funding rate   : {funding_rate*100:.4f}%  ->  {funding_bias}
Tail dof       : {tail_dof:.1f}  ({'fat tails' if tail_dof < 15 else 'near-normal'})
{time_block}{price_action_block}

{history_block}{extra_context}{contract_block}
Return your answer in EXACTLY this format (no extra text before or after):

FUNDAMENTAL: <2-3 sentences focused on current price action and momentum within the 1h window>
MACRO: <1 sentence on sentiment/news relevant to the next hour specifically>
CATALYSTS: <bullet 1> | <bullet 2> | <bullet 3>
RISKS: <bullet 1> | <bullet 2> | <bullet 3>
BIAS: <one of: strongly_bullish / bullish / neutral / bearish / strongly_bearish>
CONFIDENCE: <one of: high / medium / low>{action_field}
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
    minutes_left: float | None = None,
    price_1h_ago: float | None = None,
    price_2h_ago: float | None = None,
    contracts_context: str = "",
    history_context: str = "",
    groq_api_key: str | None = None,
    news_api_key: str | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> tuple[AnalystReport, dict]:
    """
    Analyse a BTC prediction market contract using live quant signals + news context.

    minutes_left    : actual minutes to contract expiry (overrides horizon_minutes for urgency)
    price_1h_ago    : BTC close price 1 hour ago (from intraday candles)
    price_2h_ago    : BTC close price 2 hours ago
    history_context : formatted track-record string from trade_log.build_history_context()
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
        tail_dof, horizon_minutes, minutes_left,
        price_1h_ago, price_2h_ago,
        extra_context, contracts_context, history_context,
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
