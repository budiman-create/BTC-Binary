"""
Technical analysis layer.

Derives StockState regime fields (trend_regime, vol_regime, rsi,
pct_from_200ma) from a raw OHLCV DataFrame returned by data_fetcher.py.

Analogous to the momentum / red-card adjustments in the soccer model — these
are regime multipliers fed into adjusted_params() before pricing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .market_state import TrendRegime, VolRegime


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def compute_sma(closes: pd.Series, period: int) -> float:
    return float(closes.tail(period).mean())


def pct_from_ma(current_price: float, closes: pd.Series, period: int = 200) -> float:
    """Positive means price is above the MA."""
    if len(closes) < period:
        return 0.0
    ma = compute_sma(closes, period)
    return (current_price - ma) / ma


# ---------------------------------------------------------------------------
# Trend regime classification
# ---------------------------------------------------------------------------

def classify_trend(
    closes: pd.Series,
    current_price: float,
    short_period: int = 50,
    long_period: int = 200,
) -> "TrendRegime":
    """
    Uses a dual-MA crossover + price position:
      - above both MAs and short > long → strong_uptrend
      - above long MA → uptrend
      - below both MAs and short < long → strong_downtrend
      - below long MA → downtrend
      - else → neutral
    """
    if len(closes) < long_period:
        return "neutral"

    short_ma = compute_sma(closes, short_period)
    long_ma = compute_sma(closes, long_period)
    p = current_price

    above_long = p > long_ma
    above_short = p > short_ma
    short_above_long = short_ma > long_ma

    if above_long and above_short and short_above_long:
        return "strong_uptrend"
    if above_long:
        return "uptrend"
    if not above_long and not above_short and not short_above_long:
        return "strong_downtrend"
    if not above_long:
        return "downtrend"
    return "neutral"


# ---------------------------------------------------------------------------
# Volatility regime classification
# ---------------------------------------------------------------------------

def annualised_vol(closes: pd.Series, window: int = 21) -> float:
    """Realised vol from daily log returns, annualised."""
    if len(closes) < 2:
        return 0.20
    log_returns = np.log(closes / closes.shift(1)).dropna()
    daily_vol = float(log_returns.tail(window).std())
    return daily_vol * math.sqrt(252)


def classify_vol_regime(vol: float) -> "VolRegime":
    """
    Thresholds are rough empirical buckets for US equities.
    Adjust as needed for other asset classes.
    """
    if vol < 0.15:
        return "low_vol"
    if vol < 0.30:
        return "normal_vol"
    if vol < 0.55:
        return "high_vol"
    return "crisis_vol"


# ---------------------------------------------------------------------------
# Historical drift estimation
# ---------------------------------------------------------------------------

def estimate_annual_drift(closes: pd.Series, lookback: int = 252) -> float:
    """
    Geometric mean daily return × 252 as a simple drift proxy.
    For a more robust estimate, use a CAPM or factor model.
    """
    if len(closes) < 20:
        return 0.08  # fallback: 8% equity risk premium assumption
    log_returns = np.log(closes / closes.shift(1)).dropna().tail(lookback)
    return float(log_returns.mean() * 252)


# ---------------------------------------------------------------------------
# All-in-one summary from a DataFrame
# ---------------------------------------------------------------------------

def technical_summary(df: pd.DataFrame, current_price: float) -> dict:
    """
    Given a yfinance DataFrame with a 'Close' column, return a dict of
    all fields needed to populate a StockState.
    """
    closes = df["Close"].dropna()

    rsi = compute_rsi(closes)
    annual_vol = annualised_vol(closes)
    annual_drift = estimate_annual_drift(closes)
    trend = classify_trend(closes, current_price)
    vol_regime = classify_vol_regime(annual_vol)
    pct_200ma = pct_from_ma(current_price, closes, 200)

    return {
        "rsi": round(rsi, 2),
        "annual_volatility": round(annual_vol, 4),
        "annual_drift": round(annual_drift, 4),
        "trend_regime": trend,
        "vol_regime": vol_regime,
        "pct_from_200ma": round(pct_200ma, 4),
    }
