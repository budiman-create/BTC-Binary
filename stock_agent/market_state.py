"""
Stock market state and model parameters.

Mirrors the MatchState / ModelParams pattern from the soccer model but for equities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Trend / volatility regimes (analogous to Momentum in the soccer model)
# ---------------------------------------------------------------------------

TrendRegime = Literal[
    "strong_uptrend",
    "uptrend",
    "neutral",
    "downtrend",
    "strong_downtrend",
]

VolRegime = Literal["low_vol", "normal_vol", "high_vol", "crisis_vol"]


class Horizon(str, Enum):
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1mo"
    QUARTER = "3mo"


# ---------------------------------------------------------------------------
# Core state snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StockState:
    """
    Snapshot of a stock at a given moment.  Feed fresh values from
    data_fetcher.py before each pricing run.
    """
    ticker: str
    current_price: float
    # Annualised drift (μ) and volatility (σ) estimated from historical data
    annual_drift: float          # e.g. 0.10 for 10% expected return p.a.
    annual_volatility: float     # e.g. 0.25 for 25% realised vol p.a.

    trend_regime: TrendRegime = "neutral"
    vol_regime: VolRegime = "normal_vol"

    # Horizon over which to price contracts
    horizon: Horizon = Horizon.MONTH

    # Optional: latest RSI, distance from 200-MA, etc. — used by the model
    rsi: float = 50.0            # 0-100; >70 overbought, <30 oversold
    pct_from_200ma: float = 0.0  # +0.05 means 5% above 200-day MA

    @property
    def horizon_years(self) -> float:
        mapping = {
            Horizon.DAY: 1 / 252,
            Horizon.WEEK: 5 / 252,
            Horizon.MONTH: 21 / 252,
            Horizon.QUARTER: 63 / 252,
        }
        return mapping[self.horizon]


# ---------------------------------------------------------------------------
# Calibration knobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelParams:
    """Multipliers applied to drift / vol based on regime."""

    # Trend regime: scales the drift component
    trend_drift_strong_up: float = 1.50
    trend_drift_up: float = 1.20
    trend_drift_neutral: float = 1.00
    trend_drift_down: float = 0.80
    trend_drift_strong_down: float = 0.50

    # Volatility regime: scales the vol component
    vol_scale_low: float = 0.75
    vol_scale_normal: float = 1.00
    vol_scale_high: float = 1.40
    vol_scale_crisis: float = 2.00

    # RSI extremes add a mean-reversion nudge to drift
    rsi_overbought_threshold: float = 70.0
    rsi_oversold_threshold: float = 30.0
    rsi_overbought_drift_penalty: float = -0.05   # annualised drift adjustment
    rsi_oversold_drift_bonus: float = 0.05

    def drift_multiplier(self, regime: TrendRegime) -> float:
        return {
            "strong_uptrend": self.trend_drift_strong_up,
            "uptrend": self.trend_drift_up,
            "neutral": self.trend_drift_neutral,
            "downtrend": self.trend_drift_down,
            "strong_downtrend": self.trend_drift_strong_down,
        }[regime]

    def vol_multiplier(self, regime: VolRegime) -> float:
        return {
            "low_vol": self.vol_scale_low,
            "normal_vol": self.vol_scale_normal,
            "high_vol": self.vol_scale_high,
            "crisis_vol": self.vol_scale_crisis,
        }[regime]


# ---------------------------------------------------------------------------
# Adjusted parameters after applying regime multipliers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdjustedParams:
    """Effective drift and vol for the pricing horizon after regime adjustments."""
    mu: float       # annualised
    sigma: float    # annualised
    T: float        # horizon in years

    @property
    def mu_T(self) -> float:
        """Drift over the horizon (log-space)."""
        return (self.mu - 0.5 * self.sigma ** 2) * self.T

    @property
    def sigma_T(self) -> float:
        """Vol over the horizon."""
        return self.sigma * self.T ** 0.5


def adjusted_params(
    state: StockState,
    params: ModelParams = ModelParams(),
) -> AdjustedParams:
    """Apply regime multipliers and RSI nudge to get effective drift/vol."""
    mu = state.annual_drift * params.drift_multiplier(state.trend_regime)
    sigma = state.annual_volatility * params.vol_multiplier(state.vol_regime)

    # RSI mean-reversion nudge
    if state.rsi >= params.rsi_overbought_threshold:
        mu += params.rsi_overbought_drift_penalty
    elif state.rsi <= params.rsi_oversold_threshold:
        mu += params.rsi_oversold_drift_bonus

    return AdjustedParams(mu=mu, sigma=sigma, T=state.horizon_years)
