"""
Stock price movement probability engine.

Uses log-normal return distributions (analogous to Poisson goal distributions
in the soccer model).  The key insight: under geometric Brownian motion,
log(S_T / S_0) ~ Normal(mu_T, sigma_T^2), so all probabilities reduce to
standard normal CDF evaluations.

Markets priced:
  - Directional:   P(S_T > target),  P(S_T < target)
  - Percent move:  P(return > r%),   P(return < r%)
  - Volatility:    implied fair value of a straddle / strangle
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from scipy.stats import norm  # type: ignore

from .market_state import AdjustedParams, ModelParams, StockState, adjusted_params


# ---------------------------------------------------------------------------
# Core log-normal primitives  (mirrors poisson_pmf / truncation_point)
# ---------------------------------------------------------------------------

def prob_above(S0: float, target: float, ap: AdjustedParams) -> float:
    """P(S_T > target) under log-normal GBM."""
    if target <= 0:
        return 1.0
    x = (math.log(target / S0) - ap.mu_T) / ap.sigma_T
    return float(norm.sf(x))          # survival function = 1 - CDF


def prob_below(S0: float, target: float, ap: AdjustedParams) -> float:
    """P(S_T < target) under log-normal GBM."""
    return 1.0 - prob_above(S0, target, ap)


def prob_in_range(
    S0: float,
    low: float,
    high: float,
    ap: AdjustedParams,
) -> float:
    """P(low < S_T < high)."""
    return prob_above(S0, low, ap) - prob_above(S0, high, ap)


def expected_price(S0: float, ap: AdjustedParams) -> float:
    """E[S_T] = S0 * exp(mu * T)  (arithmetic mean, not log-mean)."""
    return S0 * math.exp(ap.mu * ap.T)


def lognormal_quantile(S0: float, q: float, ap: AdjustedParams) -> float:
    """Price level at the q-th percentile of the distribution."""
    z = norm.ppf(q)
    return S0 * math.exp(ap.mu_T + ap.sigma_T * z)


# ---------------------------------------------------------------------------
# Market probabilities (mirrors MarketProbs in soccer model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketProbs:
    """All fair probabilities for a given stock state."""

    ticker: str
    current_price: float
    expected_price: float

    # Directional at specific price targets
    above: dict[float, float]    # target -> P(S_T > target)
    below: dict[float, float]    # target -> P(S_T < target)

    # Return-based (e.g., +5%, -5%)
    return_above: dict[float, float]   # pct_return -> P(return > r)
    return_below: dict[float, float]

    # Percentiles for context
    p10: float   # 10th percentile price
    p25: float
    p50: float   # median
    p75: float
    p90: float

    def as_cents(self) -> dict[str, float]:
        """Convert probabilities to cents (0-100), matching the soccer model."""
        result: dict[str, float] = {}
        for t, p in self.above.items():
            result[f"above_{t:.2f}"] = 100 * p
        for t, p in self.below.items():
            result[f"below_{t:.2f}"] = 100 * p
        for r, p in self.return_above.items():
            label = f"return_above_{'+' if r >= 0 else ''}{r:.1%}"
            result[label] = 100 * p
        for r, p in self.return_below.items():
            label = f"return_below_{'+' if r >= 0 else ''}{r:.1%}"
            result[label] = 100 * p
        return result

    def summary(self) -> str:
        lines = [
            f"Ticker : {self.ticker}",
            f"Current: ${self.current_price:.2f}",
            f"E[S_T] : ${self.expected_price:.2f}",
            f"10/25/50/75/90 percentiles: "
            f"${self.p10:.2f} / ${self.p25:.2f} / ${self.p50:.2f} / "
            f"${self.p75:.2f} / ${self.p90:.2f}",
        ]
        return "\n".join(lines)


def compute_probabilities(
    state: StockState,
    price_targets: Sequence[float] | None = None,
    return_targets: Sequence[float] = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
    params: ModelParams = ModelParams(),
) -> MarketProbs:
    """
    Compute fair probabilities for all standard stock markets.

    price_targets: specific $ levels to price (defaults to ±5%, ±10%, ±20% from spot)
    return_targets: return thresholds like +5% / -5%
    """
    ap = adjusted_params(state, params)
    S0 = state.current_price

    if price_targets is None:
        price_targets = [
            round(S0 * (1 + r), 4)
            for r in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
        ]

    above = {t: prob_above(S0, t, ap) for t in price_targets}
    below = {t: prob_below(S0, t, ap) for t in price_targets}

    return_above = {
        r: prob_above(S0, S0 * (1 + r), ap) for r in return_targets
    }
    return_below = {
        r: prob_below(S0, S0 * (1 + r), ap) for r in return_targets
    }

    return MarketProbs(
        ticker=state.ticker,
        current_price=S0,
        expected_price=expected_price(S0, ap),
        above=above,
        below=below,
        return_above=return_above,
        return_below=return_below,
        p10=lognormal_quantile(S0, 0.10, ap),
        p25=lognormal_quantile(S0, 0.25, ap),
        p50=lognormal_quantile(S0, 0.50, ap),
        p75=lognormal_quantile(S0, 0.75, ap),
        p90=lognormal_quantile(S0, 0.90, ap),
    )
