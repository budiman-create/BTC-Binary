"""
Market data fetcher using yfinance.

Returns everything needed to build a StockState and run technical_analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from .market_state import Horizon, ModelParams, StockState
from .technical_analysis import technical_summary


# ---------------------------------------------------------------------------
# Raw market data bundle
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    ticker: str
    current_price: float
    history: pd.DataFrame      # OHLCV, daily, at least 252 rows when available
    info: dict                  # yfinance .info dict (P/E, sector, etc.)

    @property
    def sector(self) -> str:
        return self.info.get("sector", "Unknown")

    @property
    def pe_ratio(self) -> float | None:
        return self.info.get("trailingPE")

    @property
    def market_cap(self) -> float | None:
        return self.info.get("marketCap")

    @property
    def analyst_target(self) -> float | None:
        return self.info.get("targetMeanPrice")

    def extra_context(self) -> str:
        """Human-readable context string for the AI analyst prompt."""
        parts = []
        if self.sector:
            parts.append(f"Sector: {self.sector}")
        if self.pe_ratio:
            parts.append(f"P/E: {self.pe_ratio:.1f}x")
        if self.market_cap:
            cap_b = self.market_cap / 1e9
            parts.append(f"Market cap: ${cap_b:.1f}B")
        if self.analyst_target:
            parts.append(f"Analyst consensus target: ${self.analyst_target:.2f}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Fetch function
# ---------------------------------------------------------------------------

def fetch(ticker: str, period: str = "2y") -> MarketData:
    """
    Download up to 2 years of daily OHLCV data and the .info dict.
    `period` is passed directly to yfinance (e.g. "1y", "2y", "max").
    """
    t = yf.Ticker(ticker)
    history = t.history(period=period, auto_adjust=True)

    if history.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    current_price = float(history["Close"].iloc[-1])

    try:
        info = t.info
    except Exception:
        info = {}

    return MarketData(
        ticker=ticker.upper(),
        current_price=current_price,
        history=history,
        info=info,
    )


# ---------------------------------------------------------------------------
# Build a StockState from fetched data
# ---------------------------------------------------------------------------

def build_state(
    data: MarketData,
    horizon: Horizon = Horizon.MONTH,
    analyst_drift_nudge: float = 0.0,
) -> StockState:
    """
    Derive StockState from market data + technical analysis.

    analyst_drift_nudge: annualised drift adjustment from AnalystReport.drift_nudge,
    applied on top of the historically estimated drift.
    """
    tech = technical_summary(data.history, data.current_price)

    return StockState(
        ticker=data.ticker,
        current_price=data.current_price,
        annual_drift=tech["annual_drift"] + analyst_drift_nudge,
        annual_volatility=tech["annual_volatility"],
        trend_regime=tech["trend_regime"],
        vol_regime=tech["vol_regime"],
        horizon=horizon,
        rsi=tech["rsi"],
        pct_from_200ma=tech["pct_from_200ma"],
    )


# ---------------------------------------------------------------------------
# Derive market-implied probabilities from analyst consensus target
# ---------------------------------------------------------------------------

def implied_probs_from_consensus(
    data: MarketData,
    return_targets: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
) -> dict[str, float] | None:
    """
    If an analyst consensus price target is available, back out implied
    probabilities for reaching each return level.  Uses a naïve assumption
    that the consensus target represents the market's median expectation.

    Returns None if no target is available.
    """
    target = data.analyst_target
    if not target:
        return None

    S0 = data.current_price
    implied_return = (target - S0) / S0

    snapshot: dict[str, float] = {}
    for r in return_targets:
        # Crude: if implied_return > threshold, assume 60% prob of exceeding it,
        # else 40%.  A proper implementation would use options implied vols.
        prob = 0.60 if implied_return > r else 0.40
        label = f"return_above_{'+' if r >= 0 else ''}{r:.1%}"
        snapshot[label] = prob

    return snapshot
