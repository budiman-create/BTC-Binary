"""
Trading layer: edge calculation, Kelly position sizing, trade decisions.

Directly mirrors the soccer model's evaluate_contract / kelly_fraction /
TradeDecision pattern, adapted for stock directional and return contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .market_state import ModelParams, StockState
from .price_model import MarketProbs, compute_probabilities


# ---------------------------------------------------------------------------
# Signal (identical to soccer model)
# ---------------------------------------------------------------------------

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Trade parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeParams:
    bankroll: float = 10_000.0
    kelly_fraction: float = 0.25       # ¼-Kelly — conservative default
    max_position_pct: float = 0.10     # never more than 10% of bankroll in one name
    min_edge_pct: float = 0.03         # 3% net edge minimum (vs 5¢ in soccer)
    transaction_cost_pct: float = 0.005  # 0.5% round-trip slippage + commission

    @property
    def max_position(self) -> float:
        return self.bankroll * self.max_position_pct


# ---------------------------------------------------------------------------
# Trade decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeDecision:
    ticker: str
    market: str                  # e.g. "above_150.00", "return_above_+5.0%"
    fair_prob: float             # model's estimate
    market_implied_prob: float   # what the market prices in (from options or consensus)
    raw_edge: float              # fair - implied
    net_edge: float              # after transaction costs
    signal: Signal
    kelly_pct: float             # full Kelly fraction (before scaling)
    sized_dollars: float         # after ¼-Kelly and bankroll cap

    def __str__(self) -> str:
        return (
            f"{self.ticker:>6} | {self.market:<30} | "
            f"fair={self.fair_prob:6.1%}  mkt={self.market_implied_prob:6.1%}  "
            f"edge={self.net_edge:+6.1%}  "
            f"{self.signal.value:<4}  "
            f"size=${self.sized_dollars:,.0f}"
        )


# ---------------------------------------------------------------------------
# Kelly criterion for probability bets
# ---------------------------------------------------------------------------

def kelly_fraction(fair_prob: float, market_prob: float, side: Signal) -> float:
    """
    Optimal Kelly fraction for a binary probability bet.

    BUY  (bet event happens):    f* = (q - p) / (1 - p)
    SELL (bet event doesn't):    f* = ((1-q) - (1-p)) / (1-(1-p)) = (p-q)/p

    Returns 0 if the proposed side has negative edge.
    """
    if side == Signal.BUY:
        if market_prob >= 1.0:
            return 0.0
        f = (fair_prob - market_prob) / (1.0 - market_prob)
    elif side == Signal.SELL:
        if market_prob <= 0.0:
            return 0.0
        f = (market_prob - fair_prob) / market_prob
    else:
        return 0.0
    return max(0.0, f)


# ---------------------------------------------------------------------------
# Single-contract evaluation
# ---------------------------------------------------------------------------

def evaluate_contract(
    ticker: str,
    market: str,
    fair_prob: float,
    market_implied_prob: float,
    params: TradeParams = TradeParams(),
) -> TradeDecision:
    """Turn a fair-vs-implied comparison into a sized trade decision."""
    raw_edge = fair_prob - market_implied_prob

    if raw_edge > 0:
        net = raw_edge - params.transaction_cost_pct
        side = Signal.BUY
    elif raw_edge < 0:
        net = -raw_edge - params.transaction_cost_pct
        side = Signal.SELL
    else:
        net, side = 0.0, Signal.HOLD

    if net < params.min_edge_pct:
        return TradeDecision(
            ticker=ticker, market=market,
            fair_prob=fair_prob, market_implied_prob=market_implied_prob,
            raw_edge=raw_edge, net_edge=net,
            signal=Signal.HOLD, kelly_pct=0.0, sized_dollars=0.0,
        )

    f = kelly_fraction(fair_prob, market_implied_prob, side)
    sized = min(f * params.kelly_fraction * params.bankroll, params.max_position)

    return TradeDecision(
        ticker=ticker, market=market,
        fair_prob=fair_prob, market_implied_prob=market_implied_prob,
        raw_edge=raw_edge, net_edge=net,
        signal=side, kelly_pct=f, sized_dollars=sized,
    )


# ---------------------------------------------------------------------------
# Batch evaluation against a market snapshot
# ---------------------------------------------------------------------------

def evaluate_all(
    probs: MarketProbs,
    market_snapshot: dict[str, float],
    trade_params: TradeParams = TradeParams(),
) -> list[TradeDecision]:
    """
    Compare model fair probabilities against market-implied probabilities.

    market_snapshot: {market_key: implied_probability}
    Keys must match the format returned by MarketProbs.as_cents() but as
    probabilities (0-1 range), not cents.

    Example:
        {"above_155.00": 0.40, "return_above_+5.0%": 0.38}
    """
    cents = probs.as_cents()  # {key: fair_cents}
    decisions = []

    for key, implied_prob in market_snapshot.items():
        if key not in cents:
            continue
        fair_prob = cents[key] / 100.0
        decision = evaluate_contract(
            ticker=probs.ticker,
            market=key,
            fair_prob=fair_prob,
            market_implied_prob=implied_prob,
            params=trade_params,
        )
        decisions.append(decision)

    decisions.sort(key=lambda d: abs(d.net_edge), reverse=True)
    return decisions


# ---------------------------------------------------------------------------
# Portfolio-level summary
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSummary:
    decisions: list[TradeDecision]

    @property
    def actionable(self) -> list[TradeDecision]:
        return [d for d in self.decisions if d.signal != Signal.HOLD]

    @property
    def total_deployed(self) -> float:
        return sum(d.sized_dollars for d in self.actionable)

    def print_report(self) -> None:
        print(f"{'='*90}")
        print(f"  CRYPTO AI - TRADE REPORT  ({len(self.actionable)} actionable / {len(self.decisions)} evaluated)")
        print(f"{'='*90}")
        for d in self.decisions:
            marker = ">>>" if d.signal != Signal.HOLD else "   "
            print(f"  {marker} {d}")
        print(f"{'-'*90}")
        print(f"  Total capital to deploy: ${self.total_deployed:,.0f}")
        print(f"{'='*90}")
