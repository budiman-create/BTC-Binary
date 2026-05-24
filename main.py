"""
AI Agent for Stock — end-to-end example.

Runs the full pipeline:
  1. Fetch market data (yfinance)
  2. Technical analysis → StockState
  3. AI analyst (Claude) → drift nudge
  4. Log-normal price model → fair probabilities
  5. Edge calculation + Kelly sizing → trade decisions
  6. Print report

Usage:
    python main.py                      # default: AAPL, 1-month horizon
    python main.py NVDA --horizon 1w
    python main.py MSFT --no-ai         # skip Claude API call

Set ANTHROPIC_API_KEY in your environment (or .env file) to enable the AI layer.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from stock_agent.data_fetcher import build_state, fetch, implied_probs_from_consensus
from stock_agent.market_state import Horizon, ModelParams
from stock_agent.price_model import compute_probabilities
from stock_agent.trading import PortfolioSummary, TradeParams, evaluate_all

load_dotenv()


HORIZON_MAP = {
    "1d": Horizon.DAY,
    "1w": Horizon.WEEK,
    "1mo": Horizon.MONTH,
    "3mo": Horizon.QUARTER,
}


def run(
    ticker: str,
    horizon: Horizon = Horizon.MONTH,
    bankroll: float = 10_000.0,
    use_ai: bool = True,
    return_targets: tuple[float, ...] = (-0.10, -0.05, 0.05, 0.10, 0.20),
) -> None:
    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    print(f"\nFetching market data for {ticker} …")
    data = fetch(ticker)

    # ------------------------------------------------------------------
    # 2. Optional AI analyst
    # ------------------------------------------------------------------
    drift_nudge = 0.0
    if use_ai:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  [AI] ANTHROPIC_API_KEY not set — skipping AI analyst.")
            use_ai = False
        else:
            from stock_agent.ai_analyst import analyse
            print("  [AI] Calling Claude analyst …")
            report = analyse(
                state=build_state(data, horizon),   # rough state for context
                extra_context=data.extra_context(),
            )
            report.print_report()
            drift_nudge = report.drift_nudge

    # ------------------------------------------------------------------
    # 3. Build state with AI drift nudge baked in
    # ------------------------------------------------------------------
    state = build_state(data, horizon, analyst_drift_nudge=drift_nudge)

    print(f"Stock state for {state.ticker}:")
    print(f"  Price      : ${state.current_price:.2f}")
    print(f"  Drift (μ)  : {state.annual_drift:+.1%} p.a.")
    print(f"  Vol   (σ)  : {state.annual_volatility:.1%} p.a.")
    print(f"  Trend      : {state.trend_regime}")
    print(f"  Vol regime : {state.vol_regime}")
    print(f"  RSI        : {state.rsi:.1f}")
    print(f"  200-MA dist: {state.pct_from_200ma:+.1%}")
    print(f"  Horizon    : {state.horizon.value}\n")

    # ------------------------------------------------------------------
    # 4. Price model
    # ------------------------------------------------------------------
    probs = compute_probabilities(state, return_targets=return_targets)
    print(probs.summary())
    print()
    print("Fair probabilities:")
    for k, v in probs.as_cents().items():
        print(f"  {k:<40}: {v:5.1f}¢")

    # ------------------------------------------------------------------
    # 5. Market snapshot — use consensus target if available, else demo
    # ------------------------------------------------------------------
    market_snapshot = implied_probs_from_consensus(data, return_targets)

    if market_snapshot is None:
        # Fallback: build a flat 50/50 snapshot for demonstration
        print("\n  [No analyst consensus target found — using 50/50 market snapshot]")
        market_snapshot = {
            f"return_above_{'+' if r >= 0 else ''}{r:.1%}": 0.50
            for r in return_targets
        }

    # ------------------------------------------------------------------
    # 6. Trade decisions
    # ------------------------------------------------------------------
    tp = TradeParams(
        bankroll=bankroll,
        kelly_fraction=0.25,
        max_position_pct=0.10,
        min_edge_pct=0.03,
        transaction_cost_pct=0.005,
    )

    decisions = evaluate_all(probs, market_snapshot, tp)
    summary = PortfolioSummary(decisions)
    print()
    summary.print_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent for Stock")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol")
    parser.add_argument(
        "--horizon",
        default="1mo",
        choices=list(HORIZON_MAP.keys()),
        help="Analysis horizon",
    )
    parser.add_argument("--bankroll", type=float, default=10_000.0, help="Portfolio bankroll")
    parser.add_argument("--no-ai", action="store_true", help="Skip Claude AI analyst call")

    args = parser.parse_args()
    horizon = HORIZON_MAP[args.horizon]

    run(
        ticker=args.ticker.upper(),
        horizon=horizon,
        bankroll=args.bankroll,
        use_ai=not args.no_ai,
    )


if __name__ == "__main__":
    main()
