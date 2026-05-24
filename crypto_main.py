"""
AI Agent for Crypto — Robinhood edition.

Pipeline:
  1. Login to Robinhood (reads credentials from .env)
  2. Fetch live crypto quote + candle history
  3. Technical analysis → StockState
  4. Optional Claude AI analyst → drift nudge
  5. Log-normal price model → fair probabilities
  6. Edge + Kelly sizing → BUY/SELL/HOLD signals
  7. Print report (+ show current positions if --positions flag)

Usage:
    .venv/Scripts/python.exe crypto_main.py BTC
    .venv/Scripts/python.exe crypto_main.py ETH --horizon 1w
    .venv/Scripts/python.exe crypto_main.py SOL --no-ai --positions
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from stock_agent.market_state import Horizon
from stock_agent.price_model import compute_probabilities
from stock_agent.robinhood_crypto import build_state, fetch, get_positions
from stock_agent.trading import PortfolioSummary, TradeParams, evaluate_all

load_dotenv()


HORIZON_MAP = {
    "1d":  Horizon.DAY,
    "1w":  Horizon.WEEK,
    "1mo": Horizon.MONTH,
    "3mo": Horizon.QUARTER,
}

RETURN_TARGETS = (-0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.30)


def run(
    symbol: str,
    horizon: Horizon = Horizon.MONTH,
    bankroll: float = 10_000.0,
    use_ai: bool = True,
    show_positions: bool = False,
) -> None:
    # ------------------------------------------------------------------
    # Optional: show current holdings first
    # ------------------------------------------------------------------
    if show_positions:
        print("\nCurrent Robinhood Crypto Positions:")
        positions = get_positions()
        if positions:
            for p in positions:
                print(f"  {p['symbol']:>6}  qty={p['quantity']:.6f}  avg_buy=${p['average_buy_price']:.4f}")
        else:
            print("  (no open positions)")
        print()

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    print(f"Fetching Robinhood data for {symbol} …")
    data = fetch(symbol, horizon=horizon)
    print(f"  Live price : ${data.current_price:.4f}")
    if data.spread_pct is not None:
        print(f"  Bid/ask    : ${data.bid_price:.4f} / ${data.ask_price:.4f}  (spread {data.spread_pct:.3%})")

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
            rough_state = build_state(data, horizon)
            report = analyse(
                state=rough_state,
                extra_context=data.extra_context(),
            )
            report.print_report()
            drift_nudge = report.drift_nudge

    # ------------------------------------------------------------------
    # 3. Build state
    # ------------------------------------------------------------------
    state = build_state(data, horizon, analyst_drift_nudge=drift_nudge)

    print(f"\nCrypto state for {state.ticker}:")
    print(f"  Price      : ${state.current_price:.4f}")
    print(f"  Drift (mu) : {state.annual_drift:+.1%} p.a.")
    print(f"  Vol   (sig): {state.annual_volatility:.1%} p.a.")
    print(f"  Trend      : {state.trend_regime}")
    print(f"  Vol regime : {state.vol_regime}")
    print(f"  RSI        : {state.rsi:.1f}")
    print(f"  Horizon    : {state.horizon.value}\n")

    # ------------------------------------------------------------------
    # 4. Price model
    # ------------------------------------------------------------------
    probs = compute_probabilities(state, return_targets=RETURN_TARGETS)
    print(probs.summary())
    print()
    print("Fair probabilities:")
    for k, v in probs.as_cents().items():
        print(f"  {k:<45}: {v:5.1f}c")

    # ------------------------------------------------------------------
    # 5. Market snapshot
    #    Use bid/ask spread as the transaction cost proxy.
    #    Market-implied probs: flat 50/50 baseline (replace with options
    #    data or your own consensus when available).
    # ------------------------------------------------------------------
    spread = data.spread_pct or 0.005
    tp = TradeParams(
        bankroll=bankroll,
        kelly_fraction=0.25,
        max_position_pct=0.10,
        min_edge_pct=0.04,
        transaction_cost_pct=spread,
    )

    market_snapshot = {
        f"return_above_{'+' if r >= 0 else ''}{r:.1%}": 0.50
        for r in RETURN_TARGETS
    }

    decisions = evaluate_all(probs, market_snapshot, tp)
    print()
    PortfolioSummary(decisions).print_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent for Crypto (Robinhood)")
    parser.add_argument("symbol", nargs="?", default="BTC", help="Crypto symbol (BTC, ETH, SOL …)")
    parser.add_argument("--horizon", default="1mo", choices=list(HORIZON_MAP.keys()))
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    parser.add_argument("--no-ai", action="store_true", help="Skip Claude AI analyst")
    parser.add_argument("--positions", action="store_true", help="Show current Robinhood holdings")

    args = parser.parse_args()
    run(
        symbol=args.symbol.upper(),
        horizon=HORIZON_MAP[args.horizon],
        bankroll=args.bankroll,
        use_ai=not args.no_ai,
        show_positions=args.positions,
    )


if __name__ == "__main__":
    main()
