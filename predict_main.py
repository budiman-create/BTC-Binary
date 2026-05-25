"""
Robinhood Predict — 15-minute BTC binary contract pricing tool.

Two modes:

  1. LADDER MODE (default) — shows fair probabilities at strikes around spot.
     Use this to quickly scan which contracts on Robinhood are mispriced.

     python predict_main.py BTC

  2. CONTRACT MODE — evaluate a specific contract you see on Robinhood.
     Pass the strike and the Yes price shown on screen.

     python predict_main.py BTC --strike 76500 --yes-price 45

  3. SCAN MODE — evaluate multiple contracts at once (paste from Robinhood).

     python predict_main.py BTC --contracts "76000:62,76500:45,77000:28"

Options:
  --horizon   Minutes until expiry (default: 60)
  --bankroll  Your bankroll in USD (default: 1000)
  --vol-window  Hourly candles used for vol estimate (default: 60)
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from stock_agent.prediction_market import (
    ContractDecision,
    DEFAULT_HORIZON,
    VOL_WINDOW,
    build_probability_calibration,
    blended_vol_annual,
    evaluate_contract,
    evaluate_range_contract,
    estimate_momentum_drift,
    estimate_tail_dof,
    fetch_intraday,
    probability_ladder,
    scan_contracts,
    sigma_over_horizon,
)
from stock_agent.robinhood_crypto import RobinhoodCryptoClient, fetch as rh_fetch
from stock_agent.market_state import Horizon
from stock_agent.trading import Signal, TradeParams

load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_ladder(
    symbol: str,
    current_price: float,
    annual_vol: float,
    horizon: int,
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
    calibration=None,
) -> None:
    sig_T = sigma_over_horizon(annual_vol, horizon)
    rows  = probability_ladder(current_price, annual_vol, horizon, num_strikes=12,
                               pct_range=0.025, annual_drift=annual_drift,
                               tail_dof=tail_dof, calibration=calibration)

    print(f"\n{'='*65}")
    print(f"  {symbol} PREDICTION MARKET - PROBABILITY LADDER")
    print(f"{'='*65}")
    print(f"  Current price : ${current_price:,.2f}")
    print(f"  Intraday vol  : {annual_vol:.1%} p.a.  ({horizon}-min sigma: {sig_T:.3%})")
    print(f"{'='*65}")
    print(f"  {'Strike':>10}  {'Raw':>8}  {'Cal':>8}  {'P(No)':>8}  {'Bar (Cal)':<22}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*22}")
    for r in rows:
        atm = " <-- ATM" if abs(r["strike"] - current_price) / current_price < 0.002 else ""
        print(
            f"  ${r['strike']:>9,.2f}  "
            f"{r['raw_fair_yes']:>7.1%}  "
            f"{r['fair_yes']:>7.1%}  "
            f"{r['fair_no']:>7.1%}  "
            f"{_bar(r['fair_yes'])}{atm}"
        )
    print(f"{'='*65}")
    print(f"  Compare these fair probabilities to the cents shown on")
    print(f"  Robinhood Predict. If fair > market -> BUY Yes.")
    print(f"  If fair < market -> BUY No (or SELL Yes).")
    print(f"{'='*65}\n")


def _print_decisions(decisions: list[ContractDecision], bankroll: float) -> None:
    actionable = [d for d in decisions if d.signal != Signal.HOLD]
    total = sum(d.sized_dollars for d in actionable)

    print(f"\n{'='*75}")
    print(f"  CONTRACT EVALUATION  ({len(actionable)} actionable / {len(decisions)} total)")
    print(f"{'='*75}")
    for d in decisions:
        print(f"  {d}")
    print(f"{'-'*75}")
    print(f"  Total to deploy: ${total:,.0f}  (bankroll: ${bankroll:,.0f})")
    print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    symbol: str,
    horizon: int,
    bankroll: float,
    vol_window: int,
    strike: float | None,
    yes_price: float | None,
    contracts_str: str | None,
    kalshi_ticker: str | None = None,
    find_kalshi_btc: bool = False,
) -> None:
    if find_kalshi_btc:
        from stock_agent.kalshi_market import find_btc_markets

        markets = find_btc_markets(max_expiry_hours=2)
        if not markets:
            print("No open BTC/Bitcoin Kalshi markets expiring in the next 2 hours.")
            print("  Try: from stock_agent.kalshi_market import find_btc_markets; find_btc_markets()")
            return
        print("\nOpen BTC/Bitcoin Kalshi markets (≤2 hours, soonest first):")
        for m in markets:
            strike_str = f"${m['strike']:,.2f}" if m.get("strike") is not None else "range"
            mins = f"{m['minutes_left']:.0f} min" if m.get("minutes_left") is not None else m.get("close_time", "")
            print(f"  {m['ticker']:<35} {strike_str:<16} {mins:<12}  {m.get('title', '')}")
        print("\nTip: pass one ticker to --kalshi-ticker to evaluate it against the model.")
        return

    # ------------------------------------------------------------------
    # 1. Live price from Robinhood
    # ------------------------------------------------------------------
    print(f"\nFetching live {symbol} price from Robinhood ...")
    try:
        client = RobinhoodCryptoClient()
        rh_data = rh_fetch(symbol, horizon=Horizon.DAY, client=client)
        current_price = rh_data.current_price
        print(f"  Live price : ${current_price:,.4f}")
        if rh_data.spread_pct:
            print(f"  Bid/ask    : ${rh_data.bid_price:,.4f} / ${rh_data.ask_price:,.4f}  "
                  f"(spread {rh_data.spread_pct:.3%})")
    except Exception as e:
        print(f"  [!] Robinhood unavailable ({e}), falling back to yfinance price.")
        df_tmp = fetch_intraday(symbol, lookback_hours=1)
        current_price = float(df_tmp["Close"].iloc[-1])
        print(f"  yfinance price: ${current_price:,.4f}")

    # ------------------------------------------------------------------
    # 2. Intraday volatility (GARCH preferred) + momentum drift
    # ------------------------------------------------------------------
    lookback_hours = max(240, vol_window + max(1, round(horizon / 60)) + 48)
    print(f"  Fetching {lookback_hours} hours of intraday data ...")
    df = fetch_intraday(symbol, lookback_hours=lookback_hours)

    vol_est = blended_vol_annual(df, window=vol_window)
    annual_vol = vol_est.annual_vol
    vol_label = vol_est.source
    garch_text = f"{vol_est.garch:.1%}" if vol_est.garch is not None else "N/A"
    print(
        f"  Blended vol  : {annual_vol:.1%} p.a.  "
        f"(realised {vol_est.realised:.1%}, EWMA {vol_est.ewma:.1%}, "
        f"GARCH {garch_text}, baseline {vol_est.baseline:.1%})"
    )

    sig_T        = sigma_over_horizon(annual_vol, horizon)
    annual_drift = estimate_momentum_drift(df)
    tail_dof     = estimate_tail_dof(df)
    calibration  = build_probability_calibration(df, horizon, vol_window)
    drift_T      = annual_drift * (horizon / 525_600)
    print(f"  {horizon}-min sigma : {sig_T:.3%}  |  vol source: {vol_label}")
    print(f"  Momentum drift: {annual_drift:+.2f} p.a.  ({drift_T:+.4%} over {horizon} min)")
    tail_str = f"{tail_dof:.1f}" if tail_dof < 30 else "normal (>=30)"
    calibration_text = (
        f"{calibration.samples} samples, Brier {calibration.brier:.3f}"
        if calibration.samples
        else "unavailable (not enough history)"
    )
    print(f"  Calibration   : {calibration_text}")
    print(f"  Tail dof      : {tail_str}  ({'fat tails — OTM probs boosted' if tail_dof < 15 else 'near-normal'})")

    # ------------------------------------------------------------------
    # 3. Run requested mode
    # ------------------------------------------------------------------

    tp = TradeParams(
        bankroll=bankroll,
        kelly_fraction=0.25,
        max_position_pct=0.10,
        min_edge_pct=0.03,
        transaction_cost_pct=0.02,   # prediction market spread ~2%
    )

    if kalshi_ticker:
        from stock_agent.kalshi_market import get_quote

        q = get_quote(kalshi_ticker)
        contract_price = yes_price if yes_price is not None else q.display_price_cents
        if contract_price is None:
            raise ValueError("Kalshi orderbook has no usable Yes price.")

        print(f"\nKalshi market : {q.ticker}")
        print(f"  {q.title}")
        print(
            f"  Yes bid/ask/mid: "
            f"{q.yes_bid_cents if q.yes_bid_cents is not None else 'N/A'} / "
            f"{q.yes_ask_cents if q.yes_ask_cents is not None else 'N/A'} / "
            f"{q.yes_mid_cents if q.yes_mid_cents is not None else 'N/A'} cents"
        )
        if q.floor_strike is not None or q.cap_strike is not None:
            d = evaluate_range_contract(symbol, q.floor_strike, q.cap_strike,
                                        contract_price, current_price, annual_vol,
                                        horizon, tp, annual_drift, tail_dof)
        else:
            contract_strike = strike if strike is not None else q.strike
            if contract_strike is None:
                raise ValueError("Kalshi strike could not be inferred; pass --strike.")
            d = evaluate_contract(symbol, contract_strike, contract_price, current_price,
                                  annual_vol, horizon, tp, annual_drift, tail_dof,
                                  calibration)
        _print_decisions([d], bankroll)

    elif contracts_str:
        # SCAN MODE: --contracts "76000:62,76500:45,77000:28"
        contracts: dict[float, float] = {}
        for item in contracts_str.split(","):
            k, v = item.strip().split(":")
            contracts[float(k.strip())] = float(v.strip())
        decisions = scan_contracts(symbol, current_price, annual_vol,
                                   contracts, horizon, tp, annual_drift, tail_dof,
                                   calibration)
        _print_decisions(decisions, bankroll)

    elif strike is not None and yes_price is not None:
        # CONTRACT MODE: single contract
        d = evaluate_contract(symbol, strike, yes_price, current_price,
                              annual_vol, horizon, tp, annual_drift, tail_dof,
                              calibration)
        _print_decisions([d], bankroll)

    else:
        # LADDER MODE: show probability table
        _print_ladder(symbol, current_price, annual_vol, horizon, annual_drift,
                      tail_dof, calibration)
        print("  Tip: once you see prices on Robinhood, run:")
        print(f"  python predict_main.py {symbol} --contracts \"STRIKE:YES_PRICE,...\"")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Robinhood 1-hour prediction market pricer")
    parser.add_argument("symbol",       nargs="?", default="BTC")
    parser.add_argument("--horizon",    type=int,   default=DEFAULT_HORIZON,
                        help="Minutes to expiry (default: 60)")
    parser.add_argument("--bankroll",   type=float, default=1_000.0)
    parser.add_argument("--vol-window", type=int,   default=VOL_WINDOW,
                        help="Hourly candles for vol estimate (default: 60)")
    parser.add_argument("--strike",     type=float, default=None,
                        help="Contract strike price (e.g. 76500)")
    parser.add_argument("--yes-price",  type=float, default=None,
                        help="Robinhood Yes price in cents (e.g. 45)")
    parser.add_argument("--contracts",  type=str,   default=None,
                        help="Multiple contracts: \"76000:62,76500:45,77000:28\"")
    parser.add_argument("--kalshi-ticker", type=str, default=None,
                        help="Kalshi market ticker to price from live orderbook")
    parser.add_argument("--find-kalshi-btc", action="store_true",
                        help="List open BTC/Bitcoin Kalshi market tickers")

    args = parser.parse_args()
    run(
        symbol=args.symbol.upper(),
        horizon=args.horizon,
        bankroll=args.bankroll,
        vol_window=args.vol_window,
        strike=args.strike,
        yes_price=args.yes_price,
        contracts_str=args.contracts,
        kalshi_ticker=args.kalshi_ticker,
        find_kalshi_btc=args.find_kalshi_btc,
    )


if __name__ == "__main__":
    main()
