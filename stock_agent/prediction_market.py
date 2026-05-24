"""
15-minute BTC prediction market engine.

Models Robinhood Predict-style binary contracts:
  "Will BTC be above $X at time T?"  (Yes/No, priced in cents 0-100)

Pricing approach:
  - Fetch last 60 minutes of 1-min candles from yfinance for real-time vol
  - GARCH(1,1) vol forecast (captures vol clustering); falls back to realised vol
  - Log-normal GBM with optional momentum drift nudge
  - 5-min momentum → annualised drift (20% persistence, capped at ±20 p.a.)
  - Compare fair probability to contract price → edge → Kelly size

Key insight: this is IDENTICAL to the soccer Kalshi model — binary contract,
fair price vs market price, Kelly position sizing. Only the probability
engine changes (log-normal instead of Poisson).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from .trading import Signal, TradeParams


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINUTES_PER_YEAR = 525_600    # BTC is 24/7/365
DEFAULT_HORIZON  = 15         # minutes
VOL_WINDOW       = 60         # minutes of 1-min data used to estimate vol


# ---------------------------------------------------------------------------
# Intraday data + vol estimation
# ---------------------------------------------------------------------------

def fetch_intraday(symbol: str, lookback_hours: int = 3) -> pd.DataFrame:
    """
    Download 1-minute OHLCV candles for the last N hours from yfinance.
    Uses start/end datetime because yfinance rejects period strings < 1d for 1m interval.
    """
    import datetime as dt
    yf_sym = f"{symbol.upper()}-USD" if "-" not in symbol else symbol.upper()
    end   = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=lookback_hours)
    df = yf.download(
        yf_sym,
        start=start,
        end=end,
        interval="1m",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No intraday data for {yf_sym}. Check symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def realised_vol_annual(df: pd.DataFrame, window: int = VOL_WINDOW) -> float:
    """
    Annualised volatility from the last `window` 1-min log returns.
    Annualisation factor: sqrt(MINUTES_PER_YEAR) because BTC trades 24/7.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent  = log_ret.tail(window)
    if len(recent) < 5:
        raise ValueError("Not enough intraday data to estimate volatility.")
    vol_per_min = float(recent.std())
    return vol_per_min * math.sqrt(MINUTES_PER_YEAR)


def garch_vol_annual(df: pd.DataFrame, window: int = VOL_WINDOW) -> float:
    """
    GARCH(1,1) one-step-ahead vol forecast, annualised.
    Captures vol clustering: recent high vol predicts near-future high vol.
    Requires the `arch` package (pip install arch).
    """
    from arch import arch_model  # optional dependency
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent  = log_ret.tail(window)
    if len(recent) < 20:
        raise ValueError("Need >= 20 observations for GARCH.")
    pct = recent * 100  # arch library expects percentage returns
    model = arch_model(pct, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
    fit   = model.fit(disp="off", show_warning=False)
    forecast     = fit.forecast(horizon=1, reindex=False)
    var_pct_sq   = float(forecast.variance.iloc[-1, 0])   # in (%²)
    vol_per_min  = math.sqrt(var_pct_sq) / 100.0
    return vol_per_min * math.sqrt(MINUTES_PER_YEAR)


def estimate_momentum_drift(df: pd.DataFrame, lookback_min: int = 5) -> float:
    """
    Convert 5-min realised return into an annualised drift nudge.

    Crypto momentum has ~20% persistence — it fades fast.
    Cap at ±20 annualised, which shifts ATM probability by at most ~8%.
    Returns 0.0 for moves below 0.1% (noise threshold).
    """
    recent = df["Close"].tail(lookback_min + 1)
    if len(recent) < 2:
        return 0.0
    r = float(math.log(float(recent.iloc[-1]) / float(recent.iloc[0])))
    if abs(r) < 0.001:          # sub-0.1% move — treat as noise
        return 0.0
    T_window = lookback_min / MINUTES_PER_YEAR
    annual   = (r / T_window) * 0.20    # 20% persistence factor
    return float(max(-20.0, min(20.0, annual)))


def estimate_tail_dof(df: pd.DataFrame, window: int = VOL_WINDOW) -> float:
    """
    Estimate Student-t degrees of freedom from recent log-return kurtosis.

    BTC returns are fat-tailed (excess kurtosis 3-8), meaning the normal
    distribution underestimates the probability of extreme moves.
    Lower dof = fatter tails (BTC typically lands around 4-6).
    Returns a value in [3, 30]; at 30 the t-distribution is ~normal.

    Method of moments: excess_kurtosis = 6 / (dof - 4)  →  dof = 6/k + 4
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent  = log_ret.tail(window)
    if len(recent) < 20:
        return 5.0          # fallback: moderately fat tails
    excess_kurt = float(recent.kurtosis())   # pandas returns excess kurtosis
    if excess_kurt <= 0:
        return 30.0         # thin/normal tails
    dof = 6.0 / excess_kurt + 4.0
    return float(max(3.0, min(30.0, dof)))


def sigma_over_horizon(annual_vol: float, horizon_minutes: int = DEFAULT_HORIZON) -> float:
    """Vol scaled to the prediction window."""
    T = horizon_minutes / MINUTES_PER_YEAR
    return annual_vol * math.sqrt(T)


# ---------------------------------------------------------------------------
# Fair probability
# ---------------------------------------------------------------------------

def fair_prob_yes(
    current_price: float,
    strike: float,
    annual_vol: float,
    horizon_minutes: int = DEFAULT_HORIZON,
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> float:
    """
    P(BTC > strike in `horizon_minutes`) under log-normal GBM.

    tail_dof: Student-t degrees of freedom for fat-tail correction.
              30+ → standard normal (no correction).
              4-6 → BTC-typical fat tails; raises OTM probabilities.
    Formula: P(S_T > K) = t.sf( (ln(K/S0) + 0.5*sigma_T^2 - mu*T) / sigma_T, df=tail_dof )
    """
    from scipy.stats import t as t_dist
    T       = horizon_minutes / MINUTES_PER_YEAR
    sigma_T = annual_vol * math.sqrt(T)
    if sigma_T <= 0:
        return 1.0 if current_price > strike else 0.0
    x = (math.log(strike / current_price) + 0.5 * sigma_T ** 2 - annual_drift * T) / sigma_T
    if tail_dof >= 30.0:
        return float(norm.sf(x))
    return float(t_dist.sf(x, df=tail_dof))


def fair_prob_no(
    current_price: float,
    strike: float,
    annual_vol: float,
    horizon_minutes: int = DEFAULT_HORIZON,
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> float:
    return 1.0 - fair_prob_yes(current_price, strike, annual_vol,
                               horizon_minutes, annual_drift, tail_dof)


# ---------------------------------------------------------------------------
# Probability ladder (show fair odds at many strikes)
# ---------------------------------------------------------------------------

def probability_ladder(
    current_price: float,
    annual_vol: float,
    horizon_minutes: int = DEFAULT_HORIZON,
    num_strikes: int = 10,
    pct_range: float = 0.03,
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> list[dict]:
    """
    Generate a table of fair probabilities at strikes around the current price.
    Returns list of {strike, fair_yes, fair_no, sigma_T}.
    """
    sig_T = sigma_over_horizon(annual_vol, horizon_minutes)

    low     = current_price * (1 - pct_range)
    high    = current_price * (1 + pct_range)
    strikes = [low + (high - low) * i / (num_strikes - 1) for i in range(num_strikes)]

    rows = []
    for k in strikes:
        p_yes = fair_prob_yes(current_price, k, annual_vol, horizon_minutes,
                              annual_drift, tail_dof)
        rows.append({
            "strike":   round(k, 2),
            "fair_yes": p_yes,
            "fair_no":  1 - p_yes,
            "sigma_T":  sig_T,
        })
    return rows


# ---------------------------------------------------------------------------
# Single contract evaluation (mirrors soccer evaluate_contract exactly)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractDecision:
    symbol: str
    strike: float
    side: str                    # "YES" or "NO"
    fair_prob: float
    contract_price_pct: float    # market's implied probability (0-1)
    raw_edge: float
    net_edge: float
    signal: Signal
    kelly_pct: float
    sized_dollars: float
    sigma_T: float               # 15-min vol (for context)

    def __str__(self) -> str:
        arrow = ">>" if self.signal != Signal.HOLD else "  "
        return (
            f"{arrow} {self.symbol} > ${self.strike:,.2f} ({self.side:<3})  "
            f"fair={self.fair_prob:5.1%}  mkt={self.contract_price_pct:5.1%}  "
            f"edge={self.net_edge:+5.1%}  "
            f"{self.signal.value:<4}  size=${self.sized_dollars:,.0f}"
        )


def evaluate_contract(
    symbol: str,
    strike: float,
    contract_yes_price: float,      # Robinhood's Yes price in cents (0-100)
    current_price: float,
    annual_vol: float,
    horizon_minutes: int = DEFAULT_HORIZON,
    params: TradeParams = TradeParams(),
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> ContractDecision:
    """
    Evaluate one Yes/No binary contract.

    contract_yes_price: the price shown on Robinhood in cents, e.g. 45 means
                        the market thinks there's a 45% chance BTC ends above strike.
    annual_drift: momentum-derived drift from estimate_momentum_drift().
    tail_dof: Student-t dof from estimate_tail_dof(); corrects for fat tails.
    """
    fair   = fair_prob_yes(current_price, strike, annual_vol, horizon_minutes,
                           annual_drift, tail_dof)
    mkt    = contract_yes_price / 100.0
    sig_T  = sigma_over_horizon(annual_vol, horizon_minutes)

    raw_edge = fair - mkt

    if raw_edge > 0:
        net  = raw_edge - params.transaction_cost_pct
        side = Signal.BUY    # buy Yes (market underpricing Yes)
        side_label = "YES"
    elif raw_edge < 0:
        net  = -raw_edge - params.transaction_cost_pct
        side = Signal.SELL   # buy No (market overpricing Yes)
        side_label = "NO"
    else:
        net, side, side_label = 0.0, Signal.HOLD, "---"

    if net < params.min_edge_pct:
        return ContractDecision(
            symbol=symbol, strike=strike, side=side_label,
            fair_prob=fair, contract_price_pct=mkt,
            raw_edge=raw_edge, net_edge=net,
            signal=Signal.HOLD, kelly_pct=0.0, sized_dollars=0.0,
            sigma_T=sig_T,
        )

    # Kelly: BUY Yes at price p, true prob q  →  f* = (q-p)/(1-p)
    #        BUY No  at price p, true prob q  →  f* = (p-q)/p  [selling Yes]
    if side == Signal.BUY:
        f = (fair - mkt) / (1 - mkt) if mkt < 1 else 0.0
    else:
        f = (mkt - fair) / mkt if mkt > 0 else 0.0

    sized = min(max(f, 0) * params.kelly_fraction * params.bankroll, params.max_position)

    return ContractDecision(
        symbol=symbol, strike=strike, side=side_label,
        fair_prob=fair, contract_price_pct=mkt,
        raw_edge=raw_edge, net_edge=net,
        signal=side, kelly_pct=f, sized_dollars=sized,
        sigma_T=sig_T,
    )


# ---------------------------------------------------------------------------
# Scan multiple contracts at once
# ---------------------------------------------------------------------------

def scan_contracts(
    symbol: str,
    current_price: float,
    annual_vol: float,
    contracts: dict[float, float],     # {strike: yes_price_in_cents}
    horizon_minutes: int = DEFAULT_HORIZON,
    params: TradeParams = TradeParams(),
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> list[ContractDecision]:
    """
    Evaluate a batch of contracts. Pass all the Yes prices you see on Robinhood.
    contracts = {76500: 42, 77000: 28, 76000: 61, ...}
    """
    decisions = [
        evaluate_contract(symbol, strike, yes_price, current_price,
                          annual_vol, horizon_minutes, params, annual_drift, tail_dof)
        for strike, yes_price in contracts.items()
    ]
    decisions.sort(key=lambda d: abs(d.net_edge), reverse=True)
    return decisions
