"""
1-hour BTC prediction market engine.

Models Robinhood Predict-style binary contracts:
  "Will BTC be above $X at time T?"  (Yes/No, priced in cents 0-100)

Pricing approach:
  - Fetch last 30 days of 1-hour candles from yfinance for real-time vol
  - GARCH(1,1) vol forecast (captures vol clustering); falls back to realised vol
  - Log-normal GBM with optional momentum drift nudge
  - 5-hour momentum → annualised drift (20% persistence, capped at ±20 p.a.)
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
import requests
import yfinance as yf
from scipy.stats import norm

from .trading import Signal, TradeParams


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINUTES_PER_YEAR = 525_600    # BTC is 24/7/365 — used for horizon scaling only
HOURS_PER_YEAR   = 8_760      # BTC is 24/7/365 — used for hourly vol annualisation
DEFAULT_HORIZON  = 60         # minutes
VOL_WINDOW       = 60         # 1-hour candles used to estimate vol (= 2.5 days)
BASELINE_WINDOW  = 240        # 10 days of 1-hour candles for a slow vol anchor

# Binance symbol map for perpetual futures funding
_BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
}
_FUNDING_NEUTRAL = 0.0001    # ~0.01% per 8h — typical baseline, not a signal


# ---------------------------------------------------------------------------
# Funding rate (Binance perp futures — free, no auth)
# ---------------------------------------------------------------------------

def fetch_funding_rate(symbol: str) -> tuple[float, str]:
    """
    Fetch the latest perpetual futures funding rate from Binance.

    Returns (funding_rate, status_str).
    Funding is paid every 8 hours.
      Positive → longs pay shorts (market overcrowded long → bearish lean)
      Negative → shorts pay longs (market overcrowded short → bullish lean)
    """
    binance_sym = _BINANCE_SYMBOLS.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        resp = requests.get(url, params={"symbol": binance_sym}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["lastFundingRate"])
        return rate, "ok"
    except Exception as e:
        return 0.0, f"unavailable ({e})"


def funding_rate_to_drift(funding_rate: float) -> float:
    """
    Convert 8-hour funding rate into an annualised drift nudge (contrarian).

    Logic:
      - Excess funding = rate above neutral baseline (0.01%/8h)
      - Contrarian: positive excess → bearish drift (longs overcrowded)
      - Scale: 0.01% excess → ~2 annualised drift units
      - Cap at ±10 to prevent extreme distortion

    At 0.05% funding (elevated): drift ≈ -8.0 (bearish)
    At 0.10% funding (extreme):  drift ≈ -10.0 (capped)
    """
    excess = funding_rate - _FUNDING_NEUTRAL
    drift  = -excess * 20_000          # contrarian scaling
    return float(max(-10.0, min(10.0, drift)))


# ---------------------------------------------------------------------------
# Intraday data + vol estimation
# ---------------------------------------------------------------------------

def fetch_intraday(symbol: str, lookback_hours: int = 720) -> pd.DataFrame:
    """
    Download 1-hour OHLCV candles for the last N hours from yfinance.
    Default 720h = 30 days, giving ~720 candles for robust GARCH/vol estimation.
    """
    import datetime as dt
    yf_sym = f"{symbol.upper()}-USD" if "-" not in symbol else symbol.upper()
    end   = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=lookback_hours)
    df = yf.download(
        yf_sym,
        start=start,
        end=end,
        interval="1h",
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
    Annualised volatility from the last `window` 1-hour log returns.
    Annualisation factor: sqrt(HOURS_PER_YEAR) because BTC trades 24/7.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent  = log_ret.tail(window)
    if len(recent) < 5:
        raise ValueError("Not enough intraday data to estimate volatility.")
    vol_per_hour = float(recent.std())
    return vol_per_hour * math.sqrt(HOURS_PER_YEAR)


def ewma_vol_annual(df: pd.DataFrame, span: int = 24) -> float:
    """Exponentially weighted hourly volatility, annualised."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(log_ret) < 5:
        raise ValueError("Not enough intraday data to estimate EWMA volatility.")
    vol_per_hour = float(log_ret.ewm(span=span, adjust=False).std().iloc[-1])
    return vol_per_hour * math.sqrt(HOURS_PER_YEAR)


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
    vol_per_hour = math.sqrt(var_pct_sq) / 100.0
    return vol_per_hour * math.sqrt(HOURS_PER_YEAR)


@dataclass(frozen=True)
class VolEstimate:
    annual_vol: float
    source: str
    realised: float
    ewma: float
    baseline: float
    garch: float | None


def blended_vol_annual(
    df: pd.DataFrame,
    window: int = VOL_WINDOW,
    baseline_window: int = BASELINE_WINDOW,
) -> VolEstimate:
    """
    Blend fast, medium, model-based, and baseline volatility estimates.

    GARCH is useful when available, but hourly contract probabilities should not
    swing entirely on one estimator. Missing components are reweighted away.
    """
    components: list[tuple[str, float, float]] = []

    realised = realised_vol_annual(df, window=window)
    ewma = ewma_vol_annual(df, span=max(8, min(window, 48)))
    baseline = realised_vol_annual(df, window=min(baseline_window, max(5, len(df) - 1)))

    components.extend([
        ("realised", realised, 0.40),
        ("ewma", ewma, 0.30),
        ("baseline", baseline, 0.10),
    ])

    garch: float | None
    try:
        garch = garch_vol_annual(df, window=window)
        components.append(("garch", garch, 0.20))
    except Exception:
        garch = None

    weight_sum = sum(weight for _, _, weight in components)
    annual_vol = sum(value * weight for _, value, weight in components) / weight_sum
    source = "Blend" if garch is not None else "Blend(no GARCH)"

    return VolEstimate(
        annual_vol=float(annual_vol),
        source=source,
        realised=float(realised),
        ewma=float(ewma),
        baseline=float(baseline),
        garch=None if garch is None else float(garch),
    )


def estimate_momentum_drift(df: pd.DataFrame, lookback_min: int = 5) -> float:
    """
    Convert 5-hour realised return into an annualised drift nudge.

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
    T_window = lookback_min / HOURS_PER_YEAR
    annual   = (r / T_window) * 0.20    # 20% persistence factor
    return float(max(-20.0, min(20.0, annual)))


def estimate_chart_signal(df: pd.DataFrame, lookback_min: int = 5) -> tuple[float, dict]:
    """
    Combines EMA cross, price position, and volume into an annualised drift signal.

    Three components:
      1. EMA cross (EMA9 vs EMA21) — primary trend direction and strength
      2. Price position relative to EMAs — confirms or weakens the trend
      3. Volume confirmation — scales signal up if recent volume is elevated

    Blended 70/30 with raw 5-min momentum so the chart and the model agree.
    Returns (annual_drift, signal_details_dict).
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    c  = float(close.iloc[-1])
    e9  = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])

    # 1 — EMA cross: fractional separation → annual drift scale
    ema_sep   = (e9 - e21) / e21  # positive = EMA9 above = bullish
    ema_drift = math.copysign(min(abs(ema_sep) * 3000, 10.0), ema_sep)

    # 2 — Price position vs EMAs
    bullish_cross = e9 > e21
    if bullish_cross and c > e9:
        pos_drift =  4.0   # price above fast EMA in uptrend — full confirmation
    elif bullish_cross and c > e21:
        pos_drift =  2.0   # price between EMAs — pullback in uptrend
    elif bullish_cross:
        pos_drift =  0.0   # uptrend but price below both — caution
    elif not bullish_cross and c < e9:
        pos_drift = -4.0   # price below fast EMA in downtrend — full confirmation
    elif not bullish_cross and c < e21:
        pos_drift = -2.0   # price between EMAs — rally in downtrend
    else:
        pos_drift =  0.0   # downtrend but price above both — caution

    # 3 — Volume confirmation
    recent_vol = float(volume.tail(5).mean())
    avg_vol    = float(volume.mean())
    vol_factor = min(1.4, max(0.6, recent_vol / avg_vol)) if avg_vol > 0 else 1.0

    # Combine EMA cross + price position, scale by volume
    chart_signal = (ema_drift * 0.6 + pos_drift * 0.4) * vol_factor

    # Blend with raw momentum (keeps short-term price action in the mix)
    momentum = estimate_momentum_drift(df, lookback_min)
    blended  = chart_signal * 0.70 + momentum * 0.30

    drift = float(max(-20.0, min(20.0, blended)))

    details = {
        "ema_cross":    "Bullish" if bullish_cross else "Bearish",
        "price_pos":    "Above Both" if (bullish_cross and c > e9)
                        else "Below Both" if (not bullish_cross and c < e9)
                        else "Between",
        "vol_factor":   round(vol_factor, 2),
        "ema9":         round(e9, 2),
        "ema21":        round(e21, 2),
    }
    return drift, details


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


def fair_prob_range(
    current_price: float,
    low: float | None,
    high: float | None,
    annual_vol: float,
    horizon_minutes: int = DEFAULT_HORIZON,
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> float:
    """P(low <= BTC <= high), with open-ended bounds allowed."""
    above_low = 1.0 if low is None else fair_prob_yes(
        current_price, low, annual_vol, horizon_minutes, annual_drift, tail_dof
    )
    above_high = 0.0 if high is None else fair_prob_yes(
        current_price, high, annual_vol, horizon_minutes, annual_drift, tail_dof
    )
    return float(max(0.0, min(1.0, above_low - above_high)))


@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    avg_pred: float
    observed: float


@dataclass(frozen=True)
class CalibrationReport:
    samples: int
    brier: float
    bins: tuple[CalibrationBin, ...]


def build_probability_calibration(
    df: pd.DataFrame,
    horizon_minutes: int = DEFAULT_HORIZON,
    vol_window: int = VOL_WINDOW,
    num_bins: int = 10,
) -> CalibrationReport:
    """
    Backtest raw 1-hour-style probabilities on the loaded intraday history.

    The historical candles are hourly, so horizons are rounded to whole candles.
    Strikes are generated around each historical spot at several sigma distances.
    """
    close = df["Close"].dropna().astype(float)
    if len(close) < vol_window + 10:
        return CalibrationReport(samples=0, brier=float("nan"), bins=())

    horizon_bars = max(1, int(round(horizon_minutes / 60)))
    z_levels = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    preds: list[float] = []
    outcomes: list[int] = []

    end_idx = len(close) - horizon_bars
    for i in range(vol_window, end_idx):
        hist = close.iloc[: i + 1]
        spot = float(hist.iloc[-1])
        future = float(close.iloc[i + horizon_bars])
        recent = hist.pct_change().dropna().tail(vol_window)
        if len(recent) < 5:
            continue

        log_ret = np.log(hist / hist.shift(1)).dropna().tail(vol_window)
        annual_vol = float(log_ret.std() * math.sqrt(HOURS_PER_YEAR))
        sig_T = sigma_over_horizon(annual_vol, horizon_minutes)
        if sig_T <= 0 or not math.isfinite(sig_T):
            continue

        for z in z_levels:
            strike = spot * math.exp(z * sig_T)
            pred = fair_prob_yes(spot, strike, annual_vol, horizon_minutes)
            if not math.isfinite(pred):
                continue
            preds.append(pred)
            outcomes.append(1 if future > strike else 0)

    if not preds:
        return CalibrationReport(samples=0, brier=float("nan"), bins=())

    pred_arr = np.array(preds)
    out_arr = np.array(outcomes)
    brier = float(np.mean((pred_arr - out_arr) ** 2))

    bins: list[CalibrationBin] = []
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        if high >= 1.0:
            mask = (pred_arr >= low) & (pred_arr <= high)
        else:
            mask = (pred_arr >= low) & (pred_arr < high)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(CalibrationBin(
            low=float(low),
            high=float(high),
            count=count,
            avg_pred=float(pred_arr[mask].mean()),
            observed=float(out_arr[mask].mean()),
        ))

    return CalibrationReport(samples=len(preds), brier=brier, bins=tuple(bins))


def calibrate_probability(
    raw_prob: float,
    calibration: CalibrationReport | None,
    strength: float = 0.35,
) -> float:
    """
    Pull a raw model probability toward its recent empirical hit rate.

    Strength scales with sample count so tiny histories do not overfit the live
    number. With enough samples, the default uses 35% calibration / 65% raw.
    """
    if calibration is None or calibration.samples <= 0 or not calibration.bins:
        return raw_prob

    matching = [
        b for b in calibration.bins
        if b.low <= raw_prob <= b.high or (b.low <= raw_prob < b.high)
    ]
    if matching:
        empirical = matching[0].observed
        bin_weight = min(1.0, matching[0].count / 100.0)
    else:
        nearest = min(calibration.bins, key=lambda b: abs(b.avg_pred - raw_prob))
        empirical = nearest.observed
        bin_weight = min(0.5, nearest.count / 200.0)

    sample_weight = min(1.0, calibration.samples / 500.0)
    w = max(0.0, min(strength * sample_weight * bin_weight, 0.50))
    calibrated = raw_prob * (1.0 - w) + empirical * w
    return float(max(0.0, min(1.0, calibrated)))


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
    calibration: CalibrationReport | None = None,
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
        raw_yes = fair_prob_yes(current_price, k, annual_vol, horizon_minutes,
                                annual_drift, tail_dof)
        p_yes = calibrate_probability(raw_yes, calibration)
        rows.append({
            "strike":   round(k, 2),
            "raw_fair_yes": raw_yes,
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
    sigma_T: float               # horizon vol (for context)
    raw_fair_prob: float | None = None
    description: str | None = None

    def __str__(self) -> str:
        arrow = ">>" if self.signal != Signal.HOLD else "  "
        contract = self.description or f"{self.symbol} > ${self.strike:,.2f}"
        return (
            f"{arrow} {contract} ({self.side:<3})  "
            f"fair={self.fair_prob:5.1%}  mkt={self.contract_price_pct:5.1%}  "
            f"edge={self.net_edge:+5.1%}  "
            f"{self.signal.value:<4}  size=${self.sized_dollars:,.0f}"
        )


def _decision_from_probability(
    symbol: str,
    strike: float,
    contract_yes_price: float,
    fair: float,
    raw_fair: float,
    sigma_T: float,
    params: TradeParams,
    description: str | None = None,
) -> ContractDecision:
    mkt = contract_yes_price / 100.0
    raw_edge = fair - mkt

    if raw_edge > 0:
        net = raw_edge - params.transaction_cost_pct
        side = Signal.BUY
        side_label = "YES"
    elif raw_edge < 0:
        net = -raw_edge - params.transaction_cost_pct
        side = Signal.SELL
        side_label = "NO"
    else:
        net, side, side_label = 0.0, Signal.HOLD, "---"

    if net < params.min_edge_pct:
        return ContractDecision(
            symbol=symbol, strike=strike, side=side_label,
            fair_prob=fair, contract_price_pct=mkt,
            raw_edge=raw_edge, net_edge=net,
            signal=Signal.HOLD, kelly_pct=0.0, sized_dollars=0.0,
            sigma_T=sigma_T, raw_fair_prob=raw_fair, description=description,
        )

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
        sigma_T=sigma_T, raw_fair_prob=raw_fair, description=description,
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
    calibration: CalibrationReport | None = None,
) -> ContractDecision:
    """
    Evaluate one Yes/No binary contract.

    contract_yes_price: the price shown on Robinhood in cents, e.g. 45 means
                        the market thinks there's a 45% chance BTC ends above strike.
    annual_drift: momentum-derived drift from estimate_momentum_drift().
    tail_dof: Student-t dof from estimate_tail_dof(); corrects for fat tails.
    """
    raw_fair = fair_prob_yes(current_price, strike, annual_vol, horizon_minutes,
                             annual_drift, tail_dof)
    fair  = calibrate_probability(raw_fair, calibration)
    sig_T = sigma_over_horizon(annual_vol, horizon_minutes)
    return _decision_from_probability(
        symbol, strike, contract_yes_price, fair, raw_fair, sig_T, params
    )


def evaluate_range_contract(
    symbol: str,
    low: float | None,
    high: float | None,
    contract_yes_price: float,
    current_price: float,
    annual_vol: float,
    horizon_minutes: int = DEFAULT_HORIZON,
    params: TradeParams = TradeParams(),
    annual_drift: float = 0.0,
    tail_dof: float = 30.0,
) -> ContractDecision:
    """Evaluate a Kalshi-style price range contract."""
    raw_fair = fair_prob_range(current_price, low, high, annual_vol,
                               horizon_minutes, annual_drift, tail_dof)
    sig_T = sigma_over_horizon(annual_vol, horizon_minutes)
    if low is None:
        description = f"{symbol} <= ${high:,.2f}"
        strike_for_sort = high or current_price
    elif high is None:
        description = f"{symbol} >= ${low:,.2f}"
        strike_for_sort = low
    else:
        description = f"{symbol} ${low:,.2f}-${high:,.2f}"
        strike_for_sort = low
    return _decision_from_probability(
        symbol, strike_for_sort, contract_yes_price, raw_fair, raw_fair,
        sig_T, params, description
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
    calibration: CalibrationReport | None = None,
) -> list[ContractDecision]:
    """
    Evaluate a batch of contracts. Pass all the Yes prices you see on Robinhood.
    contracts = {76500: 42, 77000: 28, 76000: 61, ...}
    """
    decisions = [
        evaluate_contract(symbol, strike, yes_price, current_price,
                          annual_vol, horizon_minutes, params, annual_drift, tail_dof,
                          calibration)
        for strike, yes_price in contracts.items()
    ]
    decisions.sort(key=lambda d: abs(d.net_edge), reverse=True)
    return decisions
