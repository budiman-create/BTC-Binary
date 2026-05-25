"""
BTC Prediction Market AI — Streamlit web dashboard.

Run:
    streamlit run web_app.py

Features:
  - Live BTC price + bid/ask from Robinhood
  - Real-time intraday volatility from 1-hour candle data (30 days)
  - Probability ladder: fair Yes/No odds at strikes around spot
  - Contract evaluator: paste strike + Yes price → instant BUY/SELL/HOLD signal
  - 1-hour price chart (last 30 days)
  - Auto-refresh every 30 seconds
"""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BTC Prediction Market AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached data fetchers (cache 25 sec so auto-refresh gets fresh data)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=8)
def get_live_price(symbol: str):
    from stock_agent.robinhood_crypto import RobinhoodCryptoClient, fetch as rh_fetch
    from stock_agent.market_state import Horizon
    try:
        client = RobinhoodCryptoClient()
        data = rh_fetch(symbol, horizon=Horizon.DAY, client=client)
        return {
            "price": data.current_price,
            "bid":   data.bid_price,
            "ask":   data.ask_price,
            "spread": data.spread_pct,
        }
    except Exception as e:
        try:
            import yfinance as yf

            yf_symbol = f"{symbol.upper()}-USD"
            df = yf.download(yf_symbol, period="1d", interval="1m",
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("yfinance returned no fallback price data")
            return {
                "price": float(df["Close"].dropna().iloc[-1]),
                "bid": None,
                "ask": None,
                "spread": None,
                "source": "yfinance fallback",
                "warning": str(e),
            }
        except Exception:
            return {"error": str(e)}


@st.cache_data(ttl=8)
def get_intraday(symbol: str, hours: int = 720, vol_window: int = 60, horizon: int = 60):
    from stock_agent.prediction_market import (
        build_probability_calibration, blended_vol_annual, estimate_chart_signal,
        estimate_tail_dof, fetch_intraday,
    )
    df = fetch_intraday(symbol, lookback_hours=hours)
    drift, signal_details = estimate_chart_signal(df)
    tail_dof = estimate_tail_dof(df)
    vol_est = blended_vol_annual(df, window=vol_window)
    calibration = build_probability_calibration(df, horizon_minutes=horizon, vol_window=vol_window)
    return df, vol_est.annual_vol, drift, vol_est.source, tail_dof, signal_details, calibration, vol_est


@st.cache_data(ttl=300)   # funding changes every 8h — no need to refresh often
def get_funding(symbol: str):
    from stock_agent.prediction_market import fetch_funding_rate, funding_rate_to_drift
    rate, status = fetch_funding_rate(symbol)
    drift = funding_rate_to_drift(rate)
    return rate, drift, status


@st.cache_data(ttl=8)
def get_ladder(symbol: str, current_price: float, annual_vol: float, horizon: int,
               annual_drift: float = 0.0, tail_dof: float = 30.0,
               calibration=None):
    from stock_agent.prediction_market import probability_ladder, sigma_over_horizon
    rows = probability_ladder(current_price, annual_vol, horizon,
                              num_strikes=14, pct_range=0.025,
                              annual_drift=annual_drift, tail_dof=tail_dof,
                              calibration=calibration)
    sig_T = sigma_over_horizon(annual_vol, horizon)
    return rows, sig_T



@st.cache_data(ttl=600)   # 10 min — conserves Groq free-tier token quota (100k/day)
def get_ai_analysis(
    symbol: str,
    current_price: float,
    annual_vol: float,
    annual_drift: float,
    ema_cross: str,
    price_pos: str,
    vol_factor: float,
    funding_rate: float,
    tail_dof: float,
    horizon: int,
    minutes_left: float | None,
    price_1h_ago: float | None,
    price_2h_ago: float | None,
    contracts_context: str = "",
    history_context: str = "",
):
    from stock_agent.ai_analyst import analyse_btc
    try:
        report, raw_news = analyse_btc(
            symbol=symbol,
            current_price=current_price,
            annual_vol=annual_vol,
            annual_drift=annual_drift,
            ema_cross=ema_cross,
            price_pos=price_pos,
            vol_factor=vol_factor,
            funding_rate=funding_rate,
            tail_dof=tail_dof,
            horizon_minutes=horizon,
            minutes_left=minutes_left,
            price_1h_ago=price_1h_ago,
            price_2h_ago=price_2h_ago,
            contracts_context=contracts_context,
            history_context=history_context,
        )
        return report, raw_news, None
    except Exception as e:
        return None, {}, str(e)



# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Settings")
    symbol       = "BTC"
    st.text_input("Asset", value=symbol, disabled=True)
    horizon      = st.slider("Contract horizon (min)", 15, 240, 60)
    bankroll     = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
    vol_window   = st.slider("Vol estimation window (hourly candles)", 24, 120, 60)
    auto_refresh = st.checkbox("Auto-refresh every 60s", value=True)

    st.divider()
    st.caption("Data: Robinhood (live price) + yfinance (1-hour candles)")
    st.caption("Model: blended vol + calibration + momentum + Student-t tails")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(f"BTC Prediction Market AI")
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  |  Horizon: {horizon} min")

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

price_data = get_live_price(symbol)

if "error" in price_data:
    st.error(f"Robinhood error: {price_data['error']}")
    st.stop()

current_price = price_data["price"]
if price_data.get("warning"):
    st.warning(f"Robinhood unavailable; using yfinance price. {price_data['warning']}")

try:
    df_intraday, annual_vol, annual_drift, vol_source, tail_dof, signal_details, calibration, vol_est = get_intraday(
        symbol, vol_window=vol_window, horizon=horizon
    )
except Exception as e:
    st.error(f"Could not fetch intraday data: {e}")
    st.stop()

# Funding rate (Binance perp) — contrarian drift component
funding_rate, funding_drift, funding_status = get_funding(symbol)
# Blend: 75% chart/momentum signal + 25% funding rate contrarian signal
annual_drift = annual_drift * 0.75 + funding_drift * 0.25

from stock_agent.prediction_market import sigma_over_horizon
sig_T = sigma_over_horizon(annual_vol, horizon)

# Momentum label — driven by chart signal now
ema_cross = signal_details["ema_cross"]
price_pos = signal_details["price_pos"]
vol_fac   = signal_details["vol_factor"]

if annual_drift > 5:
    drift_label = f"{ema_cross} Cross"
    drift_delta = f"+{annual_drift:.1f} drift"
elif annual_drift < -5:
    drift_label = f"{ema_cross} Cross"
    drift_delta = f"{annual_drift:.1f} drift"
else:
    drift_label = "Neutral"
    drift_delta = "no signal"

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("BTC Live Price",  f"${current_price:,.2f}")
col2.metric("Bid",             f"${price_data['bid']:,.2f}" if price_data.get("bid") else "N/A")
col3.metric("Ask",             f"${price_data['ask']:,.2f}" if price_data.get("ask") else "N/A")
col4.metric(f"Vol ({vol_source})", f"{annual_vol:.1%} p.a.")
col5.metric(f"{horizon}-min Sigma", f"{sig_T:.3%}")
col6.metric("Momentum", drift_label, delta=drift_delta)

# Funding rate metric — positive = longs overcrowded (bearish), negative = shorts overcrowded (bullish)
if funding_status == "ok":
    funding_pct = f"{funding_rate*100:.4f}%"
    funding_bias = "Bearish" if funding_rate > 0.0002 else ("Bullish" if funding_rate < 0 else "Neutral")
    col7.metric("Funding (8h)", funding_pct, delta=funding_bias,
                delta_color="inverse")   # positive funding = bearish = red
else:
    col7.metric("Funding (8h)", "N/A")

dof_display = f"{tail_dof:.1f}" if tail_dof < 30 else "Normal"
calibration_display = (
    f"{calibration.samples} samples, Brier {calibration.brier:.3f}"
    if calibration.samples
    else "unavailable"
)
st.caption(
    f"Calibration: {calibration_display}  |  "
    f"Tail model: Student-t dof={dof_display}  |  "
    f"{'Fatter tails — OTM probabilities boosted' if tail_dof < 15 else 'Near-normal tails'}  |  "
    f"EMA cross: **{signal_details['ema_cross']}**  |  "
    f"Price: {signal_details['price_pos']}  |  "
    f"Vol factor: {signal_details['vol_factor']:.2f}x  |  "
    f"EMA9: ${signal_details['ema9']:,.0f}  EMA21: ${signal_details['ema21']:,.0f}  |  "
    f"Funding drift: {funding_drift:+.1f}  |  Blended drift: {annual_drift:+.1f}"
)

# ---------------------------------------------------------------------------
# Trade log — resolve outcomes + build history for AI
# ---------------------------------------------------------------------------

try:
    from stock_agent.trade_log import check_and_mark_outcomes, build_history_context
    check_and_mark_outcomes()
    _history_context = build_history_context(n=10)
except Exception:
    _history_context = ""

if "kxbtcd_markets" not in st.session_state:
    st.session_state["kxbtcd_markets"] = []

# ---------------------------------------------------------------------------
# Pre-compute KXBTCD contract evaluations (used by both right column + AI)
# ---------------------------------------------------------------------------
_kx_evaluated: list[dict] = []
_contracts_context = ""

# Recent price history for AI time-awareness
_close = df_intraday["Close"].squeeze()
_price_1h_ago = float(_close.iloc[-2]) if len(_close) >= 2 else None
_price_2h_ago = float(_close.iloc[-3]) if len(_close) >= 3 else None

# Minutes left on the nearest-expiry contract (most urgent clock)
_minutes_left: float | None = None
if st.session_state.get("kxbtcd_markets"):
    _ml_vals = [
        m["minutes_left"] for m in st.session_state["kxbtcd_markets"]
        if m.get("minutes_left") is not None
    ]
    if _ml_vals:
        _minutes_left = min(_ml_vals)

if st.session_state.get("kxbtcd_markets"):
    from stock_agent.prediction_market import evaluate_range_contract
    from stock_agent.trading import TradeParams, Signal

    _tp_kx = TradeParams(bankroll=bankroll, kelly_fraction=0.25,
                         max_position_pct=0.10, min_edge_pct=0.03,
                         transaction_cost_pct=0.02)
    for _m in st.session_state["kxbtcd_markets"]:
        _price = _m.get("display_price_cents")
        if _price is None:
            continue
        try:
            _d = evaluate_range_contract(
                symbol, _m.get("floor_strike"), _m.get("cap_strike"),
                float(_price), current_price, annual_vol,
                horizon, _tp_kx, annual_drift, tail_dof,
            )
            _action = "BUY YES" if _d.signal == Signal.BUY else (
                "BUY NO" if _d.signal == Signal.SELL else "HOLD"
            )
            _kx_evaluated.append({
                "ticker":      _m["ticker"],
                "floor":       _m.get("floor_strike"),
                "minutes_left": _m.get("minutes_left", "?"),
                "kalshi_c":    float(_price),
                "fair_pct":    _d.fair_prob,
                "edge_pct":    _d.net_edge,
                "action":      _action,
                "sized":       _d.sized_dollars,
                "decision":    _d,
            })
        except Exception:
            continue

    if _kx_evaluated:
        rows_txt = ["--- KXBTCD contract table (model vs Kalshi market price) ---",
                    f"{'Floor Strike':>12}  {'Kalshi':>7}  {'Fair':>7}  {'Edge':>7}  {'Min Left':>8}  Action"]
        for r in _kx_evaluated:
            rows_txt.append(
                f"  ${r['floor']:>10,.0f}  {r['kalshi_c']:>5.1f}c  "
                f"{r['fair_pct']:>6.1%}  {r['edge_pct']:>+6.1%}  "
                f"{str(r['minutes_left']):>8}  {r['action']}"
            )
        rows_txt.append("--- End of contract table ---")
        _contracts_context = "\n".join(rows_txt)

st.divider()

# ---------------------------------------------------------------------------
# Main layout: chart left, ladder + evaluator right
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2], gap="large")

# ---- Price chart -----------------------------------------------------------
with left:
    st.subheader("Price Action (1-hour candles)")

    # EMAs
    df_chart = df_intraday.copy()
    close = df_chart["Close"].squeeze()
    df_chart["EMA9"]  = close.ewm(span=9,  adjust=False).mean()
    df_chart["EMA21"] = close.ewm(span=21, adjust=False).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart["Open"].squeeze(),
        high=df_chart["High"].squeeze(),
        low=df_chart["Low"].squeeze(),
        close=close,
        increasing=dict(line=dict(color="#00d4aa"), fillcolor="#00d4aa"),
        decreasing=dict(line=dict(color="#ff6b6b"), fillcolor="#ff6b6b"),
        name="BTC",
    ), row=1, col=1)

    # EMA 9
    fig.add_trace(go.Scatter(
        x=df_chart.index,
        y=df_chart["EMA9"].squeeze(),
        mode="lines",
        line=dict(color="#ffd700", width=1.2),
        name="EMA 9",
    ), row=1, col=1)

    # EMA 21
    fig.add_trace(go.Scatter(
        x=df_chart.index,
        y=df_chart["EMA21"].squeeze(),
        mode="lines",
        line=dict(color="#ff8c00", width=1.2),
        name="EMA 21",
    ), row=1, col=1)

    # Live price line
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="white",
        annotation_text=f"Live ${current_price:,.0f}",
        annotation_position="top right",
        row=1, col=1,
    )

    # Volume bars (green if candle up, red if down)
    vol_colors = [
        "#00d4aa" if float(c) >= float(o) else "#ff6b6b"
        for c, o in zip(df_chart["Close"].squeeze(), df_chart["Open"].squeeze())
    ]
    fig.add_trace(go.Bar(
        x=df_chart.index,
        y=df_chart["Volume"].squeeze(),
        marker_color=vol_colors,
        name="Volume",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        xaxis=dict(showgrid=False, color="#aaa", rangeslider_visible=False),
        xaxis2=dict(showgrid=False, color="#aaa"),
        yaxis=dict(showgrid=True, gridcolor="#222", color="#aaa"),
        yaxis2=dict(showgrid=False, color="#aaa", title="Vol"),
        font=dict(color="#ccc"),
        legend=dict(orientation="h", x=0, y=1.04, font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability distribution chart
    st.subheader("Fair Probability Distribution")
    ladder_rows, _ = get_ladder(symbol, current_price, annual_vol, horizon,
                                annual_drift, tail_dof, calibration)
    strikes   = [r["strike"] for r in ladder_rows]
    p_yes_vals = [r["fair_yes"] * 100 for r in ladder_rows]

    fig2 = go.Figure()
    colors = ["#00d4aa" if s <= current_price else "#ff6b6b" for s in strikes]
    fig2.add_trace(go.Bar(
        x=strikes,
        y=p_yes_vals,
        marker_color=colors,
        name="P(Yes) %",
    ))
    fig2.add_vline(x=current_price, line_dash="dot", line_color="yellow",
                   annotation_text="Spot")
    fig2.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        xaxis=dict(showgrid=False, color="#aaa", tickformat=",.0f"),
        yaxis=dict(showgrid=True, gridcolor="#333", color="#aaa",
                   title="P(Yes) %", range=[0, 100]),
        font=dict(color="#ccc"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -- AI Analysis ----------------------------------------------------------
    st.subheader("AI Analysis (Groq + Live News)")
    st.caption("Llama-3.3-70b reads Fear & Greed + CryptoPanic headlines — refreshes every 10 min (auto-fallback to 8b-instant on rate limit)")

    report, raw_news, ai_error = get_ai_analysis(
        symbol, current_price, annual_vol, annual_drift,
        signal_details["ema_cross"], signal_details["price_pos"],
        signal_details["vol_factor"], funding_rate, tail_dof, horizon,
        _minutes_left, _price_1h_ago, _price_2h_ago,
        _contracts_context, _history_context,
    )

    if ai_error:
        st.error(f"AI analyst unavailable: {ai_error}")
    elif report:
        fg = raw_news.get("fear_greed", {})
        news_items = raw_news.get("news", [])

        # Bias badge
        bias_colors = {
            "strongly_bullish": "#00d4aa",
            "bullish":          "#4ade80",
            "neutral":          "#94a3b8",
            "bearish":          "#f87171",
            "strongly_bearish": "#ef4444",
        }
        bias_col, conf_col, fg_col = st.columns(3)
        bias_col.metric(
            "AI Bias",
            report.drift_bias.replace("_", " ").title(),
            delta=f"drift {report.drift_nudge:+.0%} p.a.",
            delta_color="normal" if report.drift_nudge >= 0 else "inverse",
        )
        conf_col.metric("Confidence", report.confidence.title())
        if fg:
            fg_delta = "Greed" if fg["value"] > 60 else ("Fear" if fg["value"] < 40 else "Neutral")
            fg_col.metric("Fear & Greed", f"{fg['value']} — {fg['classification']}", delta=fg_delta,
                          delta_color="inverse" if fg["value"] > 60 else "normal")

        if report.contract_action:
            action_upper = report.contract_action.upper()
            if action_upper.startswith("SKIP"):
                st.warning(f"**AI: {report.contract_action}**")
            elif "BUY YES" in action_upper:
                st.success(f"**AI: {report.contract_action}**")
            elif "BUY NO" in action_upper:
                st.error(f"**AI: {report.contract_action}**")
            else:
                st.info(f"**AI: {report.contract_action}**")

        # Log button — pick the contract with highest absolute edge that has a BUY signal
        _best_contract = next(
            (r for r in sorted(_kx_evaluated, key=lambda r: abs(r["edge_pct"]), reverse=True)
             if r["action"] in ("BUY YES", "BUY NO")),
            None,
        )
        if _best_contract and report.contract_action and not report.contract_action.upper().startswith("SKIP"):
            if st.button("Log AI Recommendation", type="primary", use_container_width=True):
                try:
                    from stock_agent.trade_log import log_recommendation
                    # minutes_left may be stored as "?" string — coerce to float or None
                    _ml_raw = _best_contract.get("minutes_left")
                    try:
                        _ml = float(_ml_raw) if _ml_raw not in (None, "?", "") else None
                    except (TypeError, ValueError):
                        _ml = None
                    _close_time = next(
                        (m.get("close_time", "") for m in st.session_state.get("kxbtcd_markets", [])
                         if m.get("ticker") == _best_contract["ticker"]), ""
                    )
                    _row_id = log_recommendation(
                        ticker=_best_contract["ticker"],
                        floor_strike=_best_contract.get("floor"),
                        close_time=_close_time,
                        kalshi_price_c=_best_contract["kalshi_c"],
                        fair_prob=_best_contract["fair_pct"],
                        edge=_best_contract["edge_pct"],
                        ai_action=report.contract_action[:50],
                        ai_confidence=report.confidence,
                        ai_bias=report.drift_bias,
                        minutes_left=_ml,
                        btc_price=current_price,
                    )
                    st.success(f"Logged {_best_contract['ticker']} — id: {_row_id}")
                except Exception as _log_err:
                    st.error(f"Log failed: {_log_err}")

        with st.expander("Analyst report", expanded=False):
            st.write(f"**Trend:** {report.fundamental_summary}")
            st.write(f"**Macro:** {report.macro_summary}")
            if report.key_catalysts:
                st.write("**Catalysts:** " + "  •  ".join(report.key_catalysts))
            if report.key_risks:
                st.write("**Risks:** " + "  •  ".join(report.key_risks))

        if news_items:
            with st.expander(f"Live headlines ({len(news_items)})", expanded=False):
                for item in news_items:
                    votes = item["votes_positive"] - item["votes_negative"]
                    sentiment_icon = "🟢" if votes > 2 else ("🔴" if votes < -2 else "⚪")
                    st.write(f"{sentiment_icon} {item['title']}")


# ---- Probability ladder + contract evaluator -------------------------------
with right:
    st.subheader("Probability Ladder")
    st.caption("Compare these to Robinhood's contract prices to find edge.")

    ladder_rows, _ = get_ladder(symbol, current_price, annual_vol, horizon,
                                annual_drift, tail_dof, calibration)
    df_ladder = pd.DataFrame(ladder_rows)
    df_ladder["Strike"]  = df_ladder["strike"].map(lambda x: f"${x:,.2f}")
    df_ladder["Raw"]     = df_ladder["raw_fair_yes"].map(lambda x: f"{x:.1%}")
    df_ladder["Cal"]     = df_ladder["fair_yes"].map(lambda x: f"{x:.1%}")
    df_ladder["P(No)"]   = df_ladder["fair_no"].map(lambda x: f"{x:.1%}")
    df_ladder["ATM"]     = df_ladder["strike"].map(
        lambda x: "<<" if abs(x - current_price) / current_price < 0.002 else ""
    )

    st.dataframe(
        df_ladder[["Strike", "Raw", "Cal", "P(No)", "ATM"]],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    st.divider()
    st.subheader("Evaluate Kalshi")

    # -- KXBTCD hourly event (fastest path for 1-hour trading) ---------------
    st.caption("**Quick: KXBTCD hourly event** — fetches all near-ATM strikes for a specific hour")
    kx_col1, kx_col2 = st.columns([1, 2])
    kxbtcd_hour = kx_col1.number_input("Expiry hour (ET, 24h)", min_value=0, max_value=23, value=11)
    if kx_col2.button("Load KXBTCD ATM Contracts", use_container_width=True):
        try:
            from stock_agent.kalshi_market import find_kxbtcd_atm_markets
            st.session_state["kxbtcd_markets"] = find_kxbtcd_atm_markets(
                current_price, hour_et=kxbtcd_hour
            )
            if not st.session_state["kxbtcd_markets"]:
                st.warning(f"No open KXBTCD markets found for {kxbtcd_hour:02d}:00 ET today.")
        except Exception as e:
            st.error(f"KXBTCD load failed: {e}")

    if _kx_evaluated:
        kx_rows = [
            {
                "Ticker":   r["ticker"],
                "Floor":    f"${r['floor'] or 0:,.0f}",
                "Min Left": f"{r['minutes_left']}",
                "Kalshi":   f"{r['kalshi_c']:.1f}c",
                "Fair":     f"{r['fair_pct']:.1%}",
                "Edge":     f"{r['edge_pct']:+.1%}",
                "Action":   r["action"],
            }
            for r in _kx_evaluated
        ]
        st.dataframe(pd.DataFrame(kx_rows), use_container_width=True, hide_index=True)
    elif st.session_state.get("kxbtcd_markets"):
        st.info("KXBTCD markets loaded but no orderbook prices available yet.")


# ---------------------------------------------------------------------------
# Trade log history
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Trade Log & AI Track Record")

try:
    from stock_agent.trade_log import get_recent_history, accuracy_stats

    stats = accuracy_stats()
    if stats["total"] > 0:
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("Resolved Trades", stats["total"])
        stat_c2.metric("Correct", stats["correct"])
        win_pct = f"{stats['win_rate']:.0%}" if stats["win_rate"] is not None else "N/A"
        stat_c3.metric("Win Rate", win_pct)
        avg_edge_c = (
            f"{stats['avg_edge_correct']:+.1%}" if stats["avg_edge_correct"] is not None else "N/A"
        )
        stat_c4.metric("Avg Edge (correct trades)", avg_edge_c)

    history_rows = get_recent_history(n=20)
    if history_rows:
        def _fmt_bool(val: str) -> str:
            if val in ("True", "true", "1"):
                return "YES"
            if val in ("False", "false", "0"):
                return "NO"
            return "—"

        log_df = pd.DataFrame([
            {
                "Time (UTC)":  r.get("logged_at", "")[:16].replace("T", " "),
                "Ticker":      r.get("ticker", ""),
                "Floor":       f"${float(r['floor_strike']):,.0f}" if r.get("floor_strike") else "—",
                "Kalshi":      f"{float(r['kalshi_price_c']):.1f}c" if r.get("kalshi_price_c") else "—",
                "Fair":        f"{float(r['fair_prob']):.1%}" if r.get("fair_prob") else "—",
                "Edge":        f"{float(r['edge']):+.1%}" if r.get("edge") else "—",
                "AI Action":   r.get("ai_action", "")[:20],
                "Conf":        r.get("ai_confidence", ""),
                "Min Left":    r.get("minutes_left", "—"),
                "BTC $":       f"${float(r['btc_price']):,.0f}" if r.get("btc_price") else "—",
                "Resolved":    _fmt_bool(r.get("resolved", "")),
                "Result":      _fmt_bool(r.get("resolved_yes", "")),
                "Correct":     _fmt_bool(r.get("ai_correct", "")),
            }
            for r in history_rows
        ])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades logged yet. Click 'Log AI Recommendation' after loading KXBTCD contracts.")
except Exception as _tl_err:
    st.warning(f"Trade log unavailable: {_tl_err}")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(60)
    st.rerun()
