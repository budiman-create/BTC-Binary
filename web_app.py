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


@st.cache_data(ttl=8)
def get_kalshi_quote(ticker: str):
    from stock_agent.kalshi_market import get_quote

    q = get_quote(ticker)
    return {
        "ticker": q.ticker,
        "title": q.title,
        "strike": q.strike,
        "floor_strike": q.floor_strike,
        "cap_strike": q.cap_strike,
        "strike_type": q.strike_type,
        "yes_bid_cents": q.yes_bid_cents,
        "yes_ask_cents": q.yes_ask_cents,
        "yes_mid_cents": q.yes_mid_cents,
        "display_price_cents": q.display_price_cents,
    }


@st.cache_data(ttl=60)
def get_kalshi_btc_markets(max_expiry_hours: float | None = None):
    from stock_agent.kalshi_market import find_btc_markets

    return find_btc_markets(max_expiry_hours=max_expiry_hours)


@st.cache_data(ttl=180)   # refresh every 3 min — news doesn't change every second
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
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=True)

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

if "kxbtcd_markets" not in st.session_state:
    st.session_state["kxbtcd_markets"] = []

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
    st.caption("Llama-3.3-70b reads Fear & Greed + CryptoPanic headlines — refreshes every 3 min")

    report, raw_news, ai_error = get_ai_analysis(
        symbol, current_price, annual_vol, annual_drift,
        signal_details["ema_cross"], signal_details["price_pos"],
        signal_details["vol_factor"], funding_rate, tail_dof, horizon,
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

        with st.expander("Analyst report", expanded=True):
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

    if st.session_state.get("kxbtcd_markets"):
        from stock_agent.prediction_market import evaluate_range_contract
        from stock_agent.trading import TradeParams, Signal

        tp_kx = TradeParams(bankroll=bankroll, kelly_fraction=0.25,
                            max_position_pct=0.10, min_edge_pct=0.03,
                            transaction_cost_pct=0.02)
        kx_rows = []
        for m in st.session_state["kxbtcd_markets"]:
            price = m.get("display_price_cents")
            if price is None:
                continue
            try:
                d = evaluate_range_contract(
                    symbol, m.get("floor_strike"), m.get("cap_strike"),
                    float(price), current_price, annual_vol,
                    horizon, tp_kx, annual_drift, tail_dof,
                )
                action = "BUY YES" if d.signal == Signal.BUY else (
                    "BUY NO" if d.signal == Signal.SELL else "HOLD"
                )
                kx_rows.append({
                    "Ticker":   m["ticker"],
                    "Floor":    f"${m.get('floor_strike') or 0:,.0f}",
                    "Min Left": f"{m.get('minutes_left', '?')}",
                    "Kalshi":   f"{price:.1f}c",
                    "Fair":     f"{d.fair_prob:.1%}",
                    "Edge":     f"{d.net_edge:+.1%}",
                    "Action":   action,
                })
            except Exception:
                continue
        if kx_rows:
            st.dataframe(pd.DataFrame(kx_rows), use_container_width=True, hide_index=True)
        else:
            st.info("KXBTCD markets loaded but no orderbook prices available yet.")

    st.divider()
    kalshi_horizon_filter = st.selectbox(
        "Show markets expiring within",
        ["1 hour", "2 hours", "6 hours", "Any"],
        index=1,
    )
    _horizon_map = {"1 hour": 1.0, "2 hours": 2.0, "6 hours": 6.0, "Any": None}
    _kalshi_max_h = _horizon_map[kalshi_horizon_filter]

    if st.button("Find Open BTC Kalshi Markets", use_container_width=True):
        try:
            st.session_state["kalshi_btc_markets"] = get_kalshi_btc_markets(
                max_expiry_hours=_kalshi_max_h
            )
            if not st.session_state["kalshi_btc_markets"]:
                st.info(f"No open BTC markets expiring within {kalshi_horizon_filter}. Try a wider filter.")
        except Exception as e:
            st.error(f"Could not search Kalshi BTC markets: {e}")

    if st.session_state.get("kalshi_btc_markets"):
        from stock_agent.prediction_market import evaluate_range_contract
        from stock_agent.trading import TradeParams, Signal

        sorted_markets = sorted(
            st.session_state["kalshi_btc_markets"],
            key=lambda m: (
                abs((m.get("strike") or current_price) - current_price),
                m.get("ticker", ""),
            ),
        )
        quoted_markets = get_kalshi_btc_quotes(sorted_markets[:24])
        scan_rows = []
        tp_scan = TradeParams(
            bankroll=bankroll,
            kelly_fraction=0.25,
            max_position_pct=0.10,
            min_edge_pct=0.03,
            transaction_cost_pct=0.02,
        )
        for market in quoted_markets[:12]:
            try:
                price = market["display_price_cents"]
                if price is None:
                    continue
                d_scan = evaluate_range_contract(
                    symbol,
                    market.get("floor_strike"),
                    market.get("cap_strike"),
                    float(price),
                    current_price,
                    annual_vol,
                    horizon,
                    tp_scan,
                    annual_drift,
                    tail_dof,
                )
                action = "BUY YES" if d_scan.signal == Signal.BUY else (
                    "BUY NO" if d_scan.signal == Signal.SELL else "HOLD"
                )
                scan_rows.append({
                    "Ticker": market["ticker"],
                    "Contract": d_scan.description or market.get("title", ""),
                    "Kalshi": f"{price:.1f}c",
                    "Fair": f"{d_scan.fair_prob:.1%}",
                    "Edge": f"{d_scan.net_edge:+.1%}",
                    "Action": action,
                })
            except Exception:
                continue
        if scan_rows:
            st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Found BTC markets, but no usable orderbook prices yet.")

    selected_kalshi_ticker = None
    if st.session_state.get("kalshi_btc_markets"):
        options = sorted(
            st.session_state["kalshi_btc_markets"],
            key=lambda m: (
                abs((m.get("strike") or current_price) - current_price),
                m.get("ticker", ""),
            ),
        )
        labels = [
            f"{m['ticker']} | "
            f"{('$' + format(m['strike'], ',.0f')) if m.get('strike') is not None else 'range'} | "
            f"{str(round(m['minutes_left'])) + ' min' if m.get('minutes_left') is not None else m.get('close_time', '')}"
            for m in options
        ]
        selected_label = st.selectbox("Open BTC markets", labels)
        selected_kalshi_ticker = options[labels.index(selected_label)]["ticker"]

    kalshi_ticker = st.text_input(
        "Kalshi market ticker",
        value=selected_kalshi_ticker or "",
        placeholder="KXBTC...",
        key="kalshi_ticker",
    )
    active_kalshi_ticker = kalshi_ticker or selected_kalshi_ticker
    if st.button("Fetch Kalshi Price", use_container_width=True) and active_kalshi_ticker:
        from stock_agent.prediction_market import evaluate_contract
        from stock_agent.trading import TradeParams, Signal

        try:
            q = get_kalshi_quote(active_kalshi_ticker)
            k_price = q["display_price_cents"]
            if k_price is None:
                st.error("Kalshi orderbook has no usable Yes price.")
            else:
                st.caption(q["title"])
                st.write(
                    f"Yes bid/ask/mid: "
                    f"{q['yes_bid_cents'] if q['yes_bid_cents'] is not None else 'N/A'} / "
                    f"{q['yes_ask_cents'] if q['yes_ask_cents'] is not None else 'N/A'} / "
                    f"{q['yes_mid_cents'] if q['yes_mid_cents'] is not None else 'N/A'} cents"
                )
                tp = TradeParams(
                    bankroll=bankroll,
                    kelly_fraction=0.25,
                    max_position_pct=0.10,
                    min_edge_pct=0.03,
                    transaction_cost_pct=0.02,
                )
                if q["floor_strike"] is not None or q["cap_strike"] is not None:
                    d = evaluate_range_contract(symbol, q["floor_strike"], q["cap_strike"],
                                                float(k_price), current_price, annual_vol,
                                                horizon, tp, annual_drift, tail_dof)
                else:
                    k_strike = q["strike"]
                    if k_strike is None:
                        st.error("Cannot determine strike for this market. Use a KXBTCD range contract instead.")
                        st.stop()
                    d = evaluate_contract(symbol, float(k_strike), float(k_price),
                                          current_price, annual_vol, horizon, tp,
                                          annual_drift, tail_dof, calibration)
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Raw P(Yes)", f"{(d.raw_fair_prob or d.fair_prob):.1%}")
                r2.metric("Cal P(Yes)", f"{d.fair_prob:.1%}")
                r3.metric("Kalshi Mid", f"{k_price:.1f}c")
                r4.metric("Net Edge", f"{d.net_edge:+.1%}")
                if d.signal != Signal.HOLD:
                    action = "BUY YES" if d.signal == Signal.BUY else "BUY NO"
                    st.success(f"**{action}** - edge {d.net_edge:+.1%}, size ${d.sized_dollars:,.0f}")
                else:
                    st.warning(f"**HOLD** - edge {d.raw_edge:+.1%} is too thin after spread")
        except Exception as e:
            st.error(f"Could not fetch Kalshi market: {e}")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(30)
    st.rerun()
