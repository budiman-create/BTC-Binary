"""
BTC Prediction Market AI — Streamlit web dashboard.

Run:
    streamlit run web_app.py

Features:
  - Live BTC price + bid/ask from Robinhood
  - Real-time intraday volatility from 1-min data
  - Probability ladder: fair Yes/No odds at strikes around spot
  - Contract evaluator: paste strike + Yes price → instant BUY/SELL/HOLD signal
  - 1-min price chart (last 2 hours)
  - Auto-refresh every 30 seconds
"""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
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

@st.cache_data(ttl=25)
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
        return {"error": str(e)}


@st.cache_data(ttl=25)
def get_intraday(symbol: str, hours: int = 3):
    from stock_agent.prediction_market import fetch_intraday, realised_vol_annual
    df = fetch_intraday(symbol, lookback_hours=hours)
    vol = realised_vol_annual(df, window=60)
    return df, vol


@st.cache_data(ttl=25)
def get_ladder(symbol: str, current_price: float, annual_vol: float, horizon: int):
    from stock_agent.prediction_market import probability_ladder, sigma_over_horizon
    rows = probability_ladder(current_price, annual_vol, horizon,
                              num_strikes=14, pct_range=0.025)
    sig_T = sigma_over_horizon(annual_vol, horizon)
    return rows, sig_T


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Settings")
    symbol    = st.selectbox("Asset", ["BTC", "ETH", "SOL", "DOGE"], index=0)
    horizon   = st.slider("Contract horizon (min)", 5, 30, 15)
    bankroll  = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
    vol_window = st.slider("Vol estimation window (min)", 10, 120, 60)
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=True)

    st.divider()
    st.caption("Data: Robinhood (live price) + yfinance (1-min candles)")
    st.caption("Model: Log-normal GBM, zero drift, realised vol")

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

try:
    df_intraday, annual_vol = get_intraday(symbol)
except Exception as e:
    st.error(f"Could not fetch intraday data: {e}")
    st.stop()

from stock_agent.prediction_market import sigma_over_horizon
sig_T = sigma_over_horizon(annual_vol, horizon)

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("BTC Live Price",  f"${current_price:,.2f}")
col2.metric("Bid",             f"${price_data['bid']:,.2f}" if price_data.get("bid") else "N/A")
col3.metric("Ask",             f"${price_data['ask']:,.2f}" if price_data.get("ask") else "N/A")
col4.metric("Intraday Vol",    f"{annual_vol:.1%} p.a.")
col5.metric(f"{horizon}-min Sigma", f"{sig_T:.3%}")

st.divider()

# ---------------------------------------------------------------------------
# Main layout: chart left, ladder + evaluator right
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2], gap="large")

# ---- Price chart -----------------------------------------------------------
with left:
    st.subheader("Price (1-min candles)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_intraday.index,
        y=df_intraday["Close"].squeeze(),
        mode="lines",
        line=dict(color="#00d4aa", width=1.5),
        name="BTC",
    ))
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Live ${current_price:,.0f}",
        annotation_position="top right",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        xaxis=dict(showgrid=False, color="#aaa"),
        yaxis=dict(showgrid=True, gridcolor="#333", color="#aaa"),
        font=dict(color="#ccc"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability distribution chart
    st.subheader("Fair Probability Distribution")
    ladder_rows, _ = get_ladder(symbol, current_price, annual_vol, horizon)
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


# ---- Probability ladder + contract evaluator -------------------------------
with right:
    st.subheader("Probability Ladder")
    st.caption("Compare these to Robinhood's contract prices to find edge.")

    ladder_rows, _ = get_ladder(symbol, current_price, annual_vol, horizon)
    df_ladder = pd.DataFrame(ladder_rows)
    df_ladder["Strike"]  = df_ladder["strike"].map(lambda x: f"${x:,.2f}")
    df_ladder["P(Yes)"]  = df_ladder["fair_yes"].map(lambda x: f"{x:.1%}")
    df_ladder["P(No)"]   = df_ladder["fair_no"].map(lambda x: f"{x:.1%}")
    df_ladder["ATM"]     = df_ladder["strike"].map(
        lambda x: "<<" if abs(x - current_price) / current_price < 0.002 else ""
    )

    st.dataframe(
        df_ladder[["Strike", "P(Yes)", "P(No)", "ATM"]],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    st.divider()
    st.subheader("Evaluate a Contract")
    st.caption("Enter the strike and Yes price from Robinhood Predict.")

    c1, c2 = st.columns(2)
    strike_input    = c1.number_input("Strike price ($)", value=float(round(current_price, 2)),
                                       step=0.01, format="%.2f")
    yes_price_input = c2.number_input("Yes price (cents)", min_value=1, max_value=99,
                                       value=50, step=1)

    if st.button("Analyze Contract", type="primary", use_container_width=True):
        from stock_agent.prediction_market import evaluate_contract
        from stock_agent.trading import TradeParams, Signal

        tp = TradeParams(
            bankroll=bankroll,
            kelly_fraction=0.25,
            max_position_pct=0.10,
            min_edge_pct=0.03,
            transaction_cost_pct=0.02,
        )
        d = evaluate_contract(symbol, strike_input, yes_price_input,
                              current_price, annual_vol, horizon, tp)

        # Result card
        if d.signal == Signal.BUY:
            color = "normal"
            action = f"BUY YES at {yes_price_input}c"
        elif d.signal == Signal.SELL:
            color = "inverse"
            action = f"BUY NO at {100 - yes_price_input}c"
        else:
            color = "off"
            action = "HOLD — edge too thin"

        r1, r2, r3 = st.columns(3)
        r1.metric("Fair P(Yes)", f"{d.fair_prob:.1%}")
        r2.metric("Net Edge",    f"{d.net_edge:+.1%}")
        r3.metric("Size",        f"${d.sized_dollars:,.0f}")

        if d.signal != Signal.HOLD:
            st.success(f"**{action}** — edge {d.net_edge:+.1%} vs market")
        else:
            st.warning(f"**HOLD** — edge {d.raw_edge:+.1%} is too thin after spread")

    # ---- Batch scan --------------------------------------------------------
    st.divider()
    st.subheader("Scan Multiple Contracts")
    st.caption('Format: "strike:yes_price, strike:yes_price"')

    batch_input = st.text_input(
        "Contracts",
        placeholder="76385:61, 76500:45, 76200:72",
    )

    if st.button("Scan All", use_container_width=True) and batch_input:
        from stock_agent.prediction_market import scan_contracts
        from stock_agent.trading import TradeParams, Signal

        tp = TradeParams(bankroll=bankroll, kelly_fraction=0.25,
                         max_position_pct=0.10, min_edge_pct=0.03,
                         transaction_cost_pct=0.02)
        try:
            contracts = {
                float(item.split(":")[0].strip()): float(item.split(":")[1].strip())
                for item in batch_input.split(",")
            }
            decisions = scan_contracts(symbol, current_price, annual_vol,
                                       contracts, horizon, tp)

            rows = []
            for d in decisions:
                rows.append({
                    "Strike":   f"${d.strike:,.2f}",
                    "Fair":     f"{d.fair_prob:.1%}",
                    "Market":   f"{d.contract_price_pct:.1%}",
                    "Edge":     f"{d.net_edge:+.1%}",
                    "Signal":   d.signal.value,
                    "Action":   f"BUY {'YES' if d.signal==Signal.BUY else 'NO'}  ${d.sized_dollars:,.0f}" if d.signal != Signal.HOLD else "HOLD",
                })

            df_scan = pd.DataFrame(rows)

            def highlight_signal(row):
                if "BUY" in row["Signal"]:
                    return ["background-color: #0d3d2e"] * len(row)
                if "SELL" in row["Signal"]:
                    return ["background-color: #3d0d0d"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_scan.style.apply(highlight_signal, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Parse error: {e}. Use format: 76385:61, 76500:45")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(30)
    st.rerun()
