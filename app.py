import streamlit as st
import pandas as pd, numpy as np, time, io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

st.set_page_config(layout='wide', page_title='Streamlit Crypto TA Analyzer — Interactive v1.5')
st.title('Streamlit Crypto TA Analyzer — Interactive v1.5')

# --- Sidebar settings ---
st.sidebar.header('Data & Settings')
use_cg = st.sidebar.checkbox('Use CoinGecko (public API)', value=True)
use_cmc = st.sidebar.checkbox('Use CoinMarketCap Pro (requires API key)', value=False)
cmc_key = st.sidebar.text_input('CMC API Key (optional)', type='password')
days_opt = st.sidebar.selectbox('Days to fetch', options=[1,7,30,90,365], index=2)
n_sims = st.sidebar.slider('Monte Carlo sims', min_value=500, max_value=20000, value=5000, step=500)

# --- Token input section ---
if "token_text" not in st.session_state:
    st.session_state["token_text"] = "solana"

token_text = st.sidebar.text_input(
    "Token (CoinGecko id)",
    value=st.session_state["token_text"],
    key="token_text_input"
)

# --- CSV upload ---
st.sidebar.subheader("Upload CSV (optional) to use your own data:")
uploaded_file = st.sidebar.file_uploader("Upload CSV with OHLCV", type=["csv"], label_visibility="collapsed")

# --- Top coin selector ---
with st.sidebar.expander("Pick from top coins"):
    try:
        coins = fetch_coingecko_list()
        names = [f"{c['id']} ({c['symbol']})" for c in coins]
        choice = st.selectbox("Top coins", options=names, index=0, key="coin_choice")
        if st.button("Use selected coin"):
            chosen = choice.split(" ")[0]
            st.session_state["token_text"] = chosen
            st.session_state["token_text_input"] = chosen
            st.rerun()
    except Exception as e:
        st.warning(f"Could not fetch coin list:\n{e}")

# --- Fetch & Analyze ---
if st.button("Fetch & Analyze"):
    with st.spinner("Fetching or loading data..."):
        if uploaded_file is not None:
    # --- Improved CSV handling (handles semicolons, alternate names, flexible dates) ---
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")  # auto-detect delimiter
    except Exception:
        df = pd.read_csv(uploaded_file, sep=";")  # fallback for semicolon-separated files

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename time/date columns → timestamp
    if "timestamp" not in df.columns:
        for alt in ["time", "date", "datetime"]:
            if alt in df.columns:
                df.rename(columns={alt: "timestamp"}, inplace=True)
                break
        else:
            st.error("❌ No 'timestamp' column found in uploaded CSV. Please include a time column.")
            st.stop()

    # Convert timestamp to datetime
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)
    except Exception:
        st.warning("⚠️ Trying with dayfirst=True for timestamp parsing...")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True, dayfirst=True)

    # Drop invalid timestamps
    bad_rows = df["timestamp"]._

        else:
            fetch_days = max(days_opt, 90)
            try:
                if use_cmc and cmc_key.strip():
                    end = datetime.utcnow().date()
                    start = end - timedelta(days=fetch_days)
                    df = fetch_cmc_pro(token_text.upper(), start.isoformat(), end.isoformat(), api_key=cmc_key)
                else:
                    df = fetch_coingecko_daily(token_text, days=fetch_days)
            except Exception as e:
                st.error("Error fetching data: " + str(e))
                st.stop()

    if df is None or df.empty:
        st.error("No data returned. Try another token or upload a CSV.")
        st.stop()

    df["MA20"] = sma(df["close"], 20)
    df["MA50"] = sma(df["close"], 50)
    df["RSI14"] = compute_rsi(df["close"], 14)
    macd, sig, hist = compute_macd(df["close"])
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd, sig, hist
    support, resistance = simple_support_resistance(df, window=20)

    window_30 = df.tail(30) if len(df) >= 30 else df
    st_high, st_low = float(window_30["high"].max()), float(window_30["low"].min())
    st_retr, st_ext = fib_levels(st_high, st_low)
    lt_high, lt_low = float(df["high"].max()), float(df["low"].min())
    lt_retr, lt_ext = fib_levels(lt_high, lt_low)

    probs = compute_probs_from_df(df, support, resistance, horizons=[1,7,30,90,365], n_sims=n_sims)
    mapping = {1:"1D",7:"7D",30:"1M",90:"90D",365:"365D"}
    selected = days_opt
    sel_probs = probs.get(selected, {})
    bull_target = lt_ext.get("161.8%", None)
    bear_target = max(0.0, st_low - (st_high - st_low)*0.618)
    base_target = (st_high + st_low) / 2

    st.subheader(f"💠 Token: {token_text.capitalize()} — {token_text.upper()}")
    st.markdown(f"💰 **Latest close:** ${df['close'].iloc[-1]:.4f} USD")

    scenario = pd.DataFrame([{
        "Horizon": mapping.get(selected, str(selected) + "d"),
        "Prob Bull": sel_probs.get("Prob Bull", None),
        "Prob Base": sel_probs.get("Prob Base", None),
        "Prob Bear": sel_probs.get("Prob Bear", None),
        "Bull target": bull_target,
        "Base target": base_target,
        "Bear target": bear_target,
    }])
    st.subheader("Scenario probabilities & targets (selected horizon)")
    st.dataframe(scenario.style.format("{:.4f}"))

    chart_window = max(60, days_opt * 3)
    df_chart = df.tail(chart_window)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.2, 0.25],
                        specs=[[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]],
                        subplot_titles=("Price with Fibonacci, MA20/50, Support & Resistance",
                                        "RSI (Relative Strength Index)",
                                        "MACD (Moving Average Convergence Divergence)"))

    fig.add_trace(go.Candlestick(x=df_chart["timestamp"], open=df_chart["open"], high=df_chart["high"],
                                 low=df_chart["low"], close=df_chart["close"], name="Candles"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA20"], name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA50"], name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                             y=[support, support], mode="lines", name="Support"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                             y=[resistance, resistance], mode="lines", name="Resistance"), row=1, col=1)
    for k, v in st_retr.items():
        fig.add_hline(y=v, line_dash="dot", annotation_text=f"ST {k} {v:.2f}", annotation_position="right", row=1, col=1)
    for k, v in lt_retr.items():
        fig.add_hline(y=v, line_dash="dash", annotation_text=f"LT {k} {v:.2f}", annotation_position="left", row=1, col=1)

    fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["RSI14"], name="RSI14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="lightgray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="lightgray", row=2, col=1)

    fig.add_trace(go.Bar(x=df_chart["timestamp"], y=df_chart["MACD_hist"], name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD"], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD_sig"], name="Signal"), row=3, col=1)

    fig.update_layout(height=950, showlegend=True, legend_tracegroupgap=5)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monte Carlo summary (all horizons)")
    mc_summary = pd.DataFrame.from_dict({str(k)+"d": v for k, v in probs.items()}, orient="index")
    st.dataframe(mc_summary.style.format("{:.4f}"))

    if st.button("Export PDF report"):
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Crypto TA Report - {token_text}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Latest close: ${df['close'].iloc[-1]:.4f} USD", styles["Normal"]),
            Spacer(1, 12),
        ]
        for k, v in st_retr.items():
            story.append(Paragraph(f"ST {k}: {v:.4f}", styles["Normal"]))
        story.append(Spacer(1, 12))
        for k, v in lt_retr.items():
            story.append(Paragraph(f"LT {k}: {v:.4f}", styles["Normal"]))
        doc.build(story)
        st.download_button("Download PDF", data=pdf_buffer.getvalue(),
                           file_name=f"{token_text}_ta_report.pdf", mime="application/pdf")
