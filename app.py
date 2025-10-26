
import streamlit as st
import pandas as pd, numpy as np, io, csv, re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout='wide', page_title='Streamlit Crypto TA Analyzer — Robust v2.0')
st.title('Streamlit Crypto TA Analyzer — Robust v2.0')

# -----------------------------
# Helpers
# -----------------------------
def sniff_delimiter(uploaded_file):
    try:
        sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        # Heuristics: prefer semicolon if many present
        txt = sample if 'sample' in locals() else ''
        if txt.count(';') > txt.count(','):
            return ';'
        return ','

def normalize_columns(df):
    # lower, strip, remove extra spaces
    df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
    return df

def map_columns(df):
    cols = list(df.columns)
    mapping = {}

    # Time column candidates
    time_candidates = ['timestamp','time','date','datetime','open time','close time']
    for c in cols:
        if c in time_candidates or c.startswith('time'):
            mapping['timestamp'] = c
            break

    # OHLC mapping (accept variations)
    def find_one(names, contains=None):
        for c in cols:
            if c in names:
                return c
        if contains:
            for c in cols:
                if contains in c:
                    return c
        return None

    mapping['open'] = find_one({'open','o'})
    mapping['high'] = find_one({'high','h'})
    mapping['low']  = find_one({'low','l'})
    mapping['close']= find_one({'close','c','final','last','close price'})
    # volume: match "volume" or any column containing 'vol'
    vol = find_one({'volume'}, contains='vol')
    mapping['volume'] = vol

    return mapping

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            # handle numbers with comma decimal (e.g., "123,45")
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(' ', '')
                # if comma as decimal and no dots, replace comma with dot
                df[c] = df[c].str.replace(',', '.', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def synthesize_ohlc(df):
    # If open/high/low missing, build from close
    if 'close' not in df.columns:
        return df
    if 'open' not in df.columns:
        df['open'] = df['close'].shift(1).bfill()
    if 'high' not in df.columns:
        df['high'] = df[['open','close']].max(axis=1)
    if 'low' not in df.columns:
        df['low'] = df[['open','close']].min(axis=1)
    return df

def detect_timeframe(ts_series):
    if ts_series.dropna().size < 2:
        return '1D', 'days', 24  # default
    diffs = ts_series.sort_values().diff().dropna().dt.total_seconds()
    med = float(diffs.median())
    # thresholds
    if med < 2*3600:          # <2h
        return '1H', 'hours', 1
    if med < 6*3600:          # <6h
        return '4H', 'hours', 4
    if med < 3*24*3600:       # <3d
        return '1D', 'days', 24
    return '1W', 'weeks', 24*7

def horizon_steps_for_timeframe(tf):
    if tf == '1H':
        # horizons in hours -> convert to steps directly
        return [24, 24*3, 24*7, 24*14, 24*30], ['24h','3d','7d','14d','30d']
    if tf == '4H':
        # steps are 4h chunks
        return [6, 18, 42, 84, 180], ['1d','3d','1w','2w','1m']
    if tf == '1W':
        return [1, 4, 12, 26, 52], ['1w','4w','12w','26w','52w']
    # default 1D
    return [1, 7, 30, 90, 365], ['1d','7d','30d','90d','365d']

def robust_plot(df_chart, timeframe, support, resistance, st_retr, lt_retr):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.2, 0.25],
                        specs=[[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]],
                        subplot_titles=(f"Price ({timeframe}) with Fibonacci, MA20/50, Support & Resistance",
                                        "RSI (Relative Strength Index)",
                                        "MACD (Moving Average Convergence Divergence)"))
    fig.add_trace(go.Candlestick(x=df_chart["timestamp"], open=df_chart["open"], high=df_chart["high"],
                                 low=df_chart["low"], close=df_chart["close"], name="Candles"), row=1, col=1)
    if 'MA20' in df_chart.columns and 'MA50' in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA50"], name="MA50"), row=1, col=1)

    if np.isfinite(support) and np.isfinite(resistance):
        fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                                 y=[support, support], mode="lines", name="Support"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                                 y=[resistance, resistance], mode="lines", name="Resistance"), row=1, col=1)

    # Fibonacci lines (best-effort, ignore if subplot hline add fails)
    try:
        for k, v in st_retr.items():
            fig.add_hline(y=v, line_dash="dot", annotation_text=f"ST {k} {v:.2f}", annotation_position="right", row=1, col=1)
        for k, v in lt_retr.items():
            fig.add_hline(y=v, line_dash="dash", annotation_text=f"LT {k} {v:.2f}", annotation_position="left", row=1, col=1)
    except Exception:
        pass

    if 'RSI14' in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["RSI14"], name="RSI14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="lightgray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lightgray", row=2, col=1)

    if {'MACD_hist','MACD','MACD_sig'}.issubset(df_chart.columns):
        fig.add_trace(go.Bar(x=df_chart["timestamp"], y=df_chart["MACD_hist"], name="MACD hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD_sig"], name="Signal"), row=3, col=1)

    #fig.update_layout(height=950, showlegend=True, legend_tracegroupgap=5)
    fig.update_layout(
        height=950,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.7)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header('Data & Settings')
use_cg = st.sidebar.checkbox('Use CoinGecko (public API)', value=True)
use_cmc = st.sidebar.checkbox('Use CoinMarketCap Pro (requires API key)', value=False)
cmc_key = st.sidebar.text_input('CMC API Key (optional)', type='password')
days_opt = st.sidebar.selectbox('Days to fetch (when using APIs)', options=[1,7,30,90,365], index=2)
n_sims = st.sidebar.slider('Monte Carlo sims', min_value=500, max_value=20000, value=5000, step=500)

if "token_text" not in st.session_state:
    st.session_state["token_text"] = "solana"

token_text = st.sidebar.text_input("Token (CoinGecko id)", value=st.session_state["token_text"], key="token_text_input")

st.sidebar.subheader("Upload CSV (optional) to use your own data:")
uploaded_file = st.sidebar.file_uploader("Upload CSV with OHLCV", type=["csv"], label_visibility="collapsed")

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

# -----------------------------
# Fetch & Analyze
# -----------------------------
if st.button("Fetch & Analyze"):
    with st.spinner("Fetching or loading data..."):
        if uploaded_file is not None:
            sep = sniff_delimiter(uploaded_file)
            try:
                df = pd.read_csv(uploaded_file, sep=sep)
            except Exception:
                # retry with common seps
                for s in [';', ',', '\t']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=s)
                        break
                    except Exception:
                        df = pd.DataFrame()
                if df.empty:
                    st.error("❌ Could not read the uploaded CSV with any common delimiter.")
                    st.stop()

            df = normalize_columns(df)
            colmap = map_columns(df)

            # Rename columns to standard names if found
            for std, actual in colmap.items():
                if actual and actual in df.columns:
                    if std != actual:
                        df.rename(columns={actual: std}, inplace=True)

            if 'timestamp' not in df.columns:
                st.error(f"❌ No recognizable time column found. Columns detected: {list(df.columns)}")
                st.stop()

            # Parse timestamp
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            except Exception:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            bad_ts = df['timestamp'].isna().sum()
            if bad_ts > 0:
                st.warning(f"🧹 Dropped {bad_ts} rows with invalid timestamps.")
                df = df.dropna(subset=['timestamp'])

            # Coerce numerics & synthesize missing OHLC
            df = coerce_numeric(df, ['open','high','low','close','volume'])
            df = synthesize_ohlc(df)

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

    # Sort & basic cleaning
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Ensure all required numeric columns exist
    for c in ['open','high','low','close']:
        if c not in df.columns:
            st.error(f"CSV missing required column: {c}.")
            st.stop()

    # Detect timeframe
    timeframe, horizon_label, unit_size = detect_timeframe(df['timestamp'])
    st.info(f"📅 Detected timeframe: **{timeframe}** data")

    # Technical indicators (best-effort)
    try:
        df["MA20"] = sma(df["close"], 20)
        df["MA50"] = sma(df["close"], 50)
        df["RSI14"] = compute_rsi(df["close"], 14)
        macd, sig, hist = compute_macd(df["close"])
        df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd, sig, hist
    except Exception as e:
        st.warning(f"Some indicators could not be computed: {e}")

    # Support/Resistance
    try:
        support, resistance = simple_support_resistance(df, window=20)
    except Exception:
        support, resistance = (np.nan, np.nan)

    # Fibonacci
    window_30 = df.tail(30) if len(df) >= 30 else df
    st_high, st_low = float(window_30["high"].max()), float(window_30["low"].min())
    st_retr, st_ext = fib_levels(st_high, st_low)
    lt_high, lt_low = float(df["high"].max()), float(df["low"].min())
    lt_retr, lt_ext = fib_levels(lt_high, lt_low)

    # Monte Carlo horizons based on timeframe
    steps, labels = horizon_steps_for_timeframe(timeframe)
    try:
        probs = compute_probs_from_df(df, support, resistance, horizons=steps, n_sims=n_sims)
    except Exception as e:
        st.warning(f"Monte Carlo simulation failed: {e}")
        probs = {k: {} for k in steps}

    # Scenario table for a representative horizon (use middle index to avoid empty)
    selected_idx = min(2, len(steps)-1)
    selected_steps = steps[selected_idx]
    selected_label = labels[selected_idx]
    sel_probs = probs.get(selected_steps, {})

    bull_target = lt_ext.get("161.8%", None)
    bear_target = max(0.0, st_low - (st_high - st_low)*0.618)
    base_target = (st_high + st_low) / 2

    st.subheader(f"💠 Token: {token_text.capitalize()} — {token_text.upper()} ({timeframe} data)")
    try:
        latest_close = float(df['close'].iloc[-1])
        st.markdown(f"💰 **Latest close:** ${latest_close:.4f} USD")
    except Exception:
        st.markdown("💰 **Latest close:** N/A")

    scenario = pd.DataFrame([{
        "Horizon": selected_label,
        "Prob Bull": sel_probs.get("Prob Bull", np.nan),
        "Prob Base": sel_probs.get("Prob Base", np.nan),
        "Prob Bear": sel_probs.get("Prob Bear", np.nan),
        "Bull target": bull_target,
        "Base target": base_target,
        "Bear target": bear_target,
    }]).replace({None: np.nan})

    st.subheader("Scenario probabilities & targets (selected horizon)")
    numeric_cols = scenario.select_dtypes(include=[np.number]).columns
    try:
        st.dataframe(scenario.style.format(subset=numeric_cols, formatter="{:.4f}"))
    except Exception:
        st.dataframe(scenario)

    # Chart
    chart_window = 300 if timeframe in ('1H','4H') else max(60, 3* (steps[2] if len(steps) > 2 else 60))
    df_chart = df.tail(chart_window)

    if df_chart.empty or df_chart["close"].isna().all():
        st.error("No valid OHLC data available for charting.")
    else:
        robust_plot(df_chart, timeframe, support if pd.notna(support) else np.nan,
                    resistance if pd.notna(resistance) else np.nan, st_retr, lt_retr)

    # Monte Carlo summary (all horizons)
    st.subheader("Monte Carlo summary (all horizons)")
    try:
        mc_summary = pd.DataFrame.from_dict({str(k): v for k, v in probs.items()}, orient="index")
        num_cols = mc_summary.select_dtypes(include=[np.number]).columns
        st.dataframe(mc_summary.style.format(subset=num_cols, formatter="{:.4f}"))
    except Exception:
        st.dataframe(pd.DataFrame(probs))

    # PDF Export
    if st.button("Export PDF report"):
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()
        
        story = [
            Paragraph(f"Crypto TA Report - {token_text.capitalize()}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Detected timeframe: {timeframe}", styles["Normal"]),
            Spacer(1, 12),
        ]

        # Add summary stats safely
        if "close" in df.columns and not df["close"].empty:
            story.append(Paragraph(f"Latest close: ${float(df['close'].iloc[-1]):.4f} USD", styles["Normal"]))
            story.append(Spacer(1, 12))

        # Add Fibonacci levels
        story.append(Paragraph("<b>Short-term Fibonacci levels:</b>", styles["Heading3"]))
        for k, v in st_retr.items():
            story.append(Paragraph(f"{k}: {v:.4f}", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Long-term Fibonacci levels:</b>", styles["Heading3"]))
        for k, v in lt_retr.items():
            story.append(Paragraph(f"{k}: {v:.4f}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Add probabilities if available
        if len(scenario.columns) > 0:
            story.append(Paragraph("<b>Scenario Probabilities:</b>", styles["Heading3"]))
            for k, v in scenario.iloc[0].items():
                story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

        doc.build(story)
        pdf_value = pdf_buffer.getvalue()
        pdf_buffer.close()

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_value,
            file_name=f"{token_text}_ta_report.pdf",
            mime="application/pdf",
        )

    # Diagnostics
    with st.expander("Diagnostics"):
        st.write("Detected columns:", list(df.columns))
        st.write("Dtypes:", df.dtypes.astype(str).to_dict())
        st.write("First 5 rows:")
        st.dataframe(df.head())
