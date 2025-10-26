
import streamlit as st
import pandas as pd, numpy as np, io, csv, re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
import plotly.io as pio

st.set_page_config(layout='wide', page_title='Streamlit Crypto TA Analyzer — Robust v2.1')
st.title('Streamlit Crypto TA Analyzer — Robust v2.1')

# ------------------------------------------------------------
# 🔍 Kaleido availability check
# ------------------------------------------------------------
def check_kaleido():
    try:
        _ = pio.to_image
        return True
    except Exception:
        return False

if "kaleido_ok" not in st.session_state:
    st.session_state["kaleido_ok"] = check_kaleido()

python_path = sys.executable
install_cmd = f'"{python_path}" -m pip install kaleido'

if not st.session_state["kaleido_ok"]:
    st.warning("⚠️ Kaleido is not installed. Chart images will not appear in PDF reports.")
    st.code(install_cmd, language="bash")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("📋 Copy Install Command"):
            try:
                import pyperclip
                pyperclip.copy(install_cmd)
                st.success("✅ Command copied to clipboard!")
            except Exception:
                st.warning("Clipboard not accessible. Please copy manually.")
    with col2:
        if st.button("🔄 Recheck Kaleido"):
            st.session_state["kaleido_ok"] = check_kaleido()
            if st.session_state["kaleido_ok"]:
                st.success("✅ Kaleido detected! Chart export is now enabled.")
            else:
                st.error("❌ Still not found. Please run the command above and recheck.")
else:
    st.success("✅ Kaleido available — PDF chart export enabled.")

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
        txt = sample if 'sample' in locals() else ''
        if txt.count(';') > txt.count(','):
            return ';'
        return ','

def normalize_columns(df):
    df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
    return df

def map_columns(df):
    cols = list(df.columns)
    mapping = {}

    time_candidates = ['timestamp','time','date','datetime','open time','close time']
    for c in cols:
        if c in time_candidates or c.startswith('time'):
            mapping['timestamp'] = c
            break

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
    vol = find_one({'volume'}, contains='vol')
    mapping['volume'] = vol

    return mapping

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(' ', '')
                df[c] = df[c].str.replace(',', '.', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def synthesize_ohlc(df):
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
        return '1D', 'days', 24
    diffs = ts_series.sort_values().diff().dropna().dt.total_seconds()
    med = float(diffs.median())
    if med < 2*3600:
        return '1H', 'hours', 1
    if med < 6*3600:
        return '4H', 'hours', 4
    if med < 3*24*3600:
        return '1D', 'days', 24
    return '1W', 'weeks', 24*7

def horizon_steps_for_timeframe(tf):
    if tf == '1H':
        return [24, 24*3, 24*7, 24*14, 24*30], ['24h','3d','7d','14d','30d']
    if tf == '4H':
        return [6, 18, 42, 84, 180], ['1d','3d','1w','2w','1m']
    if tf == '1W':
        return [1, 4, 12, 26, 52], ['1w','4w','12w','26w','52w']
    return [1, 7, 30, 90, 365], ['1d','7d','30d','90d','365d']

def build_figure(df_chart, timeframe, support, resistance, st_retr, lt_retr):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.2, 0.25],
                        specs=[[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]],
                        subplot_titles=(f"Price ({timeframe}) with Fibonacci, MA20/50, Support & Resistance",
                                        "RSI (Relative Strength Index)",
                                        "MACD (Moving Average Convergence Divergence)"))
    # Candles
    fig.add_trace(go.Candlestick(x=df_chart["timestamp"], open=df_chart["open"], high=df_chart["high"],
                                 low=df_chart["low"], close=df_chart["close"], name="Candles"), row=1, col=1)
    # MAs
    if 'MA20' in df_chart.columns and 'MA50' in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA50"], name="MA50"), row=1, col=1)

    # S/R
    if pd.notna(support) and pd.notna(resistance):
        fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                                 y=[support, support], mode="lines", name="Support"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                                 y=[resistance, resistance], mode="lines", name="Resistance"), row=1, col=1)

    # Fibs
    try:
        for k, v in st_retr.items():
            fig.add_hline(y=v, line_dash="dot", annotation_text=f"ST {k} {v:.2f}", annotation_position="right", row=1, col=1)
        for k, v in lt_retr.items():
            fig.add_hline(y=v, line_dash="dash", annotation_text=f"LT {k} {v:.2f}", annotation_position="left", row=1, col=1)
    except Exception:
        pass

    # RSI
    if 'RSI14' in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["RSI14"], name="RSI14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="lightgray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lightgray", row=2, col=1)

    # MACD
    if {'MACD_hist','MACD','MACD_sig'}.issubset(df_chart.columns):
        fig.add_trace(go.Bar(x=df_chart["timestamp"], y=df_chart["MACD_hist"], name="MACD hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD_sig"], name="Signal"), row=3, col=1)

    # Legend at bottom
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
    return fig

def build_pdf_bytes(token_text, timeframe, df, st_retr, lt_retr, scenario_df, fig=None):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    def fmt_num(x, digits=4):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "N/A"
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)
    styles = getSampleStyleSheet()

    story = [
        Paragraph(f"Crypto TA Report — {token_text.capitalize()}", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Detected timeframe: {timeframe}", styles["Normal"]),
        Spacer(1, 12),
    ]

    # Latest close
    if "close" in df.columns and not df["close"].empty:
        story.append(Paragraph(f"Latest close: ${fmt_num(df['close'].iloc[-1])} USD", styles["Normal"]))
        story.append(Spacer(1, 12))

    # Embed chart image (medium size) if possible
    if fig is not None:
        try:
            img_bytes = pio.to_image(fig, format="png", scale=2)
            img_reader = ImageReader(io.BytesIO(img_bytes))
            iw, ih = img_reader.getSize()
            target_w = 420  # medium width (~half page)
            target_h = ih * (target_w / iw)
            story.append(Image(img_reader, width=target_w, height=target_h))
            story.append(Spacer(1, 12))
        except Exception as e:
            # Kaleido likely missing; add a note
            story.append(Paragraph("Chart image could not be embedded (kaleido not available).", styles["Italic"]))
            story.append(Spacer(1, 12))

    # Short-term Fibonacci
    story.append(Paragraph("Short-term Fibonacci levels:", styles["Heading3"]))
    for k, v in (st_retr or {}).items():
        story.append(Paragraph(f"{k}: {fmt_num(v)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Long-term Fibonacci
    story.append(Paragraph("Long-term Fibonacci levels:", styles["Heading3"]))
    for k, v in (lt_retr or {}).items():
        story.append(Paragraph(f"{k}: {fmt_num(v)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Scenario (first row)
    if scenario_df is not None and len(scenario_df) > 0:
        story.append(Paragraph("Scenario probabilities (selected horizon):", styles["Heading3"]))
        for k, v in scenario_df.iloc[0].items():
            story.append(Paragraph(f"{k}: {fmt_num(v)}", styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

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

            # Rename to standard names
            for std, actual in colmap.items():
                if actual and actual in df.columns and std != actual:
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
    for c in ['open','high','low','close']:
        if c not in df.columns:
            st.error(f"CSV missing required column: {c}.")
            st.stop()

    # Detect timeframe
    timeframe, horizon_label, unit_size = detect_timeframe(df['timestamp'])
    st.info(f"📅 Detected timeframe: **{timeframe}** data")

    # Technical indicators (compute only if missing)
    if 'MA20' not in df.columns:
        df["MA20"] = sma(df["close"], 20)
    if 'MA50' not in df.columns:
        df["MA50"] = sma(df["close"], 50)
    if 'RSI14' not in df.columns:
        df["RSI14"] = compute_rsi(df["close"], 14)
    if not {'MACD','MACD_sig','MACD_hist'}.issubset(df.columns):
        macd, sig, hist = compute_macd(df["close"])
        df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd, sig, hist

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

    # Choose representative horizon
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
        fig = None
    else:
        fig = build_figure(df_chart, timeframe, support if pd.notna(support) else np.nan,
                           resistance if pd.notna(resistance) else np.nan, st_retr, lt_retr)
        st.plotly_chart(fig, use_container_width=True)

    # --- Reliable PDF download (always rendered) ---
    try:
        pdf_bytes = build_pdf_bytes(token_text, timeframe, df, st_retr, lt_retr, scenario, fig=fig)
        downloaded = st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{token_text}_ta_report.pdf",
            mime="application/pdf",
            key="pdf_download_v21",
        )
        if downloaded:
            st.success("PDF generated and download started ✅")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

    # Monte Carlo summary (all horizons)
    st.subheader("Monte Carlo summary (all horizons)")
    try:
        mc_summary = pd.DataFrame.from_dict({str(k): v for k, v in probs.items()}, orient="index")
        num_cols = mc_summary.select_dtypes(include=[np.number]).columns
        st.dataframe(mc_summary.style.format(subset=num_cols, formatter="{:.4f}"))
    except Exception:
        st.dataframe(pd.DataFrame(probs))

    # Diagnostics
    with st.expander("Diagnostics"):
        st.write("Detected columns:", list(df.columns))
        st.write("Dtypes:", df.dtypes.astype(str).to_dict())
        st.write("First 5 rows:")
        st.dataframe(df.head())
