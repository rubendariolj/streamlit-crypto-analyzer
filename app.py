
import os, io, csv, re, sys, json, requests
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide", page_title="Crypto TA Analyzer — v2.5 (Private)")
st.title("Crypto TA Analyzer — v2.5 (Private)")

# ------------------------------------------------------------
# 🔒 Restricted Access (static password via .streamlit/secrets.toml)
# ------------------------------------------------------------
def check_password():
    def password_entered():
        if st.session_state.get("password", "") == st.secrets["password"]:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password — access denied.")
        st.stop()

check_password()

from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df

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

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", " ", c).strip().lower() for c in df.columns]
    return df

def map_columns(df: pd.DataFrame):
    cols = list(df.columns)
    mapping = {}
    time_candidates = ["timestamp", "time", "date", "datetime", "open time", "close time"]
    for c in cols:
        if c in time_candidates or c.startswith("time"):
            mapping["timestamp"] = c
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
    mapping["open"] = find_one({"open","o"})
    mapping["high"] = find_one({"high","h"})
    mapping["low"]  = find_one({"low","l"})
    mapping["close"]= find_one({"close","c","final","last","close price"})
    mapping["volume"] = find_one({"volume"}, contains="vol")
    return mapping

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(" ", "")
                df[c] = df[c].str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def synthesize_ohlc(df):
    if "close" not in df.columns:
        return df
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).bfill()
    if "high" not in df.columns:
        df["high"] = df[["open","close"]].max(axis=1)
    if "low" not in df.columns:
        df["low"] = df[["open","close"]].min(axis=1)
    return df

def detect_timeframe(ts_series: pd.Series):
    if ts_series.dropna().size < 2:
        return "1D", "days", 24
    diffs = ts_series.sort_values().diff().dropna().dt.total_seconds()
    med = float(diffs.median())
    if med < 2*3600:     return "1H", "hours", 1
    if med < 6*3600:     return "4H", "hours", 4
    if med < 3*24*3600:  return "1D", "days", 24
    return "1W", "weeks", 24*7

def horizon_steps_for_timeframe(tf):
    if tf == "1H": return [24, 24*3, 24*7, 24*14, 24*30], ["24h","3d","7d","14d","30d"]
    if tf == "4H": return [6, 18, 42, 84, 180], ["1d","3d","1w","2w","1m"]
    if tf == "1W": return [1, 4, 12, 26, 52], ["1w","4w","12w","26w","52w"]
    return [1, 7, 30, 90, 365], ["1d","7d","30d","90d","365d"]

def build_figure(df_chart, timeframe, support, resistance, st_retr, lt_retr):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.2, 0.25],
                        specs=[[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]],
                        subplot_titles=(f"Price ({timeframe}) — with MA20/50, Fibonacci, S/R",
                                        "RSI (Relative Strength Index)",
                                        "MACD (Moving Average Convergence Divergence)"))
    fig.add_trace(go.Candlestick(x=df_chart["timestamp"], open=df_chart["open"], high=df_chart["high"],
                                 low=df_chart["low"], close=df_chart["close"], name="Candles"), row=1, col=1)
    if "MA20" in df_chart.columns and "MA50" in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MA50"], name="MA50"), row=1, col=1)
    if pd.notna(support) and pd.notna(resistance):
        fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                                 y=[support, support], mode="lines", name="Support"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[df_chart["timestamp"].iloc[0], df_chart["timestamp"].iloc[-1]],
                                 y=[resistance, resistance], mode="lines", name="Resistance"), row=1, col=1)
    try:
        for k, v in st_retr.items():
            fig.add_hline(y=v, line_dash="dot", annotation_text=f"ST {k} {v:.2f}",
                          annotation_position="right", row=1, col=1)
        for k, v in lt_retr.items():
            fig.add_hline(y=v, line_dash="dash", annotation_text=f"LT {k} {v:.2f}",
                          annotation_position="left", row=1, col=1)
    except Exception:
        pass
    if "RSI14" in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["RSI14"], name="RSI14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="lightgray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lightgray", row=2, col=1)
    if {"MACD_hist","MACD","MACD_sig"}.issubset(df_chart.columns):
        fig.add_trace(go.Bar(x=df_chart["timestamp"], y=df_chart["MACD_hist"], name="MACD hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["MACD_sig"], name="Signal"), row=3, col=1)
    fig.update_layout(
        height=950, showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.7)")
    )
    return fig

def build_pdf_bytes(token_text, timeframe, df, st_retr, lt_retr, scenario_df):
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)
    def fmt_num(x, digits=4):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "N/A"
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)
    story = [
        Paragraph(f"Crypto TA Report — {token_text.capitalize()}", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Detected timeframe: {timeframe}", styles["Normal"]),
        Spacer(1, 12),
    ]
    if "close" in df.columns and not df["close"].empty:
        story.append(Paragraph(f"Latest close: ${fmt_num(df['close'].iloc[-1])} USD", styles["Normal"]))
        story.append(Spacer(1, 12))
    story.append(Paragraph("Short-term Fibonacci levels:", styles["Heading3"]))
    for k, v in (st_retr or {}).items():
        story.append(Paragraph(f"{k}: {fmt_num(v)}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Long-term Fibonacci levels:", styles["Heading3"]))
    for k, v in (lt_retr or {}).items():
        story.append(Paragraph(f"{k}: {fmt_num(v)}", styles["Normal"]))
    story.append(Spacer(1, 12))
    if scenario_df is not None and len(scenario_df) > 0:
        story.append(Paragraph("Scenario probabilities (selected horizon):", styles["Heading3"]))
        for k, v in scenario_df.iloc[0].items():
            story.append(Paragraph(f"{k}: {fmt_num(v)}", styles["Normal"]))
        story.append(Spacer(1, 12))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

st.sidebar.header("Data & Settings")
use_cg = st.sidebar.checkbox("Use CoinGecko (public API)", value=True)
use_cmc = st.sidebar.checkbox("Use CoinMarketCap Pro (requires API key)", value=False)
cmc_key = st.sidebar.text_input("CMC API Key (optional)", type="password")
days_opt = st.sidebar.selectbox("Days to fetch (when using APIs)", options=[1,7,30,90,365], index=2)
n_sims = st.sidebar.slider("Monte Carlo sims", min_value=500, max_value=20000, value=5000, step=500)

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

if st.button("Fetch & Analyze"):
    with st.spinner("Fetching or loading data..."):
        if uploaded_file is not None:
            sep = sniff_delimiter(uploaded_file)
            try:
                df = pd.read_csv(uploaded_file, sep=sep)
            except Exception:
                for s in [";", ",", "\t"]:
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
            for std, actual in colmap.items():
                if actual and actual in df.columns and std != actual:
                    df.rename(columns={actual: std}, inplace=True)
            if "timestamp" not in df.columns:
                st.error(f"❌ No recognizable time column found. Columns detected: {list(df.columns)}")
                st.stop()
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            bad_ts = df["timestamp"].isna().sum()
            if bad_ts > 0:
                st.warning(f"🧹 Dropped {bad_ts} rows with invalid timestamps.")
                df = df.dropna(subset=["timestamp"])
            df = coerce_numeric(df, ["open","high","low","close","volume"])
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

    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            st.error(f"CSV missing required column: {c}.")
            st.stop()

    timeframe, horizon_label, unit_size = detect_timeframe(df["timestamp"])
    st.info(f"📅 Detected timeframe: **{timeframe}** data")

    if "MA20" not in df.columns:
        df["MA20"] = sma(df["close"], 20)
    if "MA50" not in df.columns:
        df["MA50"] = sma(df["close"], 50)
    if "RSI14" not in df.columns:
        df["RSI14"] = compute_rsi(df["close"], 14)
    if not {"MACD","MACD_sig","MACD_hist"}.issubset(df.columns):
        macd, sig, hist = compute_macd(df["close"])
        df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd, sig, hist

    try:
        support, resistance = simple_support_resistance(df, window=20)
    except Exception:
        support, resistance = (np.nan, np.nan)

    window_30 = df.tail(30) if len(df) >= 30 else df
    st_high, st_low = float(window_30["high"].max()), float(window_30["low"].min())
    st_retr, st_ext = fib_levels(st_high, st_low)
    lt_high, lt_low = float(df["high"].max()), float(df["low"].min())
    lt_retr, lt_ext = fib_levels(lt_high, lt_low)

    steps, labels = horizon_steps_for_timeframe(timeframe)
    try:
        probs = compute_probs_from_df(df, support, resistance, horizons=steps, n_sims=n_sims)
    except Exception as e:
        st.warning(f"Monte Carlo simulation failed: {e}")
        probs = {k: {} for k in steps}

    selected_idx = min(2, len(steps)-1)
    selected_steps = steps[selected_idx]
    selected_label = labels[selected_idx]
    sel_probs = probs.get(selected_steps, {})

    bull_target = lt_ext.get("161.8%", None)
    base_target = (st_high + st_low) / 2
    bear_target = max(0.0, st_low - (st_high - st_low)*0.618)

    st.subheader(f"💠 Token: {token_text.capitalize()} — {token_text.upper()} ({timeframe} data)")
    try:
        latest_close = float(df["close"].iloc[-1])
        st.markdown(f"💰 **Latest close:** ${latest_close:.4f} USD")
    except Exception:
        latest_close = np.nan
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

    chart_window = 300 if timeframe in ("1H","4H") else max(60, 3 * (steps[2] if len(steps) > 2 else 60))
    df_chart = df.tail(chart_window)
    if df_chart.empty or df_chart["close"].isna().all():
        st.error("No valid OHLC data available for charting.")
        fig = None
    else:
        fig = build_figure(
            df_chart,
            timeframe,
            support if pd.notna(support) else np.nan,
            resistance if pd.notna(resistance) else np.nan,
            st_retr, lt_retr
        )
        st.plotly_chart(fig, use_container_width=True)

    try:
        pdf_bytes = build_pdf_bytes(token_text, timeframe, df, st_retr, lt_retr, scenario)
        downloaded = st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{token_text}_ta_report.pdf",
            mime="application/pdf",
            key="pdf_download_v25",
        )
        if downloaded:
            st.success("PDF generated and download started ✅")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

    with st.expander("📊 Model Precision Tracker (3-day horizon)", expanded=False):
        hist_path = "analysis_history.csv"
        record = {
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "token": token_text,
            "timeframe": timeframe,
            "latest_close": latest_close,
            "bull_target": float(bull_target) if bull_target is not None else np.nan,
            "base_target": float(base_target) if base_target is not None else np.nan,
            "bear_target": float(bear_target) if bear_target is not None else np.nan,
            "prob_bull": float(sel_probs.get("Prob Bull", np.nan)) if sel_probs else np.nan,
            "prob_base": float(sel_probs.get("Prob Base", np.nan)) if sel_probs else np.nan,
            "prob_bear": float(sel_probs.get("Prob Bear", np.nan)) if sel_probs else np.nan,
            "actual_price": np.nan,
            "bull_error_pct": np.nan,
            "base_error_pct": np.nan,
            "bear_error_pct": np.nan,
            "direction_correct": np.nan,
            "evaluated_on": ""
        }
        df_record = pd.DataFrame([record])
        if os.path.exists(hist_path):
            df_record.to_csv(hist_path, mode="a", header=False, index=False)
        else:
            df_record.to_csv(hist_path, index=False)

        df_hist = pd.read_csv(hist_path)

        def fetch_actual_price(token_id: str) -> float:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd"
            try:
                data = requests.get(url, timeout=15).json()
                return float(data[token_id]["usd"])
            except Exception:
                return np.nan

        changed = False
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        for i, row in df_hist.iterrows():
            try:
                ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z","+00:00"))
            except Exception:
                continue
            if (now_utc - ts).days < 3:
                continue
            if not (pd.isna(row.get("actual_price")) or str(row.get("actual_price")) == "nan"):
                continue
            actual_price = fetch_actual_price(str(row["token"]))
            if np.isnan(actual_price):
                continue
            def rel_err(actual, target):
                try:
                    if pd.isna(target) or target == 0:
                        return np.nan
                    return abs(actual - target) / abs(target) * 100.0
                except Exception:
                    return np.nan
            bull_err = rel_err(actual_price, row.get("bull_target"))
            base_err = rel_err(actual_price, row.get("base_target"))
            bear_err = rel_err(actual_price, row.get("bear_target"))
            direction_correct = np.nan
            try:
                if not pd.isna(row.get("latest_close")):
                    if actual_price > row["latest_close"] and row["bull_target"] > row["latest_close"]:
                        direction_correct = True
                    elif actual_price < row["latest_close"] and row["bear_target"] < row["latest_close"]:
                        direction_correct = True
                    else:
                        direction_correct = False
            except Exception:
                direction_correct = np.nan
            df_hist.loc[i, "actual_price"] = actual_price
            df_hist.loc[i, "bull_error_pct"] = bull_err
            df_hist.loc[i, "base_error_pct"] = base_err
            df_hist.loc[i, "bear_error_pct"] = bear_err
            df_hist.loc[i, "direction_correct"] = direction_correct
            df_hist.loc[i, "evaluated_on"] = now_utc.isoformat()
            changed = True

        if changed:
            df_hist.to_csv(hist_path, index=False)

        st.markdown("**Latest evaluation records (most recent first):**")
        st.dataframe(df_hist.sort_values("timestamp", ascending=False).head(20))

        with pd.option_context('mode.use_inf_as_na', True):
            mean_base_err = df_hist["base_error_pct"].dropna().mean() if "base_error_pct" in df_hist else np.nan
            mean_bull_err = df_hist["bull_error_pct"].dropna().mean() if "bull_error_pct" in df_hist else np.nan
            mean_bear_err = df_hist["bear_error_pct"].dropna().mean() if "bear_error_pct" in df_hist else np.nan
            dir_acc = (df_hist["direction_correct"] == True).mean()*100 if "direction_correct" in df_hist and df_hist["direction_correct"].notna().any() else np.nan

        st.write(f"**Mean Base Target Error:** {mean_base_err:.2f}%") if not pd.isna(mean_base_err) else st.write("Mean Base Target Error: N/A")
        st.write(f"**Mean Bull Target Error:** {mean_bull_err:.2f}%") if not pd.isna(mean_bull_err) else st.write("Mean Bull Target Error: N/A")
        st.write(f"**Mean Bear Target Error:** {mean_bear_err:.2f}%") if not pd.isna(mean_bear_err) else st.write("Mean Bear Target Error: N/A")
        st.write(f"**Directional Accuracy:** {dir_acc:.2f}%") if not pd.isna(dir_acc) else st.write("Directional Accuracy: N/A")

        st.download_button(
            label="💾 Download Forecast Accuracy Logs (CSV)",
            data=df_hist.to_csv(index=False).encode("utf-8"),
            file_name="analysis_history.csv",
            mime="text/csv"
        )

    with st.expander("📘 Quick Reference – How to Interpret the Charts", expanded=False):
        st.markdown("""
### 🕯️ Candlestick & Trend
- **Green candle** → bullish (close > open)  
- **Red candle** → bearish (close < open)  
- **MA20 > MA50** → Uptrend  
- **MA20 < MA50** → Downtrend  

### 💹 RSI (Relative Strength Index)
- Above 70 → **Overbought**  
- Below 30 → **Oversold**  
- Crossing 50 → **Momentum shift**

### 📉 MACD (Moving Average Convergence Divergence)
- **MACD > Signal** → Bullish  
- **MACD < Signal** → Bearish  
- **Histogram widening** → Momentum strengthening

### 📏 Fibonacci Levels
- **Bounce @ 0.618** → Support confirmation  
- **Break > 1.0** → Extension target ≈ 1.618 (strong momentum)

### 🎲 Monte Carlo Probabilities
- **Prob Bull > Prob Bear** → Favorable bias / upside potential  
- **Prob Bear > Prob Bull** → Downside risk / caution zone

### ⚙️ Reading Workflow
1. Identify trend (MA20/MA50 + candles)  
2. Check RSI (50 threshold)  
3. Confirm MACD alignment  
4. Validate Fibonacci & Support/Resistance  
5. Read Monte Carlo probabilities

### 📈 Combined Signal Summary
| Indicator | Bullish | Bearish |
|------------|----------|----------|
| MA20 / MA50 | MA20 > MA50 | MA20 < MA50 |
| RSI | RSI > 50 | RSI < 50 |
| MACD | MACD > Signal | MACD < Signal |
| Fibonacci | Bounce @ 0.618 / Break > 1.0 | Drop < 0.618 / Rejection |
| Monte Carlo | Prob Bull > Prob Bear | Prob Bear > Prob Bull |

---
<i>Prepared by Ruben D. / TokenConsult © 2025</i>
""", unsafe_allow_html=True)

else:
    st.info("👆 Choose a token or upload a CSV, then click **Fetch & Analyze** to run the analysis.")
