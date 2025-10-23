
import streamlit as st
import pandas as pd, numpy as np, time, io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os, tempfile

st.set_page_config(layout='wide', page_title='Streamlit Crypto TA Analyzer')
st.title('Streamlit Crypto TA Analyzer — Interactive')

# Sidebar options
st.sidebar.header('Data & Settings')
use_cg = st.sidebar.checkbox('Use CoinGecko (public API)', value=True)
use_cmc = st.sidebar.checkbox('Use CoinMarketCap Pro (requires API key)', value=False)
cmc_key = st.sidebar.text_input('CMC API Key (optional)', type='password')
token_text = st.sidebar.text_input('Token (CoinGecko id)', value='solana')
days_opt = st.sidebar.selectbox('Days to fetch', options=[1,7,30,90,365], index=2)
n_sims = st.sidebar.slider('Monte Carlo sims', min_value=500, max_value=20000, value=5000, step=500)

# Autocomplete coin list (fetch top coins)
with st.sidebar.expander('Pick from top coins'):
    try:
        coins = fetch_coingecko_list()
        names = [f\"{c['id']} ({c['symbol']})" for c in coins]
        choice = st.selectbox('Top coins', options=names, index=0)
        if st.button('Use selected coin'):
            token_text = choice.split(' ')[0]
    except Exception as e:
        st.write('Could not fetch coin list:', e)

if st.button('Fetch & Analyze'):
    with st.spinner('Fetching data...'):
        try:
            # fetch at least 90 days for stable indicators
            fetch_days = max(days_opt, 90)
            if use_cmc and cmc_key.strip():
                # using CMC pro requires date range; compute start/end
                end = datetime.utcnow().date()
                start = end - timedelta(days=fetch_days)
                df = fetch_cmc_pro(token_text.upper(), start.isoformat(), end.isoformat(), api_key=cmc_key)
                if df is None or df.empty:
                    st.warning('CMC returned empty; falling back to CoinGecko')
                    df = fetch_coingecko_daily(token_text, days=fetch_days)
            else:
                df = fetch_coingecko_daily(token_text, days=fetch_days)
        except Exception as e:
            st.error('Error fetching data: '+str(e))
            st.stop()

    # Show cached/fresh info (best-effort; compatible across Streamlit versions)
    try:
        from streamlit.runtime.caching import cache_data_api
        info = cache_data_api.get_cached_func_info(fetch_coingecko_daily)
        if info and getattr(info, "stats", None) and getattr(info.stats, "hits", 0) > 0:
            st.info(f\"📊 Using cached data for {token_text.upper()} ({days_opt} days). Cache refreshes hourly.\")
        else:
            st.success(f\"✅ Fresh data fetched for {token_text.upper()} ({days_opt} days).")
    except Exception:
        # Non-fatal: just continue without cache-info
        pass

    if df is None or df.empty:
        st.error("No data returned. Try another token or increase the timeframe.")
        st.stop()

    # prepare indicators
    df['MA20'] = sma(df['close'], 20)
    df['MA50'] = sma(df['close'], 50)
    df['RSI14'] = compute_rsi(df['close'], 14)
    macd, sig, hist = compute_macd(df['close'])
    df['MACD'] = macd; df['MACD_sig'] = sig; df['MACD_hist'] = hist

    support, resistance = simple_support_resistance(df, window=20)

    # short swing (30d) and long (full range)
    window_30 = df.tail(30) if len(df)>=30 else df
    st_high, st_low = float(window_30['high'].max()), float(window_30['low'].min())
    st_retr, st_ext = fib_levels(st_high, st_low)
    lt_high, lt_low = float(df['high'].max()), float(df['low'].min())
    lt_retr, lt_ext = fib_levels(lt_high, lt_low)

    probs = compute_probs_from_df(df, support, resistance, horizons=[1,7,30,90,365], n_sims=n_sims)

    # Scenario table for selected horizon
    mapping = {1:'1D',7:'7D',30:'1M',90:'90D',365:'365D'}
    selected = days_opt
    sel_probs = probs.get(selected, {})
    bull_target = lt_ext.get('161.8%') if lt_ext else None
    bear_target = max(0.0, st_low - (st_high - st_low)*0.618)

    scenario = pd.DataFrame([{
        'Horizon': mapping.get(selected, str(selected)+'d'),
        'Prob Bull': sel_probs.get('Prob Bull', None),
        'Prob Base': sel_probs.get('Prob Base', None),
        'Prob Bear': sel_probs.get('Prob Bear', None),
        'Bull target': bull_target,
        'Bear target': bear_target
    }])

    st.subheader('Scenario probabilities & targets')
    # Format numeric columns only to avoid pandas Styler errors
    try:
        numeric_cols = scenario.select_dtypes(include='number').columns
        st.dataframe(scenario.style.format(subset=numeric_cols, formatter="{:.4f}"))
    except Exception:
        st.dataframe(scenario)

    # Build interactive chart (last window)
    chart_window = max(60, days_opt*3)
    df_chart = df.tail(chart_window)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.18,0.22], specs=[[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]])
    fig.add_trace(go.Candlestick(x=df_chart['timestamp'], open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name='Candles'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MA20'], name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MA50'], name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df_chart['timestamp'].iloc[0], df_chart['timestamp'].iloc[-1]], y=[support, support], mode='lines', name='Support'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df_chart['timestamp'].iloc[0], df_chart['timestamp'].iloc[-1]], y=[resistance, resistance], mode='lines', name='Resistance'), row=1, col=1)

    # fibs
    for k,v in st_retr.items():
        fig.add_hline(y=v, line_dash='dot', annotation_text=f'ST {k} {v:.2f}', annotation_position='right', row=1, col=1)
    for k,v in lt_retr.items():
        fig.add_hline(y=v, line_dash='dash', annotation_text=f'LT {k} {v:.2f}', annotation_position='left', row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['RSI14'], name='RSI14'), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='lightgray', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='lightgray', row=2, col=1)
    # MACD
    fig.add_trace(go.Bar(x=df_chart['timestamp'], y=df_chart['MACD_hist'], name='MACD hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MACD'], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MACD_sig'], name='Signal'), row=3, col=1)
    fig.update_layout(height=900, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Monte Carlo summary (all horizons)')
    # Build summary DataFrame safely and format numeric columns only
    mc_summary = pd.DataFrame.from_dict({str(k)+'d':v for k,v in probs.items()}, orient='index')
    # Attempt to cast numeric-like values to numbers where possible
    for col in mc_summary.columns:
        mc_summary[col] = pd.to_numeric(mc_summary[col], errors='ignore')

    try:
        num_cols = mc_summary.select_dtypes(include='number').columns
        st.dataframe(mc_summary.style.format(subset=num_cols, formatter="{:.4f}"))
    except Exception:
        st.dataframe(mc_summary)

    # PDF export
    if st.button('Export PDF report'):
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(tmpf.name)
        styles = getSampleStyleSheet()
        story = [Paragraph(f"Crypto TA Report - {token_text}", styles['Title']), Spacer(1,12), Paragraph(f"Latest close: {df['close'].iloc[-1]:.4f} USD", styles['Normal']), Spacer(1,6)]
        for k,v in st_retr.items():
            story.append(Paragraph(f"ST {k}: {v:.4f}", styles['Normal']))
        story.append(Spacer(1,12))
        for k,v in lt_retr.items():
            story.append(Paragraph(f"LT {k}: {v:.4f}", styles['Normal']))
        doc.build(story)
        with open(tmpf.name, 'rb') as f:
            btn = st.download_button('Download PDF', data=f, file_name=f'{token_text}_ta_report.pdf', mime='application/pdf')
        os.unlink(tmpf.name)
