import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

st.set_page_config(layout='wide', page_title='Streamlit Crypto TA Analyzer v1.4')
st.title('Streamlit Crypto TA Analyzer — Interactive v1.4')

# Session state to preserve selected token between reruns
if 'token_text' not in st.session_state:
    st.session_state['token_text'] = 'solana'
if 'coin_list' not in st.session_state:
    st.session_state['coin_list'] = None

# Sidebar controls
st.sidebar.header('Data & Settings')
use_cg = st.sidebar.checkbox('Use CoinGecko (public API)', value=True)
use_cmc = st.sidebar.checkbox('Use CoinMarketCap Pro (requires API key)', value=False)
cmc_key = st.sidebar.text_input('CMC API Key (optional)', type='password')
# Token input bound to session_state so it can be updated by picker
token_text = st.sidebar.text_input('Token (CoinGecko id)', value=st.session_state['token_text'], key='token_text_input')
days_opt = st.sidebar.selectbox('Days to fetch', options=[1,7,30,90,365], index=2, key='days_opt')
n_sims = st.sidebar.slider('Monte Carlo sims', min_value=500, max_value=20000, value=5000, step=500, key='n_sims')

st.sidebar.markdown('---')
st.sidebar.markdown('Upload CSV (optional) to use your own data:')
uploaded_file = st.sidebar.file_uploader('Upload CSV with OHLCV', type=['csv'])

# Pick from top coins
with st.sidebar.expander('Pick from top coins'):
    try:
        coins = fetch_coingecko_list()
        st.session_state['coin_list'] = coins
        names = [f"{c['id']} ({c['symbol']})" for c in coins]
        choice = st.selectbox('Top coins', options=names, index=0, key='topcoins_select')
        if st.button('Use selected coin', key='use_selected_coin'):
            chosen = choice.split(' ')[0]
            st.session_state['token_text'] = chosen
            # update sidebar input value by setting session_state key
            st.session_state['token_text'] = chosen
            st.rerun()
    except Exception as e:
        st.write('Could not fetch coin list:', e)

# Main fetch button
if st.button('Fetch & Analyze'):
    # Decide data source (uploaded CSV has priority)
    data_source = 'api'
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Try to find timestamp column
            ts_candidates = [c for c in df.columns if c.lower() in ('timestamp','time','date','datetime')]
            if ts_candidates:
                ts_col = ts_candidates[0]
                df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(df.iloc[:,0], errors='coerce')

            # Detect OHLC columns (flexible names)
            cols_lower = {c.lower(): c for c in df.columns}
            def get_col(pref_list):
                for p in pref_list:
                    if p in cols_lower:
                        return cols_lower[p]
                return None
            open_col = get_col(['open','o'])
            high_col = get_col(['high','h'])
            low_col = get_col(['low','l'])
            close_col = get_col(['close','c','price','last'])
            vol_col = get_col(['volume','vol'])

            if close_col is None:
                st.error('Uploaded CSV must contain a close price column (close, c, price or last).')
                st.stop()

            df['close'] = pd.to_numeric(df[close_col], errors='coerce')
            if open_col:
                df['open'] = pd.to_numeric(df[open_col], errors='coerce')
            else:
                df['open'] = df['close'].shift(1)
                df.loc[0, 'open'] = df.loc[0, 'close']
            if high_col:
                df['high'] = pd.to_numeric(df[high_col], errors='coerce')
            else:
                df['high'] = df[['open','close']].max(axis=1)
            if low_col:
                df['low'] = pd.to_numeric(df[low_col], errors='coerce')
            else:
                df['low'] = df[['open','close']].min(axis=1)
            if vol_col:
                df['volume'] = pd.to_numeric(df[vol_col], errors='coerce')
            df = df[['timestamp','open','high','low','close']].dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            data_source = 'csv'
        except Exception as e:
            st.error('Error reading uploaded CSV: '+str(e))
            st.stop()
    else:
        # Fetch from API
        with st.spinner('Fetching market data...'):
            try:
                fetch_days = max(days_opt, 90)
                if use_cmc and cmc_key.strip():
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

    if df is None or df.empty:
        st.error('No data returned. Try another token or timeframe.')
        st.stop()

    # Show token name and latest price under the button
    token_name = token_text
    latest_price = None
    if data_source == 'api' and st.session_state.get('coin_list'):
        coin = next((c for c in st.session_state['coin_list'] if c['id'] == token_text), None)
        if coin:
            token_name = coin.get('name', token_text)
            latest_price = coin.get('current_price', None)
    if latest_price is None:
        try:
            latest_price = float(df['close'].iloc[-1])
        except Exception:
            latest_price = None

    st.markdown('---')
    st.markdown(f"### 🪙 Token: **{token_name}**  —  **{token_text.upper()}**")
    if latest_price is not None:
        st.markdown(f"### 💰 Latest close: **${latest_price:,.4f} USD**")
    st.markdown('---')

    # Compute indicators
    df['MA20'] = sma(df['close'], 20)
    df['MA50'] = sma(df['close'], 50)
    df['RSI14'] = compute_rsi(df['close'], 14)
    macd, sig, hist = compute_macd(df['close'])
    df['MACD'], df['MACD_sig'], df['MACD_hist'] = macd, sig, hist

    support, resistance = simple_support_resistance(df, window=20)

    # Fibonacci levels
    window_30 = df.tail(30) if len(df) >= 30 else df
    st_high, st_low = float(window_30['high'].max()), float(window_30['low'].min())
    st_retr, st_ext = fib_levels(st_high, st_low)
    lt_high, lt_low = float(df['high'].max()), float(df['low'].min())
    lt_retr, lt_ext = fib_levels(lt_high, lt_low)

    # Monte Carlo probabilities
    probs = compute_probs_from_df(df, support, resistance, horizons=[1,7,30,90,365], n_sims=n_sims)
    mapping = {1:'1D',7:'7D',30:'1M',90:'90D',365:'365D'}
    selected = days_opt
    sel_probs = probs.get(selected, {})
    bull_target = lt_ext.get('161.8%') if lt_ext else None
    base_target = (support + resistance) / 2.0 if (support is not None and resistance is not None) else None
    bear_target = max(0.0, st_low - (st_high - st_low)*0.618)

    # Top scenario table
    scenario = pd.DataFrame([{
        'Horizon': mapping.get(selected, str(selected)+'d'),
        'Prob Bull': sel_probs.get('Prob Bull', None),
        'Prob Base': sel_probs.get('Prob Base', None),
        'Prob Bear': sel_probs.get('Prob Bear', None),
        'Bull target': bull_target,
        'Base target': base_target,
        'Bear target': bear_target
    }])

    st.subheader('Scenario probabilities & targets (selected horizon)')
    try:
        numeric_cols = scenario.select_dtypes(include='number').columns
        st.dataframe(scenario.style.format(subset=numeric_cols, formatter='{:.4f}'))
    except Exception:
        st.dataframe(scenario)

    # Charts
    chart_window = max(60, days_opt * 3)
    df_chart = df.tail(chart_window).reset_index(drop=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.18, 0.24],
                        specs=[[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]])

    fig.add_trace(go.Candlestick(x=df_chart['timestamp'], open=df_chart['open'], high=df_chart['high'],
                                 low=df_chart['low'], close=df_chart['close'], name='Candles'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MA20'], name='MA20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MA50'], name='MA50', line=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=[df_chart['timestamp'].iloc[0], df_chart['timestamp'].iloc[-1]], y=[support, support],
                             mode='lines', name='Support', line=dict(color='green', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df_chart['timestamp'].iloc[0], df_chart['timestamp'].iloc[-1]], y=[resistance, resistance],
                             mode='lines', name='Resistance', line=dict(color='red', dash='dot')), row=1, col=1)

    # highlight SR bands
    try:
        rng = (resistance - support) if (resistance is not None and support is not None) else 0.0
        band = max(abs(rng)*0.05, 0.0001)
        fig.add_hrect(y0=support-band, y1=support+band, fillcolor='green', opacity=0.08, row=1, col=1, line_width=0)
        fig.add_hrect(y0=resistance-band, y1=resistance+band, fillcolor='red', opacity=0.08, row=1, col=1, line_width=0)
    except Exception:
        pass

    for label, price in st_retr.items():
        fig.add_hline(y=price, line_dash='dot', line=dict(width=1), annotation_text=f'ST {label} {price:.2f}', annotation_position='right', row=1, col=1)
    for label, price in lt_retr.items():
        fig.add_hline(y=price, line_dash='dash', line=dict(width=1), annotation_text=f'LT {label} {price:.2f}', annotation_position='left', row=1, col=1)

    # RSI plot with zones
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['RSI14'], name='RSI14', line=dict(color='purple')), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='red', opacity=0.06, row=2, col=1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor='green', opacity=0.06, row=2, col=1, line_width=0)
    fig.add_hline(y=70, line_dash='dash', line_color='gray', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='gray', row=2, col=1)
    fig.update_yaxes(range=[0,100], row=2, col=1, title_text='RSI')

    # MACD
    fig.add_trace(go.Bar(x=df_chart['timestamp'], y=df_chart['MACD_hist'], name='MACD hist', marker_color='orange', opacity=0.5), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['MACD_sig'], name='Signal', line=dict(color='red', dash='dot')), row=3, col=1)

    # Centered titles above each subplot using annotations
    titles = ['Price with Fibonacci, MA20/50, Support & Resistance', 'RSI (Relative Strength Index)', 'MACD (Moving Average Convergence Divergence)']
    y_positions = [0.99, 0.635, 0.32]
    annotations = []
    for yp, t in zip(y_positions, titles):
        annotations.append(dict(x=0.5, y=yp, xref='paper', yref='paper', text=t, showarrow=False, font=dict(size=14), xanchor='center'))
    fig.update_layout(annotations=annotations,
                      height=980, showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
                      margin=dict(t=140, b=40))

    st.plotly_chart(fig, use_container_width=True)

    # Monte Carlo full summary
    st.subheader('Monte Carlo summary (all horizons)')
    mc_summary = pd.DataFrame.from_dict({str(k)+'d':v for k,v in probs.items()}, orient='index')
    for col in mc_summary.columns:
        mc_summary[col] = pd.to_numeric(mc_summary[col], errors='ignore')
    try:
        num_cols = mc_summary.select_dtypes(include='number').columns
        st.dataframe(mc_summary.style.format(subset=num_cols, formatter='{:.4f}'))
    except Exception:
        st.dataframe(mc_summary)

    # PDF export in-memory
    if st.button('Export PDF report'):
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph(f'Crypto TA Report - {token_text}', styles['Title']), Spacer(1,12),
                     Paragraph(f'Latest close: {df["close"].iloc[-1]:.4f} USD', styles['Normal']), Spacer(1,6)]
            # add fibs as table
            data = [['Level', 'Price']]
            for label, price in st_retr.items():
                data.append([f'ST {label}', f'{price:.4f}'])
            data.append(['', ''])
            for label, price in lt_retr.items():
                data.append([f'LT {label}', f'{price:.4f}'])
            table = Table(data, hAlign='LEFT')
            table.setStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                            ('GRID',(0,0),(-1,-1),0.5,colors.lightgrey)])
            story.append(table)
            doc.build(story)
            buffer.seek(0)
            st.download_button('Download PDF', data=buffer, file_name=f'{token_text}_ta_report.pdf', mime='application/pdf')
        except Exception as e:
            st.error('PDF export failed: '+str(e))

# End of app
