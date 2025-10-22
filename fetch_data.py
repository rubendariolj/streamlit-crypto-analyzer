import streamlit as st
import requests, pandas as pd, os, json

COINGECKO_API = "https://api.coingecko.com/api/v3"

@st.cache_data(ttl=3600)
def fetch_coingecko_list(vs_currency='usd', per_page=250):
    """
    Fetch the top coins from CoinGecko (cached for 1 hour).
    This reduces API calls and avoids 429 Too Many Requests errors.
    """
    coins = []
    page = 1
    while True:
        url = f"{COINGECKO_API}/coins/markets"
        params = {
            'vs_currency': vs_currency,
            'order': 'market_cap_desc',
            'per_page': per_page,
            'page': page,
            'sparkline': 'false'
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.warning("⚠️ CoinGecko rate limit reached. Please wait a few minutes.")
                break
            else:
                st.error(f"Error fetching coin list: {e}")
                break
        data = r.json()
        if not data:
            break
        coins.extend(data)
        page += 1
        # limit pages to avoid long waits (we only need top ~1000)
        if page > 4:
            break
    return coins


@st.cache_data(ttl=3600)
def fetch_coingecko_daily(coin_id: str, days: int = 365):
    """
    Fetch daily OHLCV data from CoinGecko and cache results for 1 hour.
    """
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error("🚫 CoinGecko rate limit reached. Please try again in a few minutes.")
        else:
            st.error(f"Error fetching data for {coin_id}: {e}")
        return pd.DataFrame()

    j = r.json()
    prices = j.get("prices", [])
    vols = j.get("total_volumes", [])
    mcaps = j.get("market_caps", [])

    rows = []
    for i, (ts_ms, price) in enumerate(prices):
        dt = pd.to_datetime(ts_ms, unit='ms', utc=True)
        vol = vols[i][1] if i < len(vols) else None
        mcap = mcaps[i][1] if i < len(mcaps) else None
        rows.append({
            "timestamp": dt,
            "close": float(price),
            "volume": vol,
            "market_cap": mcap
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return df

    df['open'] = df['close'].shift(1)
    df.loc[0, 'open'] = df.loc[0, 'close']
    df['high'] = df[['open', 'close']].max(axis=1)
    df['low'] = df[['open', 'close']].min(axis=1)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap']]


def fetch_cmc_pro(symbol, start_date, end_date, api_key, convert='USD'):
    """
    Fetch OHLCV data from CoinMarketCap Pro API (requires API key).
    """
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    headers = {"X-CMC_PRO_API_KEY": api_key}
    params = {
        "symbol": symbol,
        "convert": convert,
        "time_start": f"{start_date}T00:00:00",
        "time_end": f"{end_date}T23:59:59",
        "interval": "daily"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching CMC data for {symbol}: {e}")
        return pd.DataFrame()

    j = r.json().get('data', {})
    quotes = j.get('quotes', [])
    rows = []
    for q in quotes:
        ts = q.get('time_open') or q.get('time_close') or q.get('timestamp')
        quote = q.get('quote', {}).get(convert, {})
        rows.append({
            "timestamp": ts,
            "open": quote.get('open'),
            "high": quote.get('high'),
            "low": quote.get('low'),
            "close": quote.get('close'),
            "volume": quote.get('volume'),
            "market_cap": quote.get('market_cap')
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df
