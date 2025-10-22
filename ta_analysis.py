import numpy as np, pandas as pd, math
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = ema(series, span_short)
    ema_long = ema(series, span_long)
    macd = ema_short - ema_long
    signal = ema(macd, span_signal)
    hist = macd - signal
    return macd, signal, hist

def fib_levels(high, low):
    diff = high - low
    retr = {'0.0%': high, '23.6%': high - diff*0.236, '38.2%': high - diff*0.382, '50.0%': high - diff*0.5, '61.8%': high - diff*0.618, '78.6%': high - diff*0.786, '100.0%': low}
    exts = {'127.2%': high + diff*(1.272-1), '161.8%': high + diff*(1.618-1), '261.8%': high + diff*(2.618-1)}
    return retr, exts

def simple_support_resistance(df, window=20):
    support = float(df['low'].rolling(window=window, min_periods=1).min().iloc[-1])
    resistance = float(df['high'].rolling(window=window, min_periods=1).max().iloc[-1])
    return support, resistance

def monte_carlo_gbm(S0, mu, sigma, days, n_sims=5000, seed=42):
    np.random.seed(seed)
    dt = 1
    rand = np.random.normal((mu - 0.5*sigma*sigma)*dt, sigma*math.sqrt(dt), size=(n_sims, days))
    paths = S0 * np.exp(np.cumsum(rand, axis=1))
    return paths

def compute_probs_from_df(df, support, resistance, horizons=[1,7,30,90,365], n_sims=5000):
    logrets = np.log(df['close'] / df['close'].shift(1)).dropna()
    mu = float(logrets.mean())
    sigma = float(logrets.std())
    S0 = float(df['close'].iloc[-1])
    max_h = max(horizons)
    paths = monte_carlo_gbm(S0, mu, sigma, max_h, n_sims=n_sims)
    results = {}
    for h in horizons:
        sims = paths[:, h-1]
        bull = float((sims >= resistance).mean())
        bear = float((sims <= support).mean())
        base = max(0.0, 1.0 - bull - bear)
        results[h] = {'Prob Bull': round(bull,4), 'Prob Base': round(base,4), 'Prob Bear': round(bear,4), 'Median': round(float(np.median(sims)),4), 'Mean': round(float(np.mean(sims)),4)}
    return results
