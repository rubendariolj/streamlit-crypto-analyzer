Streamlit Crypto Analyzer
-------------------------
This app fetches data from CoinGecko (fallback to CoinMarketCap if API key provided),
computes technical indicators (MA20/50, RSI, MACD), Fibonacci levels, support/resistance,
runs a Monte Carlo GBM scenario simulator, and displays interactive Plotly charts.
It also offers PDF export of the chart and summary.

Run:
  pip install -r requirements.txt
  streamlit run app.py

Files:
  app.py - main Streamlit app
  fetch_data.py - helper for fetching CoinGecko/CMC data (used inside app)
  ta_analysis.py - technical indicator helpers & Monte Carlo
