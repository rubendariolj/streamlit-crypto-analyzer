📊 Streamlit Crypto TA Analyzer

Streamlit Crypto TA Analyzer is an interactive web application that lets users perform quick and powerful technical analysis (TA) on any cryptocurrency.
It fetches real market data (from CoinGecko or CoinMarketCap Pro), computes popular TA indicators, runs Monte Carlo simulations, and visualizes everything with Plotly charts — all inside a clean, browser-based interface.

🚀 Key Features
🪙 1. Real-time crypto data

Fetch live OHLCV (Open, High, Low, Close, Volume) data directly from:

CoinGecko API (free, no API key required)

CoinMarketCap Pro API (optional, for Pro users)

📈 2. Core Technical Indicators

Automatically computes:

SMA (20 & 50) — Simple Moving Averages

RSI (14) — Relative Strength Index

MACD (12/26/9) — Moving Average Convergence Divergence

Support and Resistance levels (rolling window method)

Fibonacci retracements and extensions

🎲 3. Monte Carlo price simulation

Simulates thousands of possible future price paths using Geometric Brownian Motion (GBM).

Produces probabilistic scenarios:

📈 Bullish

⚖️ Base

📉 Bearish

Calculates estimated probabilities for each scenario at multiple horizons (1D, 7D, 30D, 90D, 365D).

📊 4. Interactive Plotly charts

Beautiful candlestick charts with overlays:

SMA20 / SMA50

Fibonacci levels

Support & resistance lines

RSI and MACD shown in dedicated subplots

Zoom, pan, and hover for details

Legend positioned neatly below the chart

🧾 5. PDF Report Export

Generate a summary report with:

Technical analysis results

Fibonacci levels

Monte Carlo scenario probabilities

Downloadable as a PDF file (optional chart image embedding if Kaleido is available)

📂 6. CSV Upload Support

Already have your own data?
Just upload a CSV with OHLCV columns — the app automatically detects delimiters and timestamps.

🧠 7. Diagnostics Panel

For developers and advanced users:

View detected column names and data types

Inspect the first few rows of parsed data

See any dropped or cleaned rows

🛠️ Installation
🧩 Requirements

This app requires Python 3.9+ and the following libraries (already listed in requirements.txt):

streamlit
pandas
numpy
requests
plotly
reportlab
matplotlib
mplfinance
kaleido

🪟 Run Locally

Clone the repository

git clone https://github.com/yourusername/streamlit-crypto-analyzer.git
cd streamlit-crypto-analyzer


(Optional) Create a virtual environment

python -m venv venv
venv\\Scripts\\activate   # On Windows
source venv/bin/activate  # On macOS/Linux


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py


Then open the browser at http://localhost:8501
.

☁️ Deploy on Streamlit Cloud

Push your project (including app.py, fetch_data.py, ta_analysis.py, and requirements.txt) to GitHub.

Go to streamlit.io/cloud
.

Click “New app” → connect your GitHub repo → select the branch → click Deploy.

Streamlit Cloud will automatically install dependencies and launch your app online.

⚙️ File Overview
File	Description
app.py	Main Streamlit application (frontend + logic)
fetch_data.py	Handles data fetching from APIs (CoinGecko / CMC)
ta_analysis.py	Contains all TA calculations (RSI, MACD, Fibonacci, etc.)
requirements.txt	Dependencies required for the app
README.md	Project overview and setup guide
💡 Example Use

Enter a crypto symbol (e.g., solana or bitcoin)

Choose your data source (CoinGecko / CoinMarketCap)

Click Fetch & Analyze

View interactive charts and computed indicators

Optionally, download a full PDF report

📸 Screenshots

(Optional: You can upload screenshots of your app interface here)
Example:

/screenshots/
 ├── main_dashboard.png
 ├── montecarlo_table.png
 └── pdf_report_example.png

🧰 Technologies Used

Python 3.11+

Streamlit (UI & interactivity)

Plotly (charting)

Pandas / NumPy (data analysis)

ReportLab (PDF report generation)

Kaleido (static chart image export)

CoinGecko & CoinMarketCap APIs

📬 Support & Contributions

Pull requests, feedback, and feature suggestions are always welcome!

If you find a bug or have an improvement idea:

Open an Issue on GitHub

Or submit a Pull Request with your enhancements

🧑‍💻 Author

Ruben Leon

📧 youaremadetoshinenow@gmail.com

🪙 License

This project is licensed under the MIT License.
You’re free to use, modify, and distribute it — just keep attribution in your fork.
