
import streamlit as st
import pandas as pd, numpy as np, io, csv, re, sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fetch_data import fetch_coingecko_list, fetch_coingecko_daily, fetch_cmc_pro
from ta_analysis import sma, compute_rsi, compute_macd, fib_levels, simple_support_resistance, compute_probs_from_df
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
import plotly.io as pio

st.set_page_config(layout='wide', page_title='Streamlit Crypto TA Analyzer — Robust v2.2')
st.title('Streamlit Crypto TA Analyzer — Robust v2.2')

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

# ------------------------------------------------------------
# Here we would include all logic from app_v2_1 (unchanged).
# For brevity, this snippet only shows the Kaleido-checking logic.
# In your full app, place this section at the very top (after imports)
# and keep the rest of v2.1 code exactly as-is.
# ------------------------------------------------------------
