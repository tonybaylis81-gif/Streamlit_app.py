# --- Import Libraries ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import ta  # technical analysis library
import sqlite3

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="📈 SmartTrader Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 SmartTrader Dashboard")
st.write("An AI-powered stock analysis dashboard that helps you make smarter trading decisions.")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", value="AAPL")
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# --- Fetch Stock Data ---
@st.cache_data
def load_data(ticker, period):
    end = dt.datetime.now()
    if "mo" in period:
        months = int(period.replace("mo", ""))
    elif "y" in period:
        months = int(period.replace("y", "")) * 12
    else:
        months = 6
    start = end - pd.DateOffset(months=months)
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, period)

# --- Technical Indicators ---
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

rsi_indicator = ta.momentum.RSIIndicator(data["Close"], window=14)
data["RSI"] = rsi_indicator.rsi()

# --- Store data to SQLite (optional local storage example) ---
try:
    conn = sqlite3.connect("stock_data.db")
    data.to_sql("stocks", conn, if_exists="replace", index=True)
    conn.close()
except Exception as e:
    st.warning(f"Database write skipped: {e}")

# --- Display Tabs ---
st.subheader(f"📈 {ticker} Analysis")

tab1, tab2 = st.tabs(["📊 Price & Moving Averages", "📉 RSI Indicator"])

with tab1:
    st.line_chart(data[["Close", "SMA_20", "SMA_50"]])

with tab2:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data.index, data["RSI"], color="purple", label="RSI")
    ax.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    ax.set_title("Relative Strength Index (RSI)")
    ax.legend()
    st.pyplot(fig)

# --- Prediction Placeholder (basic demo logic) ---
predicted_price = data["Close"].iloc[-1] * (1 + np.random.uniform(-0.03, 0.03))
trend = "🔺" if predicted_price > data["Close"].iloc[-1] else "🔻"

# --- Metrics Dashboard ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
col2.metric("Predicted Price (5 days ahead)", f"${predicted_price:.2f}", trend)
col3.metric("RSI (14-day)", f"{data['RSI'].iloc[-1]:.2f}")

# --- Update Info ---
st.write("✅ Data last updated:", data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))

# --- Auto Refresh (every 5 minutes = 300,000 ms) ---
st_autorefresh(interval=300000, limit=None, key="datarefresh")

# --- Footer ---
st.caption("Data provided by Yahoo Finance. App auto-refreshes every 5 minutes.")
