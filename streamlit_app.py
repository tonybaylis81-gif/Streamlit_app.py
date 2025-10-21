# SmartTrader Streamlit App
# (Full app content already present in your canvas, just simplified placeholder here)
print("SmartTrader Streamlit app placeholder")
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 SmartTrader Dashboard", layout="wide")

st.title("📊 SmartTrader – Automated Stock Insights")

# Sidebar Inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter stock symbol (e.g. AAPL, TSLA, MSFT):", "AAPL")
period = st.sidebar.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval = st.sidebar.selectbox("Select interval:", ["1d", "1h", "30m", "15m", "5m"])

# Load data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, period, interval)

st.subheader(f"📅 Price Data for {ticker}")
st.line_chart(data["Close"])

# Calculate Moving Averages
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

# Calculate RSI using `ta`
data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()

# Show Data and Indicators
st.subheader("📈 Moving Averages and RSI")
tab1, tab2 = st.tabs(["Chart", "RSI Indicator"])

with tab1:
    st.line_chart(data[["Close", "SMA_20", "SMA_50"]])

with tab2:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data.index, data["RSI"], color="purple", label="RSI")
    ax.axhline(70, color="red", linestyle="--")
    ax.axhline(30, color="green", linestyle="--")
    ax.set_title("Relative Strength Index (RSI)")
    ax.legend()
    st.pyplot(fig)

st.write("✅ Data last updated:", data.index[-1])
