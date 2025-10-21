# --- from streamlit_autorefresh import st_autorefresh

# Refresh every 5 minutes (300,000 ms)
st_autorefresh(interval=300000, limit=None, key="datarefresh")
 Summary st.set_page_config(page_title="📈 SmartTrader Dashboard", layout="wide")

# Auto-refresh every 5 minutes (300,000 ms)
st_autorefresh = st.experimental_rerun  # For safety in Streamlit Cloud
st_autorefresh_interval = 300000  # milliseconds = 5 minutes
st_autorefresh_count = st.experimental_data_editor  # Placeholder

st_autorefresh = st.experimental_rerun
st_autorefresh_interval = 300000

# Actual refresh
st_autorefresh = st.experimental_rerun
st_autorefresh_interval = 300000
 ---
st.subheader("📆 Last 5 Trading Days Summary")

summary = data[["Close", "RSI", "SMA_20", "SMA_50"]].tail(5).copy()

def signal_for_row(row):
    if row["RSI"] < 30 and row["SMA_20"] > row["SMA_50"]:
        return "🟢 BUY"
    elif row["RSI"] > 70 and row["SMA_20"] < row["SMA_50"]:
        return "🔴 SELL"
    else:
        return "⚪ HOLD"

summary["Signal"] = summary.apply(signal_for_row, axis=1)
summary = summary.round(2)
summary.index = summary.index.strftime("%Y-%m-%d")

# Add colored styling
def highlight_signal(row):
    color = ""
    if "BUY" in row["Signal"]:
        color = "background-color: #d4edda;"   # light green
    elif "SELL" in row["Signal"]:
        color = "background-color: #f8d7da;"   # light red
    else:
        color = "background-color: #f0f0f0;"   # light grey
    return [color] * len(row)

styled_summary = summary.style.apply(highlight_signal, axis=1)

st.dataframe(styled_summary)
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

# Calculate RSI using `ta`# Calculate RSI using `ta`
close_prices = data["Close"]

# Some tickers may return multi-column data (especially if yfinance returns OHLC data as a DataFrame)
if isinstance(close_prices, pd.DataFrame):
    close_prices = close_prices.iloc[:, 0]

rsi_indicator = ta.momentum.RSIIndicator(close_prices, window=14)
data["RSI"] = rsi_indicator.rsi()

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
