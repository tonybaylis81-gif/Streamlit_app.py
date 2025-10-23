# --- from streamlit_autorefresh import st_autorefresh
# --- Import Libraries ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import ta  # technical analysis library

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
    start = end - pd.DateOffset(months=int(period.replace("mo","").replace("y","12")))
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, period)

# --- Technical Indicators ---
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

rsi_indicator = ta.momentum.RSIIndicator(data["Close"], window=14)
data["RSI"] = rsi_indicator.rsi()

# --- Display Tabs ---
st.subheader(f"📈 {ticker} Analysis")

tab1, tab2 = st.tabs(["📊 Price & Moving Averages", "📉 RSI Indicator"])

with tab1:
    st.line_chart(data[["Close", "SMA_20", "SMA_50"]])

with tab2:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data.i

# Refresh every 5 import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Page setup ---
st.set_page_config(page_title="📈 SmartTrader Dashboard", layout="wide")

# --- Auto-refresh every 5 minutes ---
st_autorefresh(interval=300000, limit=None, key="datarefresh")

st.title("📊 SmartTrader – Automated Stock Insights")

# --- Sidebar Inputs ---
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter stock symbol (e.g. AAPL, TSLA, MSFT):", "AAPL")
period = st.sidebar.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval = st.sidebar.selectbox("Select interval:", ["1d", "1h", "30m", "15m", "5m"])

# --- Load Data ---
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, period, interval)

st.subheader(f"📅 Price Data for {ticker}")
st.line_chart(data["Close"])

# --- Technical Indicators ---
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

# Handle potential multi-column yfinance returns
close_prices = data["Close"]
if isinstance(close_prices, pd.DataFrame):
    close_prices = close_prices.iloc[:, 0]

data["RSI"] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()

# --- Chart Tabs ---
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

# --- 5-Day Summary Table ---
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

def highlight_signal(row):
    if "BUY" in row["Signal"]:
        color = "background-color: #d4edda;"
    elif "SELL" in row["Signal"]:
        color = "background-color: #f8d7da;"
    else:
        color = "background-color: #f0f0f0;"
    return [color] * len(row)

st.dataframe(summary.style.apply(highlight_signal, axis=1))

# --- AI Price Prediction ---
st.subheader("🤖 AI Prediction (Next 5 Days)")

df = data[["Close", "SMA_20", "SMA_50", "RSI"]].dropna().copy()
df["Target"] = df["Close"].shift(-5)
df.dropna(inplace=True)

X = df[["Close", "SMA_20", "SMA_50", "RSI"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

latest = X.tail(1)
predicted_price = model.predict(latest)[0]
current_price = df["Close"].iloc[-1]
trend = "📈 Likely to RISE" if predicted_price > current_price else "📉 Likely to FALL"

col1, col2 = st.columns(2)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Predicted Price (5 days ahead)", f"${predicted_price:.2f}", trend)

st.write("✅ Data last updated:", data.index[-1])
 (300,000 ms)
st_autorefresh(interval=300000, limit=None, key="datarefresh")
 Summary st.set_page_config(page_title="📈 SmartTrader Dashboard", layout="wide")
# --- AI Price Prediction ---
st.subheader("🤖 AI Prediction (Next 5 Days)")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare the data
df = data[["Close", "SMA_20", "SMA_50", "RSI"]].dropna().copy()
df["Target"] = df["Close"].shift(-5)  # predict 5 days ahead
df.dropna(inplace=True)

X = df[["Close", "SMA_20", "SMA_50", "RSI"]]
y = df["Target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict the next 5 days
latest = X.tail(1)
predicted_price = model.predict(latest)[0]

current_price = df["Close"].iloc[-1]
trend = "📈 Likely to RISE" if predicted_price > current_price else "📉 Likely to FALL"

# Display results
col1, col2 = st.columns(2)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Predicted Price (5 days ahead)", f"${predicted_price:.2f}", trend)

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
