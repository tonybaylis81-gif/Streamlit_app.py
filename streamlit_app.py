# streamlit_app.py
"""
Auto Trade AI — Ten Toes in the Market
Paper-trading Streamlit app (in-memory, no DB).

Goal: maximize profit via parameter optimization and automatic daily simulated runs.
Features:
- Batched yfinance data fetching
- Strategy: momentum + SMA + RSI filters
- Fractional shares (buy-by-dollar allocation)
- Backtest and grid-search optimizer to maximize final equity over backtest period (default 3 months)
- No persistent DB: all state stored in-memory or local files if explicitly saved by user
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

st.set_page_config(page_title="Auto Trade AI — Ten Toes in the Market", layout="wide")

# ------------------------
# Defaults & Config
# ------------------------
STARTING_BALANCE = 25000.0
DEFAULT_UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","META","BRK-B","JPM","V",
    "JNJ","WMT","PG","MA","UNH","HD","DIS","BAC","PYPL","ADBE",
    "CMCSA","NFLX","XOM","INTC","CSCO","KO","PFE","NKE","MRK","PEP",
    "COST","ORCL","T","ABT","CRM","MCD","AMGN","TXN","QCOM","AVGO",
    "LLY","BMY","C","SCHW","GS","RTX","MDT","HON","SBUX","GILD"
]
BACKTEST_MONTHS_DEFAULT = 3

# ------------------------
# Helpers: data & indicators
# ------------------------
@st.cache_data(ttl=60*60)
def fetch_history(tickers, period_days=365):
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days)
    try:
        df = yf.download(tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), progress=False, threads=True)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

# ------------------------
# Strategy scoring & execution (in-memory)
# ------------------------
def score_universe(data, universe, lookback_days, sma_days, rsi_max, min_volume, top_k):
    scores = []
    if data is None:
        return scores
    for t in universe:
        try:
            close = data['Close'][t].dropna()
            vol = data['Volume'][t].dropna()
        except Exception:
            continue
        if close.empty or vol.empty:
            continue
        if vol.iloc[-1] < min_volume:
            continue
        lookback_idx = max(0, len(close) - lookback_days - 1)
        past = close.iloc[lookback_idx]
        recent = close.iloc[-1]
        if past == 0:
            continue
        momentum = (recent - past) / past
        sma_val = compute_sma(close, sma_days).iloc[-1] if len(close) >= 1 else None
        if sma_val is not None and recent < sma_val:
            continue
        rsi_val = compute_rsi(close).iloc[-1]
        if pd.isna(rsi_val):
            continue
        if rsi_val > rsi_max:
            continue
        scores.append((t, float(momentum), float(recent), float(rsi_val)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def run_simulation(universe, lookback, sma_days, rsi_max, min_vol, top_k, hold_days, months=BACKTEST_MONTHS_DEFAULT, start_balance=STARTING_BALANCE):
    period_days = max(30, months * 30)
    data = fetch_history(universe, period_days=period_days + 10)
    if data is None:
        return None
    cash = start_balance
    positions = []
    trades = []
    equity_curve = []
    start_date = datetime.utcnow().date() - timedelta(days=period_days)
    dates = pd.date_range(start=start_date, end=datetime.utcnow().date())
    for dt in dates:
        today_iso = dt.date().isoformat()
        # Sell due positions
        to_remove = []
        for pos in positions:
            if dt.date() >= datetime.fromisoformat(pos['sell_date']).date():
                try:
                    price = float(data['Close'][pos['ticker']].loc[:dt.date()].dropna().iloc[-1])
                except Exception:
                    price = pos['avg_price']
                value = pos['shares'] * price
                cash += value
                trades.append({'date': today_iso, 'action':'SELL', 'ticker':pos['ticker'], 'shares':pos['shares'], 'price':price, 'value':value})
                to_remove.append(pos)
        for r in to_remove:
            positions.remove(r)
        # Buy when flat
        if len(positions) == 0:
            selections = score_universe(data, universe, lookback, sma_days, rsi_max, min_vol, top_k)
            max_alloc = cash * 0.1  # max 10% per position generically
            alloc_per = (max_alloc / max(1, len(selections))) if selections else 0
            for t, score, price, rsi in selections:
                if price is None or price <= 0:
                    continue
                shares = alloc_per / price
                value = shares * price
                if value <= 0 or value > cash:
                    continue
                cash -= value
                sell_date = (dt.date() + timedelta(days=hold_days)).isoformat()
                positions.append({'ticker':t, 'shares':shares, 'avg_price':price, 'buy_date':today_iso, 'sell_date':sell_date})
                trades.append({'date': today_iso, 'action':'BUY', 'ticker':t, 'shares':shares, 'price':price, 'value':value})
        # record equity
        market_val = cash
        for pos in positions:
            try:
                p = float(data['Close'][pos['ticker']].loc[:dt.date()].dropna().iloc[-1])
            except Exception:
                p = pos['avg_price']
            market_val += pos['shares'] * p
        equity_curve.append({'date': today_iso, 'equity': market_val})
    df_eq = pd.DataFrame(equity_curve)
    final_equity = market_val
    total_return = final_equity - start_balance
    trades_df = pd.DataFrame(trades)
    win_rate = None
    if not trades_df.empty:
        returns = []
        buys = trades_df[trades_df['action']=='BUY']
        sells = trades_df[trades_df['action']=='SELL']
        for _, b in buys.iterrows():
            s_for = sells[sells['ticker']==b['ticker']]
            if not s_for.empty:
                r = (s_for.iloc[0]['value'] - b['value']) / b['value']
                returns.append(r)
        if returns:
            win_rate = float(np.mean([1 if r > 0 else 0 for r in returns]))
    return {'equity_curve': df_eq, 'final_equity': final_equity, 'total_return': total_return, 'trades': trades_df, 'win_rate': win_rate}

# ------------------------
# Optimizer: grid search to maximize final equity
# ------------------------
def grid_search_optimize(universe, lookback_vals, sma_vals, rsi_vals, topk_vals, hold_vals, min_vol, months, start_balance):
    best = None
    best_params = None
    results = []
    total = len(lookback_vals) * len(sma_vals) * len(rsi_vals) * len(topk_vals) * len(hold_vals)
    i = 0
    for lb in lookback_vals:
        for sma in sma_vals:
            for rsi in rsi_vals:
                for tk in topk_vals:
                    for hd in hold_vals:
                        i += 1
                        res = run_simulation(universe, lb, sma, rsi, min_vol, tk, hd, months=months, start_balance=start_balance)
                        if res is None:
                            continue
                        fe = res['final_equity']
                        results.append({'lookback':lb,'sma':sma,'rsi':rsi,'topk':tk,'hold':hd,'final_equity':fe})
                        if best is None or fe > best:
                            best = fe
                            best_params = {'lookback':lb,'sma':sma,'rsi':rsi,'topk':tk,'hold':hd}
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='final_equity', ascending=False).reset_index(drop=True)
    return best_params, df_sorted

# ------------------------
# UI
# ------------------------
st.title("Auto Trade AI — Ten Toes in the Market")
st.sidebar.header("Controls")

with st.sidebar.form("controls"):
    universe_input = st.text_area("Universe (comma-separated)", value=",".join(DEFAULT_UNIVERSE), height=120)
    universe = [t.strip().upper() for t in universe_input.split(",") if t.strip()]
    topk = st.number_input("Top K (simultaneous positions)", value=10, min_value=1, max_value=50)
    lookback = st.number_input("Momentum lookback days", value=30, min_value=5, max_value=120)
    sma_days = st.number_input("SMA days", value=50, min_value=5, max_value=200)
    rsi_max = st.number_input("RSI max (allow buy)", value=70, min_value=30, max_value=90)
    min_vol = st.number_input("Min volume filter", value=100000, min_value=0)
    hold_days = st.number_input("Hold days", value=5, min_value=1, max_value=30)
    backtest_months = st.number_input("Backtest months", value=3, min_value=1, max_value=24)
    start_balance = st.number_input("Starting balance", value=25000.0, min_value=100.0)
    run_sim = st.form_submit_button("Run Backtest")
    st.form_submit_button(" ")  # dummy to avoid streamlit quirk

st.header("Backtest & Results")

if run_sim:
    with st.spinner("Running backtest..."):
        res = run_simulation(universe, int(lookback), int(sma_days), int(rsi_max), int(min_vol), int(topk), int(hold_days), months=int(backtest_months), start_balance=float(start_balance))
    if res is None:
        st.error("Backtest failed (data fetch).")
    else:
        st.success(f"Backtest complete — final equity ${res['final_equity']:,.2f} (return ${res['total_return']:,.2f})")
        fig, ax = plt.subplots(figsize=(10,4))
        x = pd.to_datetime(res['equity_curve']['date'])
        y = res['equity_curve']['equity']
        ax.plot(x, y)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        st.pyplot(fig)
        st.subheader("Trades (sample)")
        if not res['trades'].empty:
            st.dataframe(res['trades'].head(200))
        st.write(f"Approx win rate: {res['win_rate']}")

st.header("Optimizer — grid search (maximize final equity)")
st.write("This will run multiple backtests over parameter combinations and return the best set found. Be careful: large grids take time and API calls.")

with st.form("opt_form"):
    lookback_vals = st.text_input("Lookback values (comma)", value="20,30,40")
    sma_vals = st.text_input("SMA values (comma)", value="20,50")
    rsi_vals = st.text_input("RSI max values (comma)", value="60,70")
    topk_vals = st.text_input("TopK values (comma)", value="3,5,10")
    hold_vals = st.text_input("Hold days (comma)", value="3,5")
    run_opt = st.form_submit_button("Run optimizer")
    st.form_submit_button(" ")

if run_opt:
    try:
        lb_list = [int(x.strip()) for x in lookback_vals.split(",") if x.strip()]
        sma_list = [int(x.strip()) for x in sma_vals.split(",") if x.strip()]
        rsi_list = [int(x.strip()) for x in rsi_vals.split(",") if x.strip()]
        tk_list = [int(x.strip()) for x in topk_vals.split(",") if x.strip()]
        hd_list = [int(x.strip()) for x in hold_vals.split(",") if x.strip()]
    except Exception as e:
        st.error(f"Parameter parse error: {e}")
        lb_list, sma_list, rsi_list, tk_list, hd_list = [],[],[],[],[]
    if lb_list and sma_list and rsi_list and tk_list and hd_list:
        with st.spinner("Running grid search... this may take a while."):
            best_params, df_results = grid_search_optimize(universe, lb_list, sma_list, rsi_list, tk_list, hd_list, int(min_vol), int(backtest_months), float(start_balance))
        if best_params is None:
            st.error("Optimization failed or returned no results.")
        else:
            st.success(f"Best params found: {best_params}")
            st.dataframe(df_results.head(50))
            # offer download of results CSV
            csv = df_results.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="optim_results.csv">Download results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

st.write("---")
st.write("Developer notes: Goal = maximize profit. Optimizer runs multiple backtests to find parameters that produced highest final equity over the backtest period. This is a simulation — real markets include slippage, fees, and execution risk.")
