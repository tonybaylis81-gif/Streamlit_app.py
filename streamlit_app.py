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
