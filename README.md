# SmartTrader Streamlit App

This is a Streamlit-based stock simulator and strategy tester that uses Yahoo Finance data.
It features:
- Momentum + RSI strategy
- Fractional share trading simulation
- SQLite storage for persistence
- Backtesting over the last 3 months
- Visualization of performance and trades

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment
You can deploy this app to Streamlit Cloud easily:
1. Push these files to a GitHub repository.
2. Visit [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repo and set `streamlit_app.py` as the entry point.
