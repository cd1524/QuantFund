import yfinance as yf
import pandas as pd
import numpy as np
import time
import joblib
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import alpaca_trade_api as tradeapi
from xgboost import XGBClassifier

# Set up logging for better debugging and error tracking.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALPACA_API_KEY = "{api_key}" 
ALPACA_SECRET_KEY = "{secret_key}" 
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# Cooldown period to prevent wash trades
last_trade_time = None
cooldown_period = 5  # seconds

# Load or initialize the model
try:
    clf = joblib.load('model.pkl')
    logging.info("Model loaded from file.")
except FileNotFoundError:
    clf = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    logging.info("No existing model found. A new model will be trained.")

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volatility(data, window=20):
    return data['Close'].pct_change().rolling(window).std() * np.sqrt(window)

def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

def calculate_bollinger_bands(data, window=20):
    # Ensure that we are working with a Series for the 'Close' price.
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        # If 'Close' is a DataFrame (e.g., due to MultiIndex columns), select the first column.
        close = close.iloc[:, 0]
    
    # Calculate the middle band (simple moving average)
    bb_middle = close.rolling(window=window).mean()
    data['BB_Middle'] = bb_middle

    # Calculate standard deviation and then the upper and lower Bollinger Bands
    std = close.rolling(window=window).std().fillna(0)
    data['BB_Upper'] = bb_middle + (std * 2)
    data['BB_Lower'] = bb_middle - (std * 2)

def calculate_momentum(data, window=10):
    return data['Close'].diff(window)

def calculate_roc(data, window=10):
    return (data['Close'].diff(window) / data['Close'].shift(window)) * 100

def calculate_vix_features(start_date, end_date, interval='1h'):
    """
    Download VIX data and calculate additional features.
    Note: This is a simplified incorporation of VIX features.
    """
    try:
        vix_data = yf.download('^VIX', start=start_date, end=end_date, interval=interval)
        if vix_data.empty:
            logging.warning("VIX data is empty!")
            return None

        # Calculate VIX percent change and a couple of moving averages
        vix_data['VIX_pct_change'] = vix_data['Close'].pct_change()
        vix_data['VIX_SMA_10'] = vix_data['Close'].rolling(window=10).mean()
        vix_data['VIX_SMA_50'] = vix_data['Close'].rolling(window=50).mean()

        return vix_data[['Close', 'VIX_pct_change', 'VIX_SMA_10', 'VIX_SMA_50']].rename(
            columns={'Close': 'VIX_Close'}
        )
    except Exception as e:
        logging.error(f"Error downloading or processing VIX data: {e}")
        return None

def execute_trade(signal, ticker, current_price):
    """
    Execute a trade using Alpaca API.
    """
    global last_trade_time
    try:
        # Check cooldown period
        if last_trade_time and (time.time() - last_trade_time) < cooldown_period:
            logging.info(f"Skipping trade due to cooldown: {signal} {ticker}")
            return

        limit_price = round(float(current_price), 2)

        # Check current position
        try:
            position = api.get_position(ticker)
            available_qty = int(position.qty)
        except tradeapi.rest.APIError:
            available_qty = 0  # No position held

        # Fetch account info for cash balance
        account = api.get_account()
        available_cash = float(account.cash)
        min_cash_reserve = 1000  # reserve amount
        available_cash -= min_cash_reserve

        qty = int(available_cash // limit_price)

        if signal == "BUY":
            if qty > 0:
                api.submit_order(symbol=ticker, qty=qty, side='buy', type='limit',
                                 limit_price=limit_price, time_in_force='gtc')
                logging.info(f"Executed BUY order for {ticker} with quantity {qty}")
            else:
                logging.info(f"Insufficient cash to buy {ticker}")
        elif signal == "SELL":
            qty = min(qty, available_qty)
            if qty > 0:
                api.submit_order(symbol=ticker, qty=qty, side='sell', type='limit',
                                 limit_price=limit_price, time_in_force='gtc')
                logging.info(f"Executed SELL order for {ticker} with quantity {qty}")
            else:
                logging.info(f"No available quantity to SELL for {ticker}")

        last_trade_time = time.time()
    except Exception as e:
        logging.error(f"Trade Execution Error: {e}")

def run_trading_strategy():
    global clf

    # Define the period for data
    period = '180d'
    interval = '1h'

    # Step 1: Fetch SPY data
    spy_data = yf.download('SPY', period=period, interval=interval)
    if spy_data.empty:
        logging.error("SPY data download failed!")
        return

    # Step 2: Calculate technical indicators for SPY
    spy_data['Momentum'] = calculate_momentum(spy_data)
    spy_data['ROC'] = calculate_roc(spy_data)
    spy_data['RSI'] = calculate_rsi(spy_data)
    spy_data['SMA_20'] = spy_data['Close'].rolling(window=20).mean()
    spy_data['SMA_50'] = spy_data['Close'].rolling(window=50).mean()
    spy_data['Volatility'] = calculate_volatility(spy_data)
    calculate_macd(spy_data)
    calculate_bollinger_bands(spy_data)

    # Step 3: Incorporate VIX data and features
    start_date = spy_data.index[0].strftime("%Y-%m-%d")
    end_date = spy_data.index[-1].strftime("%Y-%m-%d")
    vix_features = calculate_vix_features(start_date, end_date, interval)
    if vix_features is not None:
        # Merge VIX features with SPY data on the index (time)
        data = pd.merge_asof(spy_data.sort_index(), 
                             vix_features.sort_index(), 
                             left_index=True, right_index=True, direction='backward')
    else:
        logging.warning("VIX features not available. Continuing with SPY data only.")
        data = spy_data.copy()

    # Step 4: Stationarity check (ADF Test) on SPY Close
    try:
        result = adfuller(data['Close'].dropna())
        logging.info(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    except Exception as e:
        logging.error(f"ADF Test Error: {e}")

    # Step 5: Prepare features and target for the ML model
    data.dropna(inplace=True)

    # Define features including the new VIX features (if available)
    features = ['Momentum', 'ROC', 'RSI', 'SMA_20', 'SMA_50', 'Volatility',
                'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']
    # If VIX features exist, add them
    if 'VIX_Close' in data.columns:
        features.extend(['VIX_Close', 'VIX_pct_change', 'VIX_SMA_10', 'VIX_SMA_50'])

    # Define target: 1 if next period’s close is higher, else 0
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    X = data[features]
    y = data['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 6: Train ML model and predict signal
    clf.fit(X_train, y_train)
    data['ML_Signal'] = clf.predict(X_scaled)
    data['Signal'] = np.where(data['ML_Signal'] == 1, 'BUY', 'SELL')

    # Save the updated model
    joblib.dump(clf, 'model.pkl')
    logging.info("Model saved to file.")

    # Debug: Log the distribution of signals
    logging.info("Signal distribution:")
    logging.info(data['Signal'].value_counts())

    # Step 7: Execute trades for the last few data points
    # Here, we use the most recent 5 observations to simulate trades.
    for index, row in data.tail(5).iterrows():
        signal = row['Signal']
        current_price = row['Close']
        execute_trade(signal, "SPY", current_price)

    # Optional: Log the final few rows for review
    logging.info(data[['Close', 'Momentum', 'SMA_20', 'SMA_50', 'Volatility',
                         'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Signal']].tail(5))
    logging.info("Trading strategy execution completed.")

if __name__ == "__main__":
    # Run the trading strategy in an infinite loop (with a 10-minute interval)
    while True:
        run_trading_strategy()
        time.sleep(600)  # Sleep for 10 minutes before next run