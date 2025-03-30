import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import argparse
import sqlite3
import os
from tabulate import tabulate

# SQLite database and trade log file names
DB_FILE = 'market_data.db'
TRADE_LOG_FILE = 'trade_log.csv'
INITIAL_INVESTMENT = 100000.0  # Default initial investment ($100,000)

# Fear & Greed Index data collection and storage
def get_fear_greed_data(conn):
    today = datetime.now()
    start_date = today - timedelta(days=730)
    start_date_str = start_date.strftime('%Y-%m-%d')
    url = f'https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date_str}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        historical_data = data['fear_and_greed_historical']['data']
        df = pd.json_normalize(historical_data)
        df['date'] = pd.to_datetime(df['x'], unit='ms').dt.date
        df = df[['date', 'y']].rename(columns={'y': 'fear_greed_index'})
        df.to_sql('fear_greed_index', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Fear & Greed Index data collection error: {e}")

# Stock data collection and storage
def get_stock_data(ticker, conn):
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(Date) FROM stock_data WHERE Ticker='{ticker}'")
        last_date = cursor.fetchone()[0]
        if last_date:
            last_date = datetime.strptime(last_date, '%Y-%m-%d').date()
            start_date = last_date + timedelta(days=1)
        else:
            start_date = (datetime.now() - timedelta(days=730)).date()
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=datetime.now().date())
        if hist.empty:
            return
        hist = hist.reset_index()
        df = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Date'] = df['Date'].dt.date
        df['Ticker'] = ticker
        df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        existing_dates = pd.read_sql(f"SELECT Date FROM stock_data WHERE Ticker='{ticker}'", conn)['Date'].tolist()
        df = df[~df['Date'].astype(str).isin(existing_dates)]
        if not df.empty:
            df['Date'] = df['Date'].astype(str)
            df.to_sql('stock_data', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"{ticker} data collection error: {e}")

# VIX index data collection and storage
def get_vix_data(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM vix_data")
        last_date = cursor.fetchone()[0]
        if last_date:
            last_date = datetime.strptime(last_date, '%Y-%m-%d').date()
            start_date = last_date + timedelta(days=1)
        else:
            start_date = (datetime.now() - timedelta(days=730)).date()
        
        vix_df = yf.download("^VIX", start=start_date, end=datetime.now().date())
        if vix_df.empty:
            return
        vix_df = vix_df[['Close']].reset_index()
        vix_df['Date'] = vix_df['Date'].dt.strftime('%Y-%m-%d')
        vix_df = vix_df.rename(columns={'Close': 'VIX_Close'})
        
        existing_dates = pd.read_sql("SELECT Date FROM vix_data", conn)['Date'].tolist()
        vix_df = vix_df[~vix_df['Date'].isin(existing_dates)]
        if not vix_df.empty:
            cursor.executemany("INSERT INTO vix_data (Date, VIX_Close) VALUES (?, ?)", vix_df[['Date', 'VIX_Close']].values.tolist())
            conn.commit()
    except Exception as e:
        print(f"VIX data collection error: {e}")

# Technical indicator calculation functions
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def calculate_weekly_rsi(df, period=14):
    df_weekly = df.resample('W', on='Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df_weekly = calculate_rsi(df_weekly, period)
    df_weekly = df_weekly.rename(columns={f'RSI_{period}': 'Weekly_RSI'})
    return df_weekly['Weekly_RSI']

def calculate_macd(df, short_period=12, long_period=26, signal_period=9, prefix=''):
    df[f'EMA{short_period}'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df[f'EMA{long_period}'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df[f'{prefix}MACD'] = df[f'EMA{short_period}'] - df[f'EMA{long_period}']
    df[f'{prefix}Signal'] = df[f'{prefix}MACD'].ewm(span=signal_period, adjust=False).mean()
    df[f'{prefix}MACD_Histogram'] = df[f'{prefix}MACD'] - df[f'{prefix}Signal']
    return df

def calculate_bollinger_bands(df, period=20):
    df['SMA20'] = df['Close'].rolling(window=period).mean()
    df['STD20'] = df['Close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA20'] + (2 * df['STD20'])
    df['Lower_Band'] = df['SMA20'] - (2 * df['STD20'])
    df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']
    return df

def calculate_stochastic_oscillator(df, period=14):
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['Percent_K'] = (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']) * 100
    df['Percent_D'] = df['Percent_K'].rolling(window=3).mean()
    return df

def calculate_obv(df):
    df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).astype(int)).cumsum()
    return df

def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_sma(df, periods=[5, 10, 50, 200]):
    for period in periods:
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
    return df

def calculate_volume_change(df):
    df['Volume_Change'] = df['Volume'].pct_change()
    return df

def calculate_vwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_all_indicators(df):
    df = calculate_rsi(df, period=14)
    df = calculate_rsi(df, period=5)
    df = calculate_macd(df, short_period=12, long_period=26, signal_period=9, prefix='')
    df = calculate_macd(df, short_period=5, long_period=13, signal_period=1, prefix='Short_')
    df = calculate_bollinger_bands(df)
    df = calculate_stochastic_oscillator(df)
    df = calculate_obv(df)
    df = calculate_atr(df)
    df = calculate_sma(df, periods=[5, 10, 50, 200])
    df = calculate_volume_change(df)
    df = calculate_vwap(df)
    weekly_rsi = calculate_weekly_rsi(df)
    df = df.join(weekly_rsi, on='Date')
    return df

# Save indicators to database
def save_indicators_to_db(df, ticker, conn):
    indicators = ['RSI_14', 'RSI_5', 'MACD', 'Signal', 'MACD_Histogram', 'Short_MACD', 'Short_Signal', 'Short_MACD_Histogram',
                  'SMA20', 'Upper_Band', 'Lower_Band', 'BB_Width', 'Percent_K', 'Percent_D', 'OBV', 'ATR', 'SMA5', 'SMA10', 'SMA50', 'SMA200',
                  'Volume_Change', 'VWAP', 'Weekly_RSI']
    df_indicators = df[['Date'] + [col for col in indicators if col in df.columns]].copy()
    df_indicators['Ticker'] = ticker
    df_indicators['Date'] = df_indicators['Date'].dt.strftime('%Y-%m-%d')
    existing_dates = pd.read_sql(f"SELECT Date FROM technical_indicators WHERE Ticker='{ticker}'", conn)['Date'].tolist()
    df_indicators = df_indicators[~df_indicators['Date'].isin(existing_dates)]
    if not df_indicators.empty:
        df_indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)

# Load trade log into database
def load_trade_log_to_db(conn):
    if not os.path.exists(TRADE_LOG_FILE):
        return
    df = pd.read_csv(TRADE_LOG_FILE)
    df['Shares'] = df['Shares'].astype(int)
    df.to_sql('trade_log', conn, if_exists='replace', index=False)

# Process portfolio and update state
def process_portfolio(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio_state")
    trade_log = pd.read_sql("SELECT * FROM trade_log ORDER BY Date", conn)
    if trade_log.empty:
        return
    
    cash = 0.0
    holdings = {}
    transaction_cost_rate = 0.001
    
    for _, trade in trade_log.iterrows():
        date = trade['Date']
        ticker = trade['Ticker']
        action = trade['Action'].lower()
        shares = int(trade['Shares'])
        price = trade['Price']
        trade_amount = shares * price
        cost = trade_amount * transaction_cost_rate
        
        if ticker not in holdings:
            holdings[ticker] = 0
        
        if action == 'buy':
            cash -= trade_amount + cost
            holdings[ticker] += shares
        elif action == 'sell':
            if holdings[ticker] >= shares:
                cash += trade_amount - cost
                holdings[ticker] -= shares
            else:
                print(f"Error: Not enough shares to sell {ticker} on {date}")
                continue
        elif action == 'hold':
            holdings[ticker] = shares
        
        cursor.execute("INSERT INTO portfolio_state (Date, Cash, Ticker, Shares) VALUES (?, ?, ?, ?)",
                       (date, cash, ticker, holdings[ticker]))
    
    conn.commit()

# Text-based portfolio visualization
def visualize_portfolio_text(conn, tickers, initial_investment):
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(Date) FROM portfolio_state")
    latest_date = cursor.fetchone()[0]
    if not latest_date:
        return
    
    portfolio_data = pd.read_sql(f"SELECT * FROM portfolio_state WHERE Date='{latest_date}'", conn)
    if portfolio_data.empty:
        return
    
    cash = portfolio_data['Cash'].iloc[0]
    holdings = {row['Ticker']: row['Shares'] for _, row in portfolio_data.iterrows()}
    
    prices = {}
    for ticker in tickers:
        cursor.execute(f"SELECT Close FROM stock_data WHERE Ticker='{ticker}' ORDER BY Date DESC LIMIT 1")
        result = cursor.fetchone()
        prices[ticker] = result[0] if result else 0.0
    
    total_value = cash
    portfolio_values = []
    for ticker in tickers:
        if ticker in holdings:
            value = holdings[ticker] * prices.get(ticker, 0.0)
            total_value += value
            portfolio_values.append({
                'Asset': ticker,
                'Shares': holdings[ticker],
                'Price': f"${prices.get(ticker, 0.0):.2f}",
                'Value': f"${value:.2f}",
                'Weight (%)': f"{(value / total_value * 100):.2f}%"
            })
    
    portfolio_values.append({
        'Asset': 'Cash',
        'Shares': '-',
        'Price': '-',
        'Value': f"${cash:.2f}",
        'Weight (%)': f"{(cash / total_value * 100):.2f}%"
    })
    
    return_rate = ((total_value - initial_investment) / initial_investment) * 100
    
    print(f"\n📊 Portfolio Composition on {latest_date}")
    print(f"💰 Initial Investment: ${initial_investment:.2f}")
    print(f"💰 Current Portfolio Value: ${total_value:.2f}")
    print(f"📈 Return Rate: {return_rate:.2f}%")
    df_portfolio = pd.DataFrame(portfolio_values)
    print(tabulate(df_portfolio, headers='keys', tablefmt='fancy_grid', showindex=False))

# Weight adjustment notifications with buy/sell signals
def check_weight_adjustment(df_dict, tickers):
    for ticker in tickers:
        df = df_dict[ticker]
        latest = df.iloc[-1]
        
        if latest['RSI_14'] > 70:
            print(f"⚠️ {ticker} RSI_14 > 70: Consider selling or reducing weight.")
        elif latest['RSI_14'] < 30:
            print(f"⚠️ {ticker} RSI_14 < 30: Consider buying or increasing weight.")
        
        if latest['MACD'] > latest['Signal'] and latest['MACD_Histogram'] > 0:
            print(f"📈 {ticker} MACD upward trend: Consider buying or increasing weight.")
        elif latest['MACD'] < latest['Signal'] and latest['MACD_Histogram'] < 0:
            print(f"📉 {ticker} MACD downward trend: Consider selling or reducing weight.")
        
        if latest['Close'] > latest['Upper_Band']:
            print(f"🔺 {ticker} above Upper Bollinger Band: Consider selling or reducing weight.")
        elif latest['Close'] < latest['Lower_Band']:
            print(f"🔻 {ticker} below Lower Bollinger Band: Consider buying or increasing weight.")

# Calculate initial investment from trade log
def get_initial_investment(conn):
    if not os.path.exists(TRADE_LOG_FILE):
        return INITIAL_INVESTMENT
    
    trade_log = pd.read_csv(TRADE_LOG_FILE)
    hold_trades = trade_log[trade_log['Action'].str.lower() == 'hold']
    
    if hold_trades.empty:
        return INITIAL_INVESTMENT
    
    initial_value = 0.0
    for _, trade in hold_trades.iterrows():
        initial_value += trade['Shares'] * trade['Price']
    
    return initial_value

# Main function
def main():
    parser = argparse.ArgumentParser(description='Collect 2 years of data, store in SQLite, and manage portfolio')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='Stock tickers (comma-separated, default: TSLA,TSLL)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    conn = sqlite3.connect(DB_FILE)

    # Create database tables
    conn.execute('''CREATE TABLE IF NOT EXISTS fear_greed_index (date TEXT PRIMARY KEY, fear_greed_index REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_data (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, PRIMARY KEY (Ticker, Date))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS vix_data (Date TEXT PRIMARY KEY, VIX_Close REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS technical_indicators (Ticker TEXT, Date TEXT, RSI_14 REAL, RSI_5 REAL, MACD REAL, Signal REAL, MACD_Histogram REAL, Short_MACD REAL, Short_Signal REAL, Short_MACD_Histogram REAL, SMA20 REAL, Upper_Band REAL, Lower_Band REAL, BB_Width REAL, Percent_K REAL, Percent_D REAL, OBV INTEGER, ATR REAL, SMA5 REAL, SMA10 REAL, SMA50 REAL, SMA200 REAL, Volume_Change REAL, VWAP REAL, Weekly_RSI REAL, PRIMARY KEY (Ticker, Date))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Ticker TEXT, Action TEXT, Shares INTEGER, Price REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS portfolio_state (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Cash REAL, Ticker TEXT, Shares INTEGER)''')

    # Collect and store data
    get_fear_greed_data(conn)
    for ticker in tickers:
        get_stock_data(ticker, conn)
    get_vix_data(conn)

    # Process trade log and portfolio state
    load_trade_log_to_db(conn)
    process_portfolio(conn)

    # Load data and calculate indicators
    df_dict = {}
    for ticker in tickers:
        df = pd.read_sql(f'SELECT * FROM stock_data WHERE Ticker="{ticker}"', conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = calculate_all_indicators(df)
        df_dict[ticker] = df
        save_indicators_to_db(df, ticker, conn)

    # Get initial investment
    initial_investment = get_initial_investment(conn)

    # Visualize portfolio and provide weight adjustment notifications
    visualize_portfolio_text(conn, tickers, initial_investment)
    check_weight_adjustment(df_dict, tickers)

    conn.close()

if __name__ == "__main__":
    main()
