import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import argparse
import sqlite3
import os

# SQLite 데이터베이스 파일명 및 거래 내역 파일명
DB_FILE = 'market_data.db'
TRADE_LOG_FILE = 'trade_log.csv'

# Fear & Greed Index 데이터 수집 및 저장 함수
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
        print("Fear & Greed Index data saved to SQLite.")
    except Exception as e:
        print(f"Fear & Greed Index data collection error: {e}")

# 주식 데이터 수집 및 저장 함수
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
            print(f"No new data available for {ticker}.")
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
            print(f"New data for {ticker} saved to SQLite.")
        else:
            print(f"No new data to save for {ticker} after filtering duplicates.")
    except Exception as e:
        print(f"{ticker} data collection error: {e}")

# VIX 지수 데이터 수집 및 저장 함수
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
            print("No new data available for VIX.")
            return
        vix_df = vix_df[['Close']].reset_index()
        vix_df['Date'] = vix_df['Date'].dt.strftime('%Y-%m-%d')
        vix_df = vix_df.rename(columns={'Close': 'VIX_Close'})
        
        existing_dates = pd.read_sql("SELECT Date FROM vix_data", conn)['Date'].tolist()
        vix_df = vix_df[~vix_df['Date'].isin(existing_dates)]
        if not vix_df.empty:
            cursor = conn.cursor()
            data_to_insert = vix_df[['Date', 'VIX_Close']].values.tolist()
            cursor.executemany("INSERT INTO vix_data (Date, VIX_Close) VALUES (?, ?)", data_to_insert)
            conn.commit()
            print("New VIX data saved to SQLite.")
        else:
            print("No new data to save for VIX after filtering duplicates.")
    except Exception as e:
        print(f"VIX data collection error: {e}")

# 기술적 지표 계산 함수들 (생략: 기존 코드 유지)
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def calculate_weekly_rsi(df, period=14):
    df_weekly = df.resample('W', on='Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
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

# 지표를 데이터베이스에 저장하는 함수
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
        print(f"Indicators for {ticker} saved to SQLite.")
    else:
        print(f"No new indicators to save for {ticker}.")

# 포트폴리오 관리 로직

# 거래내역을 데이터베이스에 저장하는 함수
def load_trade_log_to_db(conn):
    if not os.path.exists(TRADE_LOG_FILE):
        print(f"No trade log file found at {TRADE_LOG_FILE}.")
        return
    df = pd.read_csv(TRADE_LOG_FILE)
    df['Shares'] = df['Shares'].astype(int)  # 소수점 거래 금지
    df.to_sql('trade_log', conn, if_exists='replace', index=False)
    print("Trade log loaded into SQLite.")

# 포트폴리오 상태를 초기화하고 거래를 처리하는 함수
def process_portfolio(conn):
    cursor = conn.cursor()
    
    # 기존 portfolio_state 테이블 비우기
    cursor.execute("DELETE FROM portfolio_state")
    
    # 거래내역 로드
    trade_log = pd.read_sql("SELECT * FROM trade_log ORDER BY Date", conn)
    if trade_log.empty:
        print("No trade log data to process.")
        return
    
    cash = 0.0
    holdings = {}
    transaction_cost_rate = 0.001  # 0.1% 거래 비용
    
    for _, trade in trade_log.iterrows():
        date = trade['Date']
        ticker = trade['Ticker']
        action = trade['Action'].lower()
        shares = int(trade['Shares'])  # 정수로 변환
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
            # 초기 보유량 설정
            holdings[ticker] = shares
        
        # 현재 상태 저장
        cursor.execute("INSERT INTO portfolio_state (Date, Cash, Ticker, Shares) VALUES (?, ?, ?, ?)",
                       (date, cash, ticker, holdings[ticker]))
    
    conn.commit()
    print("Portfolio state processed and saved to SQLite.")

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description='과거 2년간의 데이터를 수집하여 SQLite에 저장하고 포트폴리오를 관리')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='주식 티커 (쉼표로 구분, 기본: TSLA,TSLL)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    # SQLite 데이터베이스 연결
    conn = sqlite3.connect(DB_FILE)

    # 테이블 생성
    conn.execute('''CREATE TABLE IF NOT EXISTS fear_greed_index (
                        date TEXT PRIMARY KEY,
                        fear_greed_index REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_data (
                        Ticker TEXT,
                        Date TEXT,
                        Open REAL,
                        High REAL,
                        Low REAL,
                        Close REAL,
                        Volume INTEGER,
                        PRIMARY KEY (Ticker, Date))''')  # 수정된 부분
    conn.execute('''CREATE TABLE IF NOT EXISTS vix_data (
                        Date TEXT PRIMARY KEY,
                        VIX_Close REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS technical_indicators (
                        Ticker TEXT,
                        Date TEXT,
                        RSI_14 REAL,
                        RSI_5 REAL,
                        MACD REAL,
                        Signal REAL,
                        MACD_Histogram REAL,
                        Short_MACD REAL,
                        Short_Signal REAL,
                        Short_MACD_Histogram REAL,
                        SMA20 REAL,
                        Upper_Band REAL,
                        Lower_Band REAL,
                        BB_Width REAL,
                        Percent_K REAL,
                        Percent_D REAL,
                        OBV INTEGER,
                        ATR REAL,
                        SMA5 REAL,
                        SMA10 REAL,
                        SMA50 REAL,
                        SMA200 REAL,
                        Volume_Change REAL,
                        VWAP REAL,
                        Weekly_RSI REAL,
                        PRIMARY KEY (Ticker, Date))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS trade_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Date TEXT,
                        Ticker TEXT,
                        Action TEXT,
                        Shares INTEGER,
                        Price REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS portfolio_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Date TEXT,
                        Cash REAL,
                        Ticker TEXT,
                        Shares INTEGER)''')

    # 데이터 수집 및 저장
    print("Starting data collection...")
    get_fear_greed_data(conn)
    for ticker in tickers:
        get_stock_data(ticker, conn)
    get_vix_data(conn)

    # 거래내역 처리 및 포트폴리오 상태 계산
    load_trade_log_to_db(conn)
    process_portfolio(conn)

    # 데이터 로드 및 지표 계산
    df_dict = {}
    for ticker in tickers:
        df = pd.read_sql(f'SELECT * FROM stock_data WHERE Ticker="{ticker}"', conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = calculate_all_indicators(df)
        df_dict[ticker] = df
        print(f"Technical indicators calculation completed for {ticker}")
        save_indicators_to_db(df, ticker, conn)

    conn.close()
    print("All data collection, portfolio processing, and indicator calculations completed.")

if __name__ == "__main__":
    main()
