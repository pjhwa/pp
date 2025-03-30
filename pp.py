
import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import argparse
import sqlite3

# SQLite 데이터베이스 파일명
DB_FILE = 'market_data.db'

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

# VIX 지수 데이터 수집 및 저장 함수 (수정됨)
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
        vix_df['Date'] = vix_df['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string explicitly
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

# 기술적 지표 계산 함수들

### RSI (Relative Strength Index)
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

### 주간 RSI 계산
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

### MACD (Moving Average Convergence Divergence)
def calculate_macd(df, short_period=12, long_period=26, signal_period=9, prefix=''):
    df[f'EMA{short_period}'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df[f'EMA{long_period}'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df[f'{prefix}MACD'] = df[f'EMA{short_period}'] - df[f'EMA{long_period}']
    df[f'{prefix}Signal'] = df[f'{prefix}MACD'].ewm(span=signal_period, adjust=False).mean()
    df[f'{prefix}MACD_Histogram'] = df[f'{prefix}MACD'] - df[f'{prefix}Signal']
    return df

### Bollinger Bands
def calculate_bollinger_bands(df, period=20):
    df['SMA20'] = df['Close'].rolling(window=period).mean()
    df['STD20'] = df['Close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA20'] + (2 * df['STD20'])
    df['Lower_Band'] = df['SMA20'] - (2 * df['STD20'])
    df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']
    return df

### Stochastic Oscillator (열 이름 변경: %K -> Percent_K, %D -> Percent_D)
def calculate_stochastic_oscillator(df, period=14):
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['Percent_K'] = (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']) * 100
    df['Percent_D'] = df['Percent_K'].rolling(window=3).mean()
    return df

### OBV (On-Balance Volume)
def calculate_obv(df):
    df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).astype(int)).cumsum()
    return df

### ATR (Average True Range)
def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

### SMA (Simple Moving Average)
def calculate_sma(df, periods=[5, 10, 50, 200]):
    for period in periods:
        df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
    return df

### Volume Change
def calculate_volume_change(df):
    df['Volume_Change'] = df['Volume'].pct_change()
    return df

### VWAP (Volume Weighted Average Price)
def calculate_vwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close'])….mean()
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

# 모든 지표를 한 번에 계산하는 함수
def calculate_all_indicators(df):
    df = calculate_rsi(df, period=14)  # 일일 RSI 14일
    df = calculate_rsi(df, period=5)   # 단기 RSI 5일
    df = calculate_macd(df, short_period=12, long_period=26, signal_period=9, prefix='')  # 기본 MACD
    df = calculate_macd(df, short_period=5, long_period=13, signal_period=1, prefix='Short_')  # 단기 MACD
    df = calculate_bollinger_bands(df)  # Bollinger Bands
    df = calculate_stochastic_oscillator(df)  # Stochastic Oscillator (Percent_K, Percent_D)
    df = calculate_obv(df)  # OBV
    df = calculate_atr(df)  # ATR
    df = calculate_sma(df, periods=[5, 10, 50, 200])  # SMA 5, 10, 50, 200일
    df = calculate_volume_change(df)  # Volume Change
    df = calculate_vwap(df)  # VWAP
    # 주간 RSI 추가
    weekly_rsi = calculate_weekly_rsi(df)
    df = df.join(weekly_rsi, on='Date')
    return df

# 지표를 데이터베이스에 저장하는 함수 (중복 방지)
def save_indicators_to_db(df, ticker, conn):
    indicators = ['RSI_14', 'RSI_5', 'MACD', 'Signal', 'MACD_Histogram', 'Short_MACD', 'Short_Signal', 'Short_MACD_Histogram',
                  'SMA20', 'Upper_Band', 'Lower_Band', 'BB_Width', 'Percent_K', 'Percent_D', 'OBV', 'ATR', 'SMA5', 'SMA10', 'SMA50', 'SMA200',
                  'Volume_Change', 'VWAP', 'Weekly_RSI']
    df_indicators = df[['Date'] + [col for col in indicators if col in df.columns]].copy()
    df_indicators['Ticker'] = ticker
    df_indicators['Date'] = df_indicators['Date'].dt.strftime('%Y-%m-%d')  # 날짜를 문자열로 변환
    # 기존 데이터의 날짜 확인
    existing_dates = pd.read_sql(f"SELECT Date FROM technical_indicators WHERE Ticker='{ticker}'", conn)['Date'].tolist()
    # 새로운 데이터만 필터링
    df_indicators = df_indicators[~df_indicators['Date'].isin(existing_dates)]
    if not df_indicators.empty:
        df_indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)
        print(f"Indicators for {ticker} saved to SQLite.")
    else:
        print(f"No new indicators to save for {ticker}.")

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description='과거 2년간의 데이터를 수집하여 SQLite에 저장하고 지표를 계산')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='주식 티커 (쉼표로 구분, 기본: TSLA,TSLL)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    # SQLite 데이터베이스 연결
    conn = sqlite3.connect(DB_FILE)

    # 테이블 생성 (필요한 경우)
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
                        PRIMARY KEY (Ticker, Date))''')
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

    # 데이터 수집 및 저장 실행
    print("Starting data collection...")
    get_fear_greed_data(conn)  # Fear & Greed Index 데이터 수집 및 저장
    for ticker in tickers:  # 주식 데이터 수집 및 저장
        get_stock_data(ticker, conn)
    get_vix_data(conn)  # VIX 지수 데이터 수집 및 저장

    # 데이터베이스 연결 닫기
    conn.close()
    print("All data collection and saving completed.")

    # SQLite에서 데이터 로드 및 지표 계산/저장
    conn = sqlite3.connect(DB_FILE)
    for ticker in tickers:
        df = pd.read_sql(f'SELECT * FROM stock_data WHERE Ticker="{ticker}"', conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = calculate_all_indicators(df)
        print(f"Technical indicators calculation completed for {ticker}")
        print(df.tail())  # 마지막 5개 행 출력
        save_indicators_to_db(df, ticker, conn)  # 지표 저장

    conn.close()

if __name__ == "__main__":
    main()
