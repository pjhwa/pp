import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import argparse
import sqlite3
import os
from tabulate import tabulate
import math
from termcolor import colored

DB_FILE = 'market_data.db'
TRADE_LOG_FILE = 'trade_log.csv'
INITIAL_INVESTMENT = 100000.0

# 데이터 수집 함수 정의
def get_fear_greed_data(conn):
    """
    Fear & Greed Index 데이터를 수집하여 데이터베이스에 저장합니다.
    """
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
        print(f"Fear & Greed Index data fetch error: {e}")

def get_stock_data(ticker, conn):
    """
    주식 데이터를 수집하여 데이터베이스에 저장합니다.
    """
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
        print(f"{ticker} data fetch error: {e}")

def get_vix_data(conn):
    """
    VIX 데이터를 수집하여 데이터베이스에 저장합니다.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM vix_data")
        last_date = cursor.fetchone()[0]
        if last_date:
            last_date = datetime.strptime(last_date, '%Y-%m-%d').date()
            start_date = last_date + timedelta(days=1)
        else:
            start_date = (datetime.now() - timedelta(days=730)).date()
        
        vix_df = yf.download("^VIX", start=start_date, end=datetime.now().date(), progress=False, auto_adjust=False)
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
        print(f"VIX data fetch error: {e}")

# 기술적 지표 계산 함수 정의
def calculate_rsi(df, period=14):
    """
    RSI(Relative Strength Index)를 계산합니다.
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def calculate_weekly_rsi(df, period=14):
    """
    주간 RSI를 계산합니다.
    """
    df_weekly = df.resample('W', on='Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df_weekly = calculate_rsi(df_weekly, period)
    df_weekly = df_weekly.rename(columns={f'RSI_{period}': 'Weekly_RSI'})
    return df_weekly['Weekly_RSI']

def calculate_macd(df, short_period=12, long_period=26, signal_period=9, prefix=''):
    """
    MACD(Moving Average Convergence Divergence)를 계산합니다.
    """
    df[f'EMA{short_period}'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df[f'EMA{long_period}'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df[f'{prefix}MACD'] = df[f'EMA{short_period}'] - df[f'EMA{long_period}']
    df[f'{prefix}Signal'] = df[f'{prefix}MACD'].ewm(span=signal_period, adjust=False).mean()
    df[f'{prefix}MACD_Histogram'] = df[f'{prefix}MACD'] - df[f'{prefix}Signal']
    return df

def calculate_bollinger_bands(df, period=20):
    """
    볼린저 밴드를 계산합니다.
    """
    df['SMA20'] = df['Close'].rolling(window=period).mean()
    df['STD20'] = df['Close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA20'] + (2 * df['STD20'])
    df['Lower_Band'] = df['SMA20'] - (2 * df['STD20'])
    df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']
    return df

def calculate_sma(df, period):
    """
    단순 이동 평균(SMA)을 계산합니다.
    """
    df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
    return df

def calculate_stochastic_oscillator(df, period=14):
    """
    스토캐스틱 오실레이터를 계산합니다.
    """
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['Percent_K'] = (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']) * 100
    df['Percent_D'] = df['Percent_K'].rolling(window=3).mean()
    return df

def calculate_obv(df):
    """
    OBV(On-Balance Volume)를 계산합니다.
    """
    df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).astype(int)).cumsum()
    return df

def calculate_atr(df, period=14):
    """
    ATR(Average True Range)를 계산합니다.
    """
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_vwap(df):
    """
    VWAP(Volume Weighted Average Price)를 계산합니다.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_all_indicators(df):
    """
    모든 기술적 지표를 계산하여 데이터프레임에 추가합니다.
    """
    df = calculate_rsi(df, 14)
    df = calculate_rsi(df, 5)
    df = calculate_macd(df, 12, 26, 9, '')
    df = calculate_macd(df, 5, 13, 1, 'Short_')
    df = calculate_bollinger_bands(df)
    df = calculate_sma(df, 5)
    df = calculate_sma(df, 10)
    df = calculate_sma(df, 50)
    df = calculate_sma(df, 200)
    df = calculate_stochastic_oscillator(df)
    df = calculate_obv(df)
    df = calculate_atr(df)
    df = calculate_vwap(df)
    weekly_rsi = calculate_weekly_rsi(df)
    df = df.join(weekly_rsi, on='Date')
    return df

# 데이터베이스 저장 및 포트폴리오 관리 함수
def save_indicators_to_db(df, ticker, conn):
    """
    계산된 기술적 지표를 데이터베이스에 저장합니다.
    """
    indicators = ['RSI_14', 'RSI_5', 'MACD', 'Signal', 'MACD_Histogram', 'Short_MACD', 'Short_Signal', 
                  'Short_MACD_Histogram', 'SMA20', 'Upper_Band', 'Lower_Band', 'BB_Width', 'SMA5', 
                  'SMA10', 'SMA50', 'SMA200', 'Weekly_RSI', 'Percent_K', 'Percent_D', 'OBV', 'ATR', 'VWAP']
    df_indicators = df[['Date'] + [col for col in indicators if col in df.columns]].copy()
    df_indicators['Ticker'] = ticker
    df_indicators['Date'] = df_indicators['Date'].dt.strftime('%Y-%m-%d')
    existing_dates = pd.read_sql(f"SELECT Date FROM technical_indicators WHERE Ticker='{ticker}'", conn)['Date'].tolist()
    df_indicators = df_indicators[~df_indicators['Date'].isin(existing_dates)]
    if not df_indicators.empty:
        df_indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)

def load_trade_log_to_db(conn):
    """
    거래 로그를 데이터베이스에 로드합니다.
    """
    if not os.path.exists(TRADE_LOG_FILE):
        return
    df = pd.read_csv(TRADE_LOG_FILE)
    df['Shares'] = df['Shares'].astype(int)
    df.to_sql('trade_log', conn, if_exists='replace', index=False)

def process_portfolio(conn):
    """
    포트폴리오 상태를 처리하고 업데이트합니다.
    """
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

def get_current_portfolio(conn, tickers):
    """
    현재 포트폴리오 상태를 가져옵니다.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(Date) FROM portfolio_state")
    latest_date = cursor.fetchone()[0]
    if not latest_date:
        return None, None, None, None
    
    portfolio_data = pd.read_sql(f"SELECT * FROM portfolio_state WHERE Date='{latest_date}'", conn)
    if portfolio_data.empty:
        return None, None, None, None
    
    cash = portfolio_data['Cash'].iloc[0]
    holdings = {row['Ticker']: row['Shares'] for _, row in portfolio_data.iterrows()}
    
    prices = {}
    for ticker in tickers:
        cursor.execute(f"SELECT Close FROM stock_data WHERE Ticker='{ticker}' ORDER BY Date DESC LIMIT 1")
        result = cursor.fetchone()
        prices[ticker] = result[0] if result else 0.0
    
    total_value = cash
    for ticker in tickers:
        if ticker in holdings:
            total_value += holdings[ticker] * prices.get(ticker, 0.0)
    
    return cash, holdings, prices, total_value

def get_initial_investment(conn):
    """
    초기 투자 금액을 가져옵니다.
    """
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

# 포트폴리오 시각화 및 시장 지표 표시 함수
def visualize_portfolio_text(conn, tickers, initial_investment):
    """
    포트폴리오 상태를 텍스트로 시각화하여 출력합니다.
    """
    cash, holdings, prices, total_value = get_current_portfolio(conn, tickers)
    if total_value is None:
        print("No portfolio data available")
        return
    
    portfolio_values = []
    for ticker in tickers:
        shares = holdings.get(ticker, 0)
        price = prices.get(ticker, 0.0)
        value = shares * price
        weight = (value / total_value * 100) if total_value > 0 else 0
        portfolio_values.append({
            'Asset': ticker,
            'Shares': shares,
            'Price': f"${price:.2f}" if price > 0 else '$0',
            'Value': f"${value:.2f}" if value > 0 else '$0',
            'Weight (%)': f"{weight:.2f}%"
        })
    
    portfolio_values.append({
        'Asset': 'Cash',
        'Shares': '-',
        'Price': '-',
        'Value': f"${cash:.2f}",
        'Weight (%)': f"{(cash / total_value * 100):.2f}%" if total_value > 0 else '0%'
    })
    
    return_rate = ((total_value - initial_investment) / initial_investment) * 100 if initial_investment > 0 else 0
    return_color = 'green' if return_rate >= 0 else 'red'
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*50}")
    print(f"📊 **Portfolio Summary ({current_date})**")
    print(f"💰 **Initial Investment:** ${initial_investment:.2f}")
    print(f"💰 **Current Value:** ${total_value:.2f}")
    print(f"📈 **Return Rate:** {colored(f'{return_rate:.2f}%', return_color)}")
    print(f"{'='*50}")
    df_portfolio = pd.DataFrame(portfolio_values)
    print(tabulate(df_portfolio, headers=['Asset', 'Shares', 'Price', 'Value', 'Weight (%)'], tablefmt='fancy_grid', showindex=False))

def display_market_indicators(df_dict, conn):
    """
    시장 지표를 카테고리별로 정리하여 출력합니다.
    """
    if 'TSLA' not in df_dict or df_dict['TSLA'].empty:
        print("No data available for TSLA")
        return
    
    df_tsla = df_dict['TSLA']
    latest_tsla = df_tsla.iloc[-1]
    
    cursor = conn.cursor()
    cursor.execute("SELECT fear_greed_index FROM fear_greed_index ORDER BY date DESC LIMIT 1")
    fear_greed = cursor.fetchone()
    fear_greed = fear_greed[0] if fear_greed else 'N/A'
    
    indicators = {
        'Momentum': [
            {'Indicator': 'Fear & Greed Index', 'Value': f"{fear_greed:.2f}" if isinstance(fear_greed, float) else fear_greed, 'Trend/Notes': 'Fear' if isinstance(fear_greed, float) and fear_greed < 30 else 'Greed' if isinstance(fear_greed, float) and fear_greed > 70 else 'Neutral'},
            {'Indicator': 'MACD Histogram', 'Value': f"{latest_tsla['MACD_Histogram']:.2f}", 'Trend/Notes': 'Bullish' if latest_tsla['MACD_Histogram'] > 0 else 'Bearish'},
            {'Indicator': 'Daily RSI', 'Value': f"{latest_tsla['RSI_14']:.2f}", 'Trend/Notes': 'Increasing' if len(df_tsla) > 1 and latest_tsla['RSI_14'] > df_tsla['RSI_14'].iloc[-2] else 'Decreasing'},
        ],
        'Volatility': [
            {'Indicator': 'ATR', 'Value': f"${latest_tsla['ATR']:.2f}", 'Trend/Notes': 'High Volatility' if latest_tsla['ATR'] > 10 else 'Low Volatility'},
            {'Indicator': 'BB Width', 'Value': f"{latest_tsla['BB_Width']:.2f}", 'Trend/Notes': 'High' if latest_tsla['BB_Width'] > 0.3 else 'Low'},
        ],
        'Trend': [
            {'Indicator': 'TSLA Close', 'Value': f"${latest_tsla['Close']:.2f}", 'Trend/Notes': f"{'Above' if latest_tsla['Close'] > latest_tsla['SMA50'] else 'Below'} SMA50"},
            {'Indicator': 'SMA50', 'Value': f"${latest_tsla['SMA50']:.2f}", 'Trend/Notes': 'Above' if latest_tsla['Close'] > latest_tsla['SMA50'] else 'Below'},
            {'Indicator': 'SMA200', 'Value': f"${latest_tsla['SMA200']:.2f}", 'Trend/Notes': 'Above' if latest_tsla['Close'] > latest_tsla['SMA200'] else 'Below'},
        ]
    }
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*50}")
    print(f"📈 **Market Indicators Summary ({current_date})**")
    print(f"{'='*50}")
    for category, ind_list in indicators.items():
        print(f"\n**{category}**")
        df_indicators = pd.DataFrame(ind_list)
        print(tabulate(df_indicators, headers=['Indicator', 'Value', 'Trend/Notes'], tablefmt='fancy_grid', showindex=False))

def check_weight_adjustment(df_dict, tickers, total_value, prices, conn, holdings):
    """
    포트폴리오 비중 조정 제안을 계산하고 출력합니다.
    """
    if 'TSLA' not in df_dict or df_dict['TSLA'].empty:
        print("No data available for TSLA")
        return
    
    df_tsla = df_dict['TSLA']
    latest_tsla = df_tsla.iloc[-1]
    
    cursor = conn.cursor()
    cursor.execute("SELECT fear_greed_index FROM fear_greed_index ORDER BY date DESC LIMIT 1")
    fear_greed_row = cursor.fetchone()
    fear_greed = fear_greed_row[0] if fear_greed_row else 50.0

    # 현재 비중 계산
    tsla_shares = holdings.get('TSLA', 0)
    tsll_shares = holdings.get('TSLL', 0)
    tsla_value = tsla_shares * prices.get('TSLA', 0.0)
    tsll_value = tsll_shares * prices.get('TSLL', 0.0)
    current_tsla_weight = (tsla_value / total_value) if total_value > 0 else 0.0
    current_tsll_weight = (tsll_value / total_value) if total_value > 0 else 0.0

    # 매수/매도 신호 정의
    buy_signals = []
    sell_signals = []
    if pd.notna(latest_tsla['Weekly_RSI']) and latest_tsla['Weekly_RSI'] < 34.78:
        buy_signals.append("Weekly RSI < 34.78 (Oversold)")
    if latest_tsla['MACD'] > latest_tsla['Signal'] and latest_tsla['Signal'] < 0:
        buy_signals.append("MACD > Signal (Signal < 0)")
    if latest_tsla['MACD_Histogram'] > 0:
        buy_signals.append("MACD Histogram > 0 (Bullish Momentum)")
    if len(df_tsla) > 1 and latest_tsla['OBV'] > df_tsla['OBV'].iloc[-2]:
        buy_signals.append("OBV Increasing (Buying Pressure)")
    if latest_tsla['SMA5'] > latest_tsla['SMA10']:
        buy_signals.append("SMA5 > SMA10 (Short-Term Bullish)")
    if latest_tsla['Close'] > latest_tsla['VWAP']:
        buy_signals.append("Close > VWAP (Above Average Price)")

    if latest_tsla['Percent_K'] > 74.37:
        sell_signals.append("Stochastic %K > 74.37 (Overbought)")
    if latest_tsla['BB_Width'] > 0.3:
        sell_signals.append("BB Width > 0.3 (High Volatility)")
    if latest_tsla['RSI_5'] > 70.26:
        sell_signals.append("Short RSI > 70.26 (Short-Term Overbought)")

    # 신호 점수 계산
    net_score = len(buy_signals) - len(sell_signals)
    max_score = max(len(buy_signals), len(sell_signals), 1)
    normalized_score = net_score / max_score

    # 위험 조정
    volatility_factor = min(latest_tsla['ATR'] / 10.0, 1.0)
    sentiment_factor = 1.0 if fear_greed > 50 else 0.5 if fear_greed < 30 else 0.75
    base_adjustment = 0.1 * abs(normalized_score) * sentiment_factor * (1 - volatility_factor)
    adjustment = max(min(base_adjustment, 0.15), 0.05)

    # 제안 계산
    reasoning = []
    if normalized_score > 0:
        tsla_change = adjustment if current_tsla_weight < 0.9 else 0.0
        tsll_change = -adjustment if current_tsll_weight > 0.1 else 0.0
        reasoning.append(f"Bullish signal (Score: {normalized_score:.2f}): Increase TSLA weight / Reduce TSLL weight due to high volatility")
    elif normalized_score < 0:
        tsla_change = -adjustment if current_tsla_weight > 0.1 else 0.0
        tsll_change = -adjustment if current_tsll_weight > 0.1 else 0.0
        reasoning.append(f"Bearish signal (Score: {normalized_score:.2f}): Reduce leveraged asset weight")
    else:
        tsla_change = 0.0
        tsll_change = 0.0
        reasoning.append("Neutral signal: No significant adjustment needed")

    # 목표 비중 계산
    target_tsla_weight = max(min(current_tsla_weight + tsla_change, 1.0), 0.0)
    target_tsll_weight = max(min(current_tsll_weight + tsll_change, 1.0 - target_tsla_weight), 0.0)
    p_tsla = prices.get('TSLA', 0.0)
    p_tsll = prices.get('TSLL', 0.0)
    target_tsla_shares = math.floor((total_value * target_tsla_weight) / p_tsla) if p_tsla > 0 else 0
    target_tsll_shares = math.floor((total_value * target_tsll_weight) / p_tsll) if p_tsll > 0 else 0

    # 출력
    print(f"\n{'='*50}")
    print("📈 **Portfolio Adjustment Suggestions**")
    print(f"{'='*50}")
    print(f"**TSLA Suggestion:** {'Increase' if tsla_change > 0 else 'Decrease' if tsla_change < 0 else 'Hold'} ({abs(tsla_change)*100:.1f}%)")
    print(f"**TSLL Suggestion:** {'Increase' if tsll_change > 0 else 'Decrease' if tsll_change < 0 else 'Hold'} ({abs(tsll_change)*100:.1f}%)")
    
    proposed = [
        {'Asset': 'TSLA', 'Current Weight': f"{current_tsla_weight*100:.1f}%", 'Target Weight': f"{target_tsla_weight*100:.1f}%", 'Target Shares': target_tsla_shares},
        {'Asset': 'TSLL', 'Current Weight': f"{current_tsll_weight*100:.1f}%", 'Target Weight': f"{target_tsll_weight*100:.1f}%", 'Target Shares': target_tsll_shares}
    ]
    print("\n**Proposed Portfolio Changes**")
    df_proposed = pd.DataFrame(proposed)
    print(tabulate(df_proposed, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    print("\n**Adjustment Rationale**")
    print("- Buy Signals:")
    if buy_signals:
        for signal in buy_signals:
            print(f"  . {signal}")
    else:
        print("  None")
    print("- Sell Signals:")
    if sell_signals:
        for signal in sell_signals:
            print(f"  . {signal}")
    else:
        print("  None")
    print(f"- Signal Score: {normalized_score:.2f} (Positive: Bullish, Negative: Bearish)")
    print(f"- Volatility (ATR): ${latest_tsla['ATR']:.2f}")
    print(f"- Fear & Greed Index: {fear_greed:.2f} ({'Fear' if fear_greed < 30 else 'Greed' if fear_greed > 70 else 'Neutral'})")
    print("- Conclusion:")
    if reasoning:
        for reason in reasoning:
            print(f"  . {reason}")
    else:
        print("  No significant adjustment needed")

# 메인 함수
def main():
    """
    프로그램의 메인 실행 함수입니다. 데이터 수집, 저장, 포트폴리오 관리 및 시각화를 수행합니다.
    """
    parser = argparse.ArgumentParser(description='Collect 2 years of data, store in SQLite, and manage portfolio')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='Stock tickers (comma-separated, default: TSLA,TSLL)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    conn = sqlite3.connect(DB_FILE)

    # 데이터베이스 테이블 생성
    conn.execute('''CREATE TABLE IF NOT EXISTS fear_greed_index (date TEXT PRIMARY KEY, fear_greed_index REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_data (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, PRIMARY KEY (Ticker, Date))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS vix_data (Date TEXT PRIMARY KEY, VIX_Close REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS technical_indicators (Ticker TEXT, Date TEXT, RSI_14 REAL, RSI_5 REAL, MACD REAL, Signal REAL, MACD_Histogram REAL, Short_MACD REAL, Short_Signal REAL, Short_MACD_Histogram REAL, SMA20 REAL, Upper_Band REAL, Lower_Band REAL, BB_Width REAL, SMA5 REAL, SMA10 REAL, SMA50 REAL, SMA200 REAL, Weekly_RSI REAL, Percent_K REAL, Percent_D REAL, OBV REAL, ATR REAL, VWAP REAL, PRIMARY KEY (Ticker, Date))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Ticker TEXT, Action TEXT, Shares INTEGER, Price REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS portfolio_state (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Cash REAL, Ticker TEXT, Shares INTEGER)''')

    # 데이터 수집 및 저장
    get_fear_greed_data(conn)
    for ticker in tickers:
        get_stock_data(ticker, conn)
    get_vix_data(conn)

    # 거래 로그 및 포트폴리오 처리
    load_trade_log_to_db(conn)
    process_portfolio(conn)

    # 지표 계산
    df_dict = {}
    for ticker in tickers:
        df = pd.read_sql(f'SELECT * FROM stock_data WHERE Ticker="{ticker}"', conn)
        if df.empty:
            print(f"No stock data available for {ticker}")
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = calculate_all_indicators(df)
        df_dict[ticker] = df
        save_indicators_to_db(df, ticker, conn)

    # 초기 투자 금액 계산
    initial_investment = get_initial_investment(conn)

    # 현재 포트폴리오 상태 가져오기
    cash, holdings, prices, total_value = get_current_portfolio(conn, tickers)
    if total_value is None:
        print("No portfolio data available")
        conn.close()
        return

    # 포트폴리오 시각화
    visualize_portfolio_text(conn, tickers, initial_investment)

    # 시장 지표 표시
    display_market_indicators(df_dict, conn)

    # 가중치 조정 제안
    check_weight_adjustment(df_dict, tickers, total_value, prices, conn, holdings)

    conn.close()

if __name__ == "__main__":
    main()
