import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import argparse
import sqlite3
import os
from tabulate import tabulate
import math
import numpy as np
from termcolor import colored
from deap import base, creator, tools, algorithms
from tqdm import tqdm

DB_FILE = 'market_data.db'
TRADE_LOG_FILE = 'trade_log.csv'
INITIAL_INVESTMENT = 100000.0
TRANSACTION_COST_RATE = 0.001  # 0.1% transaction cost

# DEAP 클래스 단일 생성으로 RuntimeWarning 방지
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# 데이터 수집 함수
def get_fear_greed_data(conn):
    """Fear & Greed Index 데이터를 가져와 저장합니다."""
    today = datetime.now()
    start_date = today - timedelta(days=730)
    start_date_str = start_date.strftime('%Y-%m-%d')
    url = f'https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date_str}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        df = pd.json_normalize(data['fear_and_greed_historical']['data'])
        df['date'] = pd.to_datetime(df['x'], unit='ms').dt.date
        df = df[['date', 'y']].rename(columns={'y': 'fear_greed_index'})
        df.to_sql('fear_greed_index', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Fear & Greed Index data fetch error: {e}")

def get_stock_data(ticker, conn):
    """주식 데이터를 점진적으로 가져와 저장합니다."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(Date) FROM stock_data WHERE Ticker='{ticker}'")
        last_date = cursor.fetchone()[0]
        start_date = (datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)) if last_date else (datetime.now() - timedelta(days=730)).date()
        
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
    """VIX 데이터를 가져와 저장하며 오류를 처리합니다."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM vix_data")
        last_date = cursor.fetchone()[0]
        start_date = (datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)) if last_date else (datetime.now() - timedelta(days=730)).date()
        
        vix_df = yf.download("^VIX", start=start_date, end=datetime.now().date(), progress=False, auto_adjust=False)
        if vix_df.empty:
            print(f"No VIX data found for {start_date} to {datetime.now().date()}")
            return None
        vix_df = vix_df[['Close']].reset_index()
        vix_df['Date'] = vix_df['Date'].dt.strftime('%Y-%m-%d')
        vix_df = vix_df.rename(columns={'Close': 'VIX_Close'})
        
        existing_dates = pd.read_sql("SELECT Date FROM vix_data", conn)['Date'].tolist()
        vix_df = vix_df[~vix_df['Date'].isin(existing_dates)]
        if not vix_df.empty:
            cursor.executemany("INSERT INTO vix_data (Date, VIX_Close) VALUES (?, ?)", vix_df[['Date', 'VIX_Close']].values.tolist())
            conn.commit()
        return vix_df
    except Exception as e:
        print(f"VIX data fetch error: {e}")
        return None

# 기술적 지표 계산 함수
def calculate_rsi(df, period=14):
    """RSI를 계산합니다."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def calculate_weekly_rsi(df, period=14):
    """주간 RSI를 계산합니다."""
    df_weekly = df.resample('W', on='Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df_weekly = calculate_rsi(df_weekly, period)
    return df_weekly.rename(columns={f'RSI_{period}': 'Weekly_RSI'})['Weekly_RSI']

def calculate_macd(df, short_period=12, long_period=26, signal_period=9, prefix=''):
    """MACD를 계산합니다."""
    df[f'EMA{short_period}'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df[f'EMA{long_period}'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df[f'{prefix}MACD'] = df[f'EMA{short_period}'] - df[f'EMA{long_period}']
    df[f'{prefix}Signal'] = df[f'{prefix}MACD'].ewm(span=signal_period, adjust=False).mean()
    df[f'{prefix}MACD_Histogram'] = df[f'{prefix}MACD'] - df[f'{prefix}Signal']
    return df

def calculate_bollinger_bands(df, period=20):
    """볼린저 밴드를 계산합니다."""
    df['SMA20'] = df['Close'].rolling(window=period).mean()
    df['STD20'] = df['Close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA20'] + (2 * df['STD20'])
    df['Lower_Band'] = df['SMA20'] - (2 * df['STD20'])
    df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']
    return df

def calculate_sma(df, period):
    """SMA를 계산합니다."""
    df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
    return df

def calculate_stochastic_oscillator(df, period=14):
    """스토캐스틱 오실레이터를 계산합니다."""
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['Percent_K'] = (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']) * 100
    df['Percent_D'] = df['Percent_K'].rolling(window=3).mean()
    return df

def calculate_obv(df):
    """OBV를 계산합니다."""
    df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).astype(int)).cumsum()
    return df

def calculate_atr(df, period=14):
    """ATR를 계산합니다."""
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_vwap(df):
    """VWAP를 계산합니다."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_all_indicators(df):
    """모든 기술적 지표를 계산합니다."""
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

# 백테스트 및 시뮬레이션 함수
def load_backtest_data(ticker, conn, start_date, end_date):
    """백테스트용 데이터를 지정된 기간 동안 로드합니다."""
    df = pd.read_sql(f"SELECT * FROM stock_data WHERE Ticker='{ticker}' AND Date BETWEEN '{start_date}' AND '{end_date}'", conn)
    if df.empty or len(df) < 50:  # 최소 50일 데이터 필요
        print(f"Insufficient data for {ticker} from {start_date} to {end_date}.")
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return calculate_all_indicators(df)

def generate_signals(df, strategy='rsi', **kwargs):
    """전략에 따라 매수/매도 신호를 생성합니다."""
    df['Signal'] = 0  # 0: 보유
    if strategy == 'rsi':
        buy_threshold = kwargs.get('buy_threshold', 30)
        sell_threshold = kwargs.get('sell_threshold', 70)
        df.loc[df['RSI_14'] < buy_threshold, 'Signal'] = 1  # 매수
        df.loc[df['RSI_14'] > sell_threshold, 'Signal'] = -1  # 매도
    elif strategy == 'macd':
        buy_threshold = kwargs.get('buy_threshold', 0)
        sell_threshold = kwargs.get('sell_threshold', 0)
        df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'Signal'] = 1  # 매수
        df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'Signal'] = -1  # 매도
    elif strategy == 'bollinger':
        buy_threshold = kwargs.get('buy_threshold', -2)
        sell_threshold = kwargs.get('sell_threshold', 2)
        df['Z_Score'] = (df['Close'] - df['SMA20']) / df['STD20']
        df.loc[df['Z_Score'] < buy_threshold, 'Signal'] = 1  # 매수
        df.loc[df['Z_Score'] > sell_threshold, 'Signal'] = -1  # 매도
    elif strategy == 'stochastic':
        buy_threshold = kwargs.get('buy_threshold', 20)
        sell_threshold = kwargs.get('sell_threshold', 80)
        df.loc[df['Percent_K'] < buy_threshold, 'Signal'] = 1  # 매수
        df.loc[df['Percent_K'] > sell_threshold, 'Signal'] = -1  # 매도
    return df

def backtest_strategy(df, ticker, initial_cash=INITIAL_INVESTMENT, strategy='rsi', **kwargs):
    """주어진 전략으로 백테스트를 수행합니다."""
    df = generate_signals(df, strategy, **kwargs)
    cash = initial_cash
    shares = 0
    portfolio_value = []
    trade_history = []

    for _, row in df.iterrows():
        price = row['Close']
        signal = row['Signal']
        trade_amount = price * 100  # 100주 거래
        cost = trade_amount * TRANSACTION_COST_RATE
        
        if signal == 1 and cash >= trade_amount + cost:  # 매수
            shares += 100
            cash -= trade_amount - cost
            trade_history.append({'Date': row['Date'], 'Ticker': ticker, 'Action': 'Buy', 'Shares': 100, 'Price': price, 'Cost': cost})
        elif signal == -1 and shares >= 100:  # 매도
            shares -= 100
            cash += trade_amount - cost
            trade_history.append({'Date': row['Date'], 'Ticker': ticker, 'Action': 'Sell', 'Shares': 100, 'Price': price, 'Cost': cost})
        
        total_value = cash + (shares * price)
        portfolio_value.append({'Date': row['Date'], 'Portfolio Value': total_value})

    return pd.DataFrame(portfolio_value), pd.DataFrame(trade_history)

def calculate_performance_metrics(portfolio_df, initial_investment):
    """성과 지표를 계산합니다."""
    total_return = (portfolio_df['Portfolio Value'].iloc[-1] - initial_investment) / initial_investment * 100
    returns = portfolio_df['Portfolio Value'].pct_change().dropna()
    annualized_return = ((1 + total_return / 100) ** (252 / len(portfolio_df)) - 1) * 100
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = ((portfolio_df['Portfolio Value'].cummax() - portfolio_df['Portfolio Value']) / portfolio_df['Portfolio Value'].cummax()).max() * 100
    return {
        'Total Return (%)': total_return,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown
    }

def genetic_algorithm_optimize(df, ticker, strategy='rsi', generations=20, pop_size=50, conn=None):
    """유전 알고리즘을 사용해 각 지표의 임계값을 최적화하며 진행률을 표시합니다."""
    toolbox = base.Toolbox()
    
    if strategy == 'rsi':
        toolbox.register("attr_buy", np.random.randint, 20, 40)
        toolbox.register("attr_sell", np.random.randint, 60, 80)
    elif strategy == 'macd':
        toolbox.register("attr_buy", np.random.uniform, -1, 1)
        toolbox.register("attr_sell", np.random.uniform, -1, 1)
    elif strategy == 'bollinger':
        toolbox.register("attr_buy", np.random.uniform, -3, -1)
        toolbox.register("attr_sell", np.random.uniform, 1, 3)
    elif strategy == 'stochastic':
        toolbox.register("attr_buy", np.random.randint, 10, 30)
        toolbox.register("attr_sell", np.random.randint, 70, 90)
    
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_buy, toolbox.attr_sell), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        portfolio_df, _ = backtest_strategy(df, ticker, strategy=strategy, buy_threshold=individual[0], sell_threshold=individual[1])
        metrics = calculate_performance_metrics(portfolio_df, INITIAL_INVESTMENT)
        return metrics['Sharpe Ratio'],
    
    if strategy in ['rsi', 'stochastic']:
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=[20, 60] if strategy == 'rsi' else [10, 70], 
                         up=[40, 80] if strategy == 'rsi' else [30, 90], indpb=0.2)
    else:
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    for gen in tqdm(range(1, generations + 1), desc=f"Optimizing {strategy.upper()} for {ticker}"):
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, pop)
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit
        pop = toolbox.select(pop, k=len(pop))
        hof.update(pop)
    
    best_params = hof[0]
    print(f"Best optimized parameters for {ticker} ({strategy.upper()}): Buy Threshold = {best_params[0]:.2f}, Sell Threshold = {best_params[1]:.2f}")
    
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimized_thresholds (
                Ticker TEXT, Strategy TEXT, Buy_Threshold REAL, Sell_Threshold REAL,
                PRIMARY KEY (Ticker, Strategy)
            )
        """)
        cursor.execute(
            "INSERT OR REPLACE INTO optimized_thresholds (Ticker, Strategy, Buy_Threshold, Sell_Threshold) VALUES (?, ?, ?, ?)",
            (ticker, strategy, best_params[0], best_params[1])
        )
        conn.commit()
    
    return {'buy_threshold': best_params[0], 'sell_threshold': best_params[1]}

def load_optimized_thresholds(conn, ticker, strategy='rsi'):
    """데이터베이스에서 특정 지표의 최적화된 임계값을 로드합니다."""
    cursor = conn.cursor()
    cursor.execute("SELECT Buy_Threshold, Sell_Threshold FROM optimized_thresholds WHERE Ticker=? AND Strategy=?", (ticker, strategy))
    result = cursor.fetchone()
    if result:
        return {'buy_threshold': result[0], 'sell_threshold': result[1]}
    defaults = {
        'rsi': {'buy_threshold': 30, 'sell_threshold': 70},
        'macd': {'buy_threshold': 0, 'sell_threshold': 0},
        'bollinger': {'buy_threshold': -2, 'sell_threshold': 2},
        'stochastic': {'buy_threshold': 20, 'sell_threshold': 80}
    }
    return defaults.get(strategy, {'buy_threshold': 30, 'sell_threshold': 70})

def display_backtest_results(portfolio_df, trade_df, metrics, strategy, ticker):
    """백테스트 결과를 표시합니다."""
    print(f"\n{'='*50}\n📊 **Backtest Results for {strategy.upper()} Strategy ({ticker})**\n{'='*50}")
    print("\n**Portfolio Value Over Time**")
    print(tabulate(portfolio_df.tail(), headers='keys', tablefmt='fancy_grid', showindex=False))
    print("\n**Trade History**")
    print(tabulate(trade_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("\n**Performance Metrics**")
    print(tabulate(pd.DataFrame([metrics]), headers='keys', tablefmt='fancy_grid', showindex=False))
    print(f"\nExpected Annualized Return: {metrics['Annualized Return (%)']:.2f}%")

# 데이터베이스 및 포트폴리오 관리 함수
def save_indicators_to_db(df, ticker, conn):
    """기술적 지표를 데이터베이스에 저장합니다."""
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
    """거래 로그를 데이터베이스에 로드합니다."""
    if not os.path.exists(TRADE_LOG_FILE):
        return
    df = pd.read_csv(TRADE_LOG_FILE)
    df['Shares'] = df['Shares'].astype(int)
    df.to_sql('trade_log', conn, if_exists='replace', index=False)

def process_portfolio(conn):
    """포트폴리오 상태를 처리하고 업데이트합니다."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio_state")
    trade_log = pd.read_sql("SELECT * FROM trade_log ORDER BY Date", conn)
    if trade_log.empty:
        return
    
    cash = 0.0
    holdings = {}
    
    for _, trade in trade_log.iterrows():
        date, ticker, action, shares, price = trade['Date'], trade['Ticker'], trade['Action'].lower(), int(trade['Shares']), trade['Price']
        trade_amount = shares * price
        cost = trade_amount * TRANSACTION_COST_RATE
        
        holdings.setdefault(ticker, 0)
        if action == 'buy':
            cash -= trade_amount + cost
            holdings[ticker] += shares
        elif action == 'sell' and holdings[ticker] >= shares:
            cash += trade_amount - cost
            holdings[ticker] -= shares
        elif action == 'hold':
            holdings[ticker] = shares
        
        cursor.execute("INSERT INTO portfolio_state (Date, Cash, Ticker, Shares) VALUES (?, ?, ?, ?)", (date, cash, ticker, holdings[ticker]))
    
    conn.commit()

def get_current_portfolio(conn, tickers):
    """현재 포트폴리오 상태를 조회합니다."""
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
    prices = {ticker: cursor.execute(f"SELECT Close FROM stock_data WHERE Ticker='{ticker}' ORDER BY Date DESC LIMIT 1").fetchone()[0] or 0.0 for ticker in tickers}
    
    total_value = cash + sum(holdings.get(ticker, 0) * prices.get(ticker, 0.0) for ticker in tickers)
    return cash, holdings, prices, total_value

def get_initial_investment(conn):
    """거래 로그에서 초기 투자 금액을 계산합니다."""
    if not os.path.exists(TRADE_LOG_FILE):
        return INITIAL_INVESTMENT
    trade_log = pd.read_csv(TRADE_LOG_FILE)
    hold_trades = trade_log[trade_log['Action'].str.lower() == 'hold']
    return sum(trade['Shares'] * trade['Price'] for _, trade in hold_trades.iterrows()) if not hold_trades.empty else INITIAL_INVESTMENT

# 시각화 및 지표 표시 함수
def visualize_portfolio_text(conn, tickers, initial_investment):
    """포트폴리오 상태를 텍스트로 시각화합니다."""
    cash, holdings, prices, total_value = get_current_portfolio(conn, tickers)
    if total_value is None:
        print("No portfolio data available.")
        return
    
    portfolio_values = [
        {'Asset': ticker, 'Shares': holdings.get(ticker, 0), 'Price': f"${prices.get(ticker, 0.0):.2f}" if prices.get(ticker) else '$0',
         'Value': f"${holdings.get(ticker, 0) * prices.get(ticker, 0.0):.2f}", 'Weight (%)': f"{(holdings.get(ticker, 0) * prices.get(ticker, 0.0) / total_value * 100):.2f}%"} 
        for ticker in tickers
    ]
    portfolio_values.append({'Asset': 'Cash', 'Shares': '-', 'Price': '-', 'Value': f"${cash:.2f}", 'Weight (%)': f"{(cash / total_value * 100):.2f}%"})
    
    return_rate = ((total_value - initial_investment) / initial_investment) * 100 if initial_investment > 0 else 0
    return_color = 'green' if return_rate >= 0 else 'red'
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*50}\n📊 **Portfolio Summary ({current_date})**\n💰 **Initial Investment:** ${initial_investment:.2f}\n💰 **Current Value:** ${total_value:.2f}\n📈 **Return Rate:** {colored(f'{return_rate:.2f}%', return_color)}\n{'='*50}")
    print(tabulate(pd.DataFrame(portfolio_values), headers=['Asset', 'Shares', 'Price', 'Value', 'Weight (%)'], tablefmt='fancy_grid', showindex=False))

def display_market_indicators(df_dict, conn):
    """시장 지표를 카테고리별로 표시합니다."""
    if 'TSLA' not in df_dict or df_dict['TSLA'].empty:
        print("No data available for TSLA.")
        return
    
    latest_tsla = df_dict['TSLA'].iloc[-1]
    fear_greed = conn.cursor().execute("SELECT fear_greed_index FROM fear_greed_index ORDER BY date DESC LIMIT 1").fetchone()
    fear_greed = fear_greed[0] if fear_greed else 'N/A'
    
    indicators = {
        'Momentum': [
            {'Indicator': 'Fear & Greed Index', 'Value': f"{fear_greed:.2f}" if isinstance(fear_greed, float) else fear_greed, 'Trend/Notes': 'Fear' if isinstance(fear_greed, float) and fear_greed < 30 else 'Greed' if isinstance(fear_greed, float) and fear_greed > 70 else 'Neutral'},
            {'Indicator': 'MACD Histogram', 'Value': f"{latest_tsla['MACD_Histogram']:.2f}", 'Trend/Notes': 'Bullish' if latest_tsla['MACD_Histogram'] > 0 else 'Bearish'},
            {'Indicator': 'Daily RSI', 'Value': f"{latest_tsla['RSI_14']:.2f}", 'Trend/Notes': 'Increasing' if len(df_dict['TSLA']) > 1 and latest_tsla['RSI_14'] > df_dict['TSLA']['RSI_14'].iloc[-2] else 'Decreasing'},
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
    print(f"\n{'='*50}\n📈 **Market Indicators Summary ({current_date})**\n{'='*50}")
    for category, ind_list in indicators.items():
        print(f"\n**{category}**")
        print(tabulate(pd.DataFrame(ind_list), headers=['Indicator', 'Value', 'Trend/Notes'], tablefmt='fancy_grid', showindex=False))

def check_weight_adjustment(df_dict, tickers, total_value, prices, conn, holdings):
    """최적화된 임계값을 사용해 포트폴리오 비중 조정을 제안합니다."""
    if 'TSLA' not in df_dict or df_dict['TSLA'].empty:
        print("No data available for TSLA.")
        return
    
    # 데이터가 충분한지 확인 후 현재와 이전 데이터 가져오기
    tsla_df = df_dict['TSLA']
    if len(tsla_df) < 2:
        print("Insufficient data for TSLA to generate signals.")
        return
    current = tsla_df.iloc[-1]  # 현재 데이터
    previous = tsla_df.iloc[-2]  # 이전 데이터
    fear_greed = conn.cursor().execute("SELECT fear_greed_index FROM fear_greed_index ORDER BY date DESC LIMIT 1").fetchone()[0] or 50.0

    tsla_value = holdings.get('TSLA', 0) * prices.get('TSLA', 0.0)
    tsll_value = holdings.get('TSLL', 0) * prices.get('TSLL', 0.0)
    current_tsla_weight = tsla_value / total_value if total_value > 0 else 0.0
    current_tsll_weight = tsll_value / total_value if total_value > 0 else 0.0

    # 각 지표에 대해 최적화된 임계값 로드
    strategies = ['rsi', 'macd', 'bollinger', 'stochastic']
    thresholds = {strategy: load_optimized_thresholds(conn, 'TSLA', strategy) for strategy in strategies}

    # 매수 및 매도 신호 생성
    buy_signals = [
        f"RSI < {thresholds['rsi']['buy_threshold']:.2f} (Oversold)" if current['RSI_14'] < thresholds['rsi']['buy_threshold'] else None,
        "MACD Cross Above Signal" if current['MACD'] > current['Signal'] and previous['MACD'] <= previous['Signal'] else None,
        f"BB Z-Score < {thresholds['bollinger']['buy_threshold']:.2f}" if (current['Close'] - current['SMA20']) / current['STD20'] < thresholds['bollinger']['buy_threshold'] else None,
        f"Stochastic %K < {thresholds['stochastic']['buy_threshold']:.2f}" if current['Percent_K'] < thresholds['stochastic']['buy_threshold'] else None,
        "OBV Increasing" if current['OBV'] > previous['OBV'] else None,
        "SMA5 > SMA10" if current['SMA5'] > current['SMA10'] else None
    ]
    sell_signals = [
        f"RSI > {thresholds['rsi']['sell_threshold']:.2f} (Overbought)" if current['RSI_14'] > thresholds['rsi']['sell_threshold'] else None,
        "MACD Cross Below Signal" if current['MACD'] < current['Signal'] and previous['MACD'] >= previous['Signal'] else None,
        f"BB Z-Score > {thresholds['bollinger']['sell_threshold']:.2f}" if (current['Close'] - current['SMA20']) / current['STD20'] > thresholds['bollinger']['sell_threshold'] else None,
        f"Stochastic %K > {thresholds['stochastic']['sell_threshold']:.2f}" if current['Percent_K'] > thresholds['stochastic']['sell_threshold'] else None,
        "BB Width > 0.3" if current['BB_Width'] > 0.3 else None
    ]
    buy_signals = [s for s in buy_signals if s]
    sell_signals = [s for s in sell_signals if s]

    net_score = len(buy_signals) - len(sell_signals)
    normalized_score = net_score / max(len(buy_signals) + len(sell_signals), 1)
    volatility_factor = min(current['ATR'] / 10.0, 1.0)
    sentiment_factor = 1.0 if fear_greed > 50 else 0.5 if fear_greed < 30 else 0.75
    adjustment = max(min(0.1 * abs(normalized_score) * sentiment_factor * (1 - volatility_factor), 0.15), 0.05)

    reasoning = []
    if normalized_score > 0:
        tsla_change = adjustment if current_tsla_weight < 0.9 else 0.0
        tsll_change = -adjustment if current_tsll_weight > 0.1 else 0.0
        reasoning.append(f"Bullish signal (Score: {normalized_score:.2f}): Increase TSLA weight / Reduce TSLL weight")
    elif normalized_score < 0:
        tsla_change = -adjustment if current_tsla_weight > 0.1 else 0.0
        tsll_change = -adjustment if current_tsll_weight > 0.1 else 0.0
        reasoning.append(f"Bearish signal (Score: {normalized_score:.2f}): Reduce leveraged asset weight")
    else:
        tsla_change = tsll_change = 0.0
        reasoning.append("Neutral signal: No significant adjustment needed")

    target_tsla_weight = max(min(current_tsla_weight + tsla_change, 1.0), 0.0)
    target_tsll_weight = max(min(current_tsll_weight + tsll_change, 1.0 - target_tsla_weight), 0.0)
    target_tsla_shares = math.floor((total_value * target_tsla_weight) / prices.get('TSLA', 1)) if prices.get('TSLA', 0) > 0 else 0
    target_tsll_shares = math.floor((total_value * target_tsll_weight) / prices.get('TSLL', 1)) if prices.get('TSLL', 0) > 0 else 0

    print(f"\n{'='*50}\n📈 **Portfolio Adjustment Suggestions**\n{'='*50}")
    print(f"**TSLA Suggestion:** {'Increase' if tsla_change > 0 else 'Decrease' if tsla_change < 0 else 'Hold'} ({abs(tsla_change)*100:.1f}%)")
    print(f"**TSLL Suggestion:** {'Increase' if tsll_change > 0 else 'Decrease' if tsll_change < 0 else 'Hold'} ({abs(tsll_change)*100:.1f}%)")
    proposed = [
        {'Asset': 'TSLA', 'Current Weight': f"{current_tsla_weight*100:.1f}%", 'Target Weight': f"{target_tsla_weight*100:.1f}%", 'Target Shares': target_tsla_shares},
        {'Asset': 'TSLL', 'Current Weight': f"{current_tsll_weight*100:.1f}%", 'Target Weight': f"{target_tsll_weight*100:.1f}%", 'Target Shares': target_tsll_shares}
    ]
    print("\n**Proposed Portfolio Changes**")
    print(tabulate(pd.DataFrame(proposed), headers='keys', tablefmt='fancy_grid', showindex=False))
    print("\n**Adjustment Rationale**")
    print(f"- Optimized Thresholds: RSI (Buy={thresholds['rsi']['buy_threshold']:.2f}, Sell={thresholds['rsi']['sell_threshold']:.2f}), "
          f"MACD (Buy={thresholds['macd']['buy_threshold']:.2f}, Sell={thresholds['macd']['sell_threshold']:.2f}), "
          f"Bollinger (Buy={thresholds['bollinger']['buy_threshold']:.2f}, Sell={thresholds['bollinger']['sell_threshold']:.2f}), "
          f"Stochastic (Buy={thresholds['stochastic']['buy_threshold']:.2f}, Sell={thresholds['stochastic']['sell_threshold']:.2f})")
    print("- Buy Signals:" + ("\n  . " + "\n  . ".join(buy_signals) if buy_signals else " None"))
    print("- Sell Signals:" + ("\n  . " + "\n  . ".join(sell_signals) if sell_signals else " None"))
    print(f"- Signal Score: {normalized_score:.2f} (Positive: Bullish, Negative: Bearish)")
    print(f"- Volatility (ATR): ${current['ATR']:.2f}")
    print(f"- Fear & Greed Index: {fear_greed:.2f} ({'Fear' if fear_greed < 30 else 'Greed' if fear_greed > 70 else 'Neutral'})")
    print("- Conclusion:" + ("\n  . " + "\n  . ".join(reasoning)))

# 메인 함수
def main():
    """데이터 수집, 포트폴리오 관리 및 백테스트를 실행하는 메인 함수입니다."""
    parser = argparse.ArgumentParser(description='Manage portfolio and perform backtesting')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='Comma-separated stock tickers')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting simulation')
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='Backtest end date (YYYY-MM-DD)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    conn = sqlite3.connect(DB_FILE)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS fear_greed_index (date TEXT PRIMARY KEY, fear_greed_index REAL);
        CREATE TABLE IF NOT EXISTS stock_data (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, PRIMARY KEY (Ticker, Date));
        CREATE TABLE IF NOT EXISTS vix_data (Date TEXT PRIMARY KEY, VIX_Close REAL);
        CREATE TABLE IF NOT EXISTS technical_indicators (Ticker TEXT, Date TEXT, RSI_14 REAL, RSI_5 REAL, MACD REAL, Signal REAL, MACD_Histogram REAL, Short_MACD REAL, Short_Signal REAL, Short_MACD_Histogram REAL, SMA20 REAL, Upper_Band REAL, Lower_Band REAL, BB_Width REAL, SMA5 REAL, SMA10 REAL, SMA50 REAL, SMA200 REAL, Weekly_RSI REAL, Percent_K REAL, Percent_D REAL, OBV REAL, ATR Disadvantages REAL, VWAP REAL, PRIMARY KEY (Ticker, Date));
        CREATE TABLE IF NOT EXISTS trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Ticker TEXT, Action TEXT, Shares INTEGER, Price REAL);
        CREATE TABLE IF NOT EXISTS portfolio_state (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Cash REAL, Ticker TEXT, Shares INTEGER);
        CREATE TABLE IF NOT EXISTS optimized_thresholds (Ticker TEXT, Strategy TEXT, Buy_Threshold REAL, Sell_Threshold REAL, PRIMARY KEY (Ticker, Strategy));
    ''')

    get_fear_greed_data(conn)
    for ticker in tickers:
        get_stock_data(ticker, conn)
    get_vix_data(conn)

    load_trade_log_to_db(conn)
    process_portfolio(conn)

    df_dict = {}
    for ticker in tickers:
        df = pd.read_sql(f'SELECT * FROM stock_data WHERE Ticker="{ticker}"', conn)
        if df.empty:
            print(f"No stock data available for {ticker}.")
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = calculate_all_indicators(df)
        df_dict[ticker] = df
        save_indicators_to_db(df, ticker, conn)

    initial_investment = get_initial_investment(conn)
    cash, holdings, prices, total_value = get_current_portfolio(conn, tickers)
    if total_value is None:
        print("No portfolio data available.")
        conn.close()
        return

    visualize_portfolio_text(conn, tickers, initial_investment)
    display_market_indicators(df_dict, conn)

    strategies = ['rsi', 'macd', 'bollinger', 'stochastic']
    if args.backtest:
        for ticker in tickers:
            df = load_backtest_data(ticker, conn, args.start_date, args.end_date)
            if df is None:
                continue
            for strategy in strategies:
                optimized_thresholds = genetic_algorithm_optimize(df, ticker, strategy=strategy, conn=conn)
                portfolio_df, trade_df = backtest_strategy(df, ticker, strategy=strategy, **optimized_thresholds)
                metrics = calculate_performance_metrics(portfolio_df, initial_investment)
                display_backtest_results(portfolio_df, trade_df, metrics, strategy, ticker)

    check_weight_adjustment(df_dict, tickers, total_value, prices, conn, holdings)
    conn.close()

if __name__ == "__main__":
    main()
