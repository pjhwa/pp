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
import json

# 상수 정의
DB_FILE = 'market_data.db'
TRADE_LOG_FILE = 'trade_log.csv'
INITIAL_INVESTMENT = 100000.0
TRANSACTION_COST_RATE = 0.001  # 0.1% 거래 비용

# DEAP 설정
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
    except Exception:
        pass

def get_stock_data(ticker, conn):
    """주식 데이터를 점진적으로 가져와 저장합니다."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(Date) FROM stock_data WHERE Ticker='{ticker}'")
        last_date = cursor.fetchone()[0]
        start_date = (datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)) if last_date else (datetime.now() - timedelta(days=730)).date()
        end_date = datetime.now().date()
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
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
    except Exception:
        pass

def get_vix_data(conn):
    """VIX 데이터를 가져와 저장합니다."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM vix_data")
        last_date = cursor.fetchone()[0]
        start_date = (datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)) if last_date else (datetime.now() - timedelta(days=730)).date()
        end_date = datetime.now().date()
        vix_df = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False, errors='ignore')
        if vix_df.empty:
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
    except Exception:
        return None

# 지표 계산 함수
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def calculate_weekly_rsi(df, period=14):
    if 'Date' not in df.columns:
        df = df.reset_index()
    df_weekly = df.resample('W', on='Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    df_weekly = calculate_rsi(df_weekly, period)
    return df_weekly.rename(columns={f'RSI_{period}': 'Weekly_RSI'})['Weekly_RSI']

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

def calculate_sma(df, period):
    df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
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

def calculate_vwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_volume_change(df):
    df['Volume_Change'] = df['Volume'].pct_change()
    return df

def calculate_all_indicators(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex")
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
    df = calculate_volume_change(df)
    weekly_rsi = calculate_weekly_rsi(df)
    weekly_rsi = weekly_rsi.reindex(df.index, method='ffill')
    df['Weekly_RSI'] = weekly_rsi
    return df

# 신호 생성 및 점수화 함수
def generate_signals(df, thresholds, fear_greed, vix):
    buy_signals = []
    sell_signals = []

    if fear_greed <= thresholds['fg_buy']:
        buy_signals.append("Fear & Greed Index ≤ fg_buy")
    if fear_greed >= thresholds['fg_sell']:
        sell_signals.append("Fear & Greed Index ≥ fg_sell")

    if df['RSI_14'].iloc[-1] < thresholds['daily_rsi_buy']:
        buy_signals.append("Daily RSI < daily_rsi_buy")
    if df['Weekly_RSI'].iloc[-1] < thresholds['weekly_rsi_buy']:
        buy_signals.append("Weekly RSI < weekly_rsi_buy")
    if df['RSI_5'].iloc[-1] < thresholds['short_rsi_buy']:
        buy_signals.append("Short RSI < short_rsi_buy")
    if df['RSI_14'].iloc[-1] > thresholds['daily_rsi_sell']:
        sell_signals.append("Daily RSI > daily_rsi_sell")
    if df['Weekly_RSI'].iloc[-1] > thresholds['weekly_rsi_sell']:
        sell_signals.append("Weekly RSI > weekly_rsi_sell")
    if df['RSI_5'].iloc[-1] > thresholds['short_rsi_sell']:
        sell_signals.append("Short RSI > short_rsi_sell")

    if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['Signal'].iloc[-1] < 0:
        buy_signals.append("MACD > MACD Signal and MACD Signal < 0")
    if df['MACD_Histogram'].iloc[-1] > 0:
        buy_signals.append("MACD Histogram > 0")
    if df['MACD'].iloc[-1] < df['Signal'].iloc[-1] and df['Signal'].iloc[-1] > 0:
        sell_signals.append("MACD < MACD Signal and MACD Signal > 0")
    if df['MACD_Histogram'].iloc[-1] < 0:
        sell_signals.append("MACD Histogram < 0")

    if df['Volume_Change'].iloc[-1] > thresholds['volume_change_strong_buy']:
        buy_signals.append("Volume Change > volume_change_strong_buy")
    if df['Volume_Change'].iloc[-1] > thresholds['volume_change_weak_buy']:
        buy_signals.append("Volume Change > volume_change_weak_buy")
    if df['Volume_Change'].iloc[-1] < thresholds['volume_change_sell']:
        sell_signals.append("Volume Change < volume_change_sell")

    if df['Close'].iloc[-1] < df['Lower_Band'].iloc[-1]:
        buy_signals.append("Close < Lower Bollinger Band")
    if df['Close'].iloc[-1] > df['Upper_Band'].iloc[-1]:
        sell_signals.append("Close > Upper Bollinger Band")

    if len(df) >= 2:
        if df['RSI_14'].iloc[-1] > df['RSI_14'].iloc[-2] and df['Close'].iloc[-1] > df['SMA200'].iloc[-1]:
            buy_signals.append("RSI Increasing and Close > SMA200")
        if df['RSI_14'].iloc[-1] < df['RSI_14'].iloc[-2] and df['Close'].iloc[-1] < df['SMA200'].iloc[-1]:
            sell_signals.append("RSI Decreasing and Close < SMA200")

    if df['Percent_K'].iloc[-1] < thresholds['stochastic_buy']:
        buy_signals.append("Stochastic %K < stochastic_buy")
    if df['Percent_K'].iloc[-1] > thresholds['stochastic_sell']:
        sell_signals.append("Stochastic %K > stochastic_sell")

    if len(df) >= 2:
        if df['OBV'].iloc[-1] > df['OBV'].iloc[-2]:
            buy_signals.append("OBV Increasing")
        if df['OBV'].iloc[-1] < df['OBV'].iloc[-2]:
            sell_signals.append("OBV Decreasing")

    if df['BB_Width'].iloc[-1] < thresholds['bb_width_low']:
        buy_signals.append("BB Width < bb_width_low")
    if df['BB_Width'].iloc[-1] > thresholds['bb_width_high']:
        sell_signals.append("BB Width > bb_width_high")

    if df['SMA5'].iloc[-1] > df['SMA10'].iloc[-1]:
        buy_signals.append("SMA5 > SMA10")
    if df['SMA5'].iloc[-1] < df['SMA10'].iloc[-1]:
        sell_signals.append("SMA5 < SMA10")

    if df['Short_MACD'].iloc[-1] > df['Short_Signal'].iloc[-1]:
        buy_signals.append("Short MACD > Signal")
    if df['Short_MACD'].iloc[-1] < df['Short_Signal'].iloc[-1]:
        sell_signals.append("Short MACD < Signal")

    if df['Close'].iloc[-1] > df['VWAP'].iloc[-1]:
        buy_signals.append("Close > VWAP")
    if df['Close'].iloc[-1] < df['VWAP'].iloc[-1]:
        sell_signals.append("Close < VWAP")

    # 신호 점수화 (10점 만점)
    buy_points = min(len(buy_signals) * 2, 10)  # 신호 1개당 2점, 최대 10점
    sell_points = min(len(sell_signals) * 2, 10)
    return buy_signals, sell_signals, buy_points, sell_points

# 포트폴리오 조정 함수
def adjust_portfolio(df, buy_signals, sell_signals, thresholds, weights, current_weight, atr):
    strong_buy_count = sum(1 for signal in buy_signals if "strong" in signal.lower())
    weak_buy_count = sum(1 for signal in buy_signals if "weak" in signal.lower())
    other_buy_count = len(buy_signals) - strong_buy_count - weak_buy_count
    sell_count = len(sell_signals)

    obv_increasing = 1 if "OBV Increasing" in buy_signals else 0
    obv_decreasing = 1 if "OBV Decreasing" in sell_signals else 0
    bb_width_low = 1 if "BB Width < bb_width_low" in buy_signals else 0
    bb_width_high = 1 if "BB Width > bb_width_high" in sell_signals else 0
    short_buy_count = sum(1 for signal in buy_signals if "Short" in signal)
    short_sell_count = sum(1 for signal in sell_signals if "Short" in signal)

    increase = (weights['w_strong_buy'] * strong_buy_count +
                weights['w_weak_buy'] * weak_buy_count +
                weights['w_weak_buy'] * other_buy_count +
                weights['obv_weight'] * obv_increasing +
                weights['bb_width_weight'] * bb_width_low +
                weights['w_short_buy'] * short_buy_count) * 0.1

    decrease = (weights['w_sell'] * sell_count +
                weights['obv_weight'] * obv_decreasing +
                weights['bb_width_weight'] * bb_width_high +
                weights['w_short_sell'] * short_sell_count) * 0.1

    net_adjustment = increase - decrease
    atr_factor = min(atr / 20.0, 0.5)  # ATR 영향 감소 (최대 50% 감소)
    net_adjustment *= (1 - atr_factor)
    target_weight = max(min(current_weight + net_adjustment, 1.0), 0.0)
    return target_weight

# 백테스트 함수
def backtest_strategy(df, ticker, initial_cash=INITIAL_INVESTMENT, thresholds=None, weights=None):
    if thresholds is None or weights is None:
        raise ValueError("Thresholds and weights must be provided")

    first_valid_index = df['SMA200'].first_valid_index()
    if first_valid_index is None:
        raise ValueError("No valid SMA200 data in the DataFrame")
    df = df.loc[first_valid_index:]

    cash = initial_cash
    shares = 0
    portfolio_value = []
    trade_history = []

    for date, row in df.iterrows():
        fear_greed_row = conn.cursor().execute(f"SELECT fear_greed_index FROM fear_greed_index WHERE date <= '{date.strftime('%Y-%m-%d')}' ORDER BY date DESC LIMIT 1").fetchone()
        fear_greed = fear_greed_row[0] if fear_greed_row else 50.0
        vix_row = conn.cursor().execute(f"SELECT VIX_Close FROM vix_data WHERE Date <= '{date.strftime('%Y-%m-%d')}' ORDER BY Date DESC LIMIT 1").fetchone()
        vix = vix_row[0] if vix_row else 20.0
        buy_signals, sell_signals, _, _ = generate_signals(df.loc[:date], thresholds, fear_greed, vix)
        current_weight = (shares * row['Close']) / (cash + shares * row['Close']) if (cash + shares * row['Close']) > 0 else 0.0
        target_weight = adjust_portfolio(df.loc[:date], buy_signals, sell_signals, thresholds, weights, current_weight, row['ATR'])

        target_shares = math.floor((target_weight * (cash + shares * row['Close'])) / row['Close'] / 100) * 100
        if target_shares > shares:
            shares_to_buy = target_shares - shares
            cost = shares_to_buy * row['Close'] * (1 + TRANSACTION_COST_RATE)
            if cash >= cost:
                shares += shares_to_buy
                cash -= cost
                trade_history.append({'Date': date, 'Ticker': ticker, 'Action': 'Buy', 'Shares': shares_to_buy, 'Price': row['Close'], 'Cost': cost})
        elif target_shares < shares:
            shares_to_sell = shares - target_shares
            revenue = shares_to_sell * row['Close'] * (1 - TRANSACTION_COST_RATE)
            shares -= shares_to_sell
            cash += revenue
            trade_history.append({'Date': date, 'Ticker': ticker, 'Action': 'Sell', 'Shares': shares_to_sell, 'Price': row['Close'], 'Revenue': revenue})

        total_value = cash + (shares * row['Close'])
        portfolio_value.append({'Date': date, 'Portfolio Value': total_value})

    return pd.DataFrame(portfolio_value), pd.DataFrame(trade_history)

# 데이터베이스 관리 함수
def save_optimal_parameters(conn, ticker, thresholds, weights, return_rate):
    cursor = conn.cursor()
    thresholds_json = json.dumps(thresholds)
    weights_json = json.dumps(weights)
    cursor.execute("INSERT OR REPLACE INTO optimal_parameters (ticker, thresholds, weights, return_rate) VALUES (?, ?, ?, ?)",
                   (ticker, thresholds_json, weights_json, return_rate))
    conn.commit()

def load_optimal_parameters(conn, ticker):
    cursor = conn.cursor()
    cursor.execute("SELECT thresholds, weights, return_rate FROM optimal_parameters WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    if row:
        thresholds = json.loads(row[0])
        weights = json.loads(row[1])
        return_rate = row[2]
        return thresholds, weights, return_rate
    return None, None, None

# 유전 알고리즘 최적화 함수
def genetic_algorithm_optimize(df, ticker, generations=50, pop_size=100):
    toolbox = base.Toolbox()

    toolbox.register("fg_buy", np.random.uniform, 10, 50)
    toolbox.register("fg_sell", np.random.uniform, 50, 90)
    toolbox.register("daily_rsi_buy", np.random.uniform, 10, 50)
    toolbox.register("daily_rsi_sell", np.random.uniform, 50, 90)
    toolbox.register("weekly_rsi_buy", np.random.uniform, 10, 50)
    toolbox.register("weekly_rsi_sell", np.random.uniform, 50, 90)
    toolbox.register("short_rsi_buy", np.random.uniform, 10, 50)
    toolbox.register("short_rsi_sell", np.random.uniform, 50, 90)
    toolbox.register("stochastic_buy", np.random.uniform, 5, 40)
    toolbox.register("stochastic_sell", np.random.uniform, 60, 95)
    toolbox.register("volume_change_strong_buy", np.random.uniform, 0.1, 1.0)
    toolbox.register("volume_change_weak_buy", np.random.uniform, 0.01, 0.3)
    toolbox.register("volume_change_sell", np.random.uniform, -1.0, -0.1)
    toolbox.register("bb_width_low", np.random.uniform, 0.01, 0.3)
    toolbox.register("bb_width_high", np.random.uniform, 0.3, 1.0)
    toolbox.register("w_strong_buy", np.random.uniform, 1, 5)
    toolbox.register("w_weak_buy", np.random.uniform, 0.5, 3)
    toolbox.register("w_sell", np.random.uniform, 1, 5)
    toolbox.register("obv_weight", np.random.uniform, 0.5, 3)
    toolbox.register("bb_width_weight", np.random.uniform, 0.5, 3)
    toolbox.register("w_short_buy", np.random.uniform, 0.5, 3)
    toolbox.register("w_short_sell", np.random.uniform, 0.5, 3)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.fg_buy, toolbox.fg_sell, toolbox.daily_rsi_buy, toolbox.daily_rsi_sell,
                      toolbox.weekly_rsi_buy, toolbox.weekly_rsi_sell, toolbox.short_rsi_buy, toolbox.short_rsi_sell,
                      toolbox.stochastic_buy, toolbox.stochastic_sell, toolbox.volume_change_strong_buy,
                      toolbox.volume_change_weak_buy, toolbox.volume_change_sell, toolbox.bb_width_low,
                      toolbox.bb_width_high, toolbox.w_strong_buy, toolbox.w_weak_buy, toolbox.w_sell,
                      toolbox.obv_weight, toolbox.bb_width_weight, toolbox.w_short_buy, toolbox.w_short_sell), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        thresholds = {
            'fg_buy': individual[0], 'fg_sell': individual[1],
            'daily_rsi_buy': individual[2], 'daily_rsi_sell': individual[3],
            'weekly_rsi_buy': individual[4], 'weekly_rsi_sell': individual[5],
            'short_rsi_buy': individual[6], 'short_rsi_sell': individual[7],
            'stochastic_buy': individual[8], 'stochastic_sell': individual[9],
            'volume_change_strong_buy': individual[10], 'volume_change_weak_buy': individual[11],
            'volume_change_sell': individual[12], 'bb_width_low': individual[13], 'bb_width_high': individual[14]
        }
        weights = {
            'w_strong_buy': individual[15], 'w_weak_buy': individual[16], 'w_sell': individual[17],
            'obv_weight': individual[18], 'bb_width_weight': individual[19],
            'w_short_buy': individual[20], 'w_short_sell': individual[21]
        }
        portfolio_df, _ = backtest_strategy(df, ticker, thresholds=thresholds, weights=weights)
        final_value = portfolio_df['Portfolio Value'].iloc[-1]
        return_rate = (final_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
        return return_rate,

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.3)  # 돌연변이 확률 증가
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    for gen in tqdm(range(generations), desc=f"Optimizing for {ticker}"):
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.8, mutpb=0.3)  # 교차 및 돌연변이 확률 조정
        fits = list(map(toolbox.evaluate, pop))
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit
        pop = toolbox.select(pop, k=len(pop))
        hof.update(pop)

    best_params = hof[0]
    optimized_thresholds = {
        'fg_buy': best_params[0], 'fg_sell': best_params[1],
        'daily_rsi_buy': best_params[2], 'daily_rsi_sell': best_params[3],
        'weekly_rsi_buy': best_params[4], 'weekly_rsi_sell': best_params[5],
        'short_rsi_buy': best_params[6], 'short_rsi_sell': best_params[7],
        'stochastic_buy': best_params[8], 'stochastic_sell': best_params[9],
        'volume_change_strong_buy': best_params[10], 'volume_change_weak_buy': best_params[11],
        'volume_change_sell': best_params[12], 'bb_width_low': best_params[13], 'bb_width_high': best_params[14]
    }
    optimized_weights = {
        'w_strong_buy': best_params[15], 'w_weak_buy': best_params[16], 'w_sell': best_params[17],
        'obv_weight': best_params[18], 'bb_width_weight': best_params[19],
        'w_short_buy': best_params[20], 'w_short_sell': best_params[21]
    }
    return_rate = hof[0].fitness.values[0]
    return optimized_thresholds, optimized_weights, return_rate

# 시뮬레이션 함수
def simulate_portfolio(conn, start_date, end_date, initial_cash=INITIAL_INVESTMENT):
    thresholds, weights, _ = load_optimal_parameters(conn, 'TSLA')
    if thresholds is None or weights is None:
        print("Optimal parameters not found. Please run backtest first.")
        return

    df_tsla = pd.read_sql(f"SELECT * FROM stock_data WHERE Ticker='TSLA' AND Date >= '{start_date}' AND Date <= '{end_date}'", conn)
    df_tsll = pd.read_sql(f"SELECT * FROM stock_data WHERE Ticker='TSLL' AND Date >= '{start_date}' AND Date <= '{end_date}'", conn)
    if df_tsla.empty or df_tsll.empty:
        print("Insufficient data for simulation.")
        return

    df_tsla['Date'] = pd.to_datetime(df_tsla['Date'])
    df_tsll['Date'] = pd.to_datetime(df_tsll['Date'])
    df_tsla = df_tsla.set_index('Date')
    df_tsll = df_tsll.set_index('Date')
    df_tsla = calculate_all_indicators(df_tsla)
    df_tsll = calculate_all_indicators(df_tsll)

    first_valid_date_tsla = df_tsla['SMA200'].first_valid_index()
    first_valid_date_tsll = df_tsll['SMA200'].first_valid_index()
    if first_valid_date_tsla is None or first_valid_date_tsll is None:
        print("Insufficient data for SMA200 calculation.")
        return

    simulation_start = max(first_valid_date_tsla, first_valid_date_tsll)
    dates = df_tsla.index.intersection(df_tsll.index)
    simulation_dates = dates[(dates >= simulation_start) & (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))]
    if simulation_dates.empty:
        print("No valid simulation dates with SMA200 data.")
        return

    cash = initial_cash
    holdings = {'TSLA': 0, 'TSLL': 0}
    portfolio_values = []

    for date in simulation_dates:
        price_tsla = df_tsla.loc[date, 'Close']
        price_tsll = df_tsll.loc[date, 'Close']
        df_up_to_date = df_tsla.loc[:date]

        fear_greed_row = conn.cursor().execute(f"SELECT fear_greed_index FROM fear_greed_index WHERE date <= '{date.strftime('%Y-%m-%d')}' ORDER BY date DESC LIMIT 1").fetchone()
        fear_greed = fear_greed_row[0] if fear_greed_row else 50.0
        vix_row = conn.cursor().execute(f"SELECT VIX_Close FROM vix_data WHERE Date <= '{date.strftime('%Y-%m-%d')}' ORDER BY Date DESC LIMIT 1").fetchone()
        vix = vix_row[0] if vix_row else 20.0

        buy_signals, sell_signals, buy_points, sell_points = generate_signals(df_up_to_date, thresholds, fear_greed, vix)
        current_value = cash + holdings['TSLA'] * price_tsla + holdings['TSLL'] * price_tsll
        current_tsla_weight = (holdings['TSLA'] * price_tsla) / current_value if current_value > 0 else 0.0
        atr = df_up_to_date['ATR'].iloc[-1] if 'ATR' in df_up_to_date.columns else 0.0
        target_tsla_weight = adjust_portfolio(df_up_to_date, buy_signals, sell_signals, thresholds, weights, current_tsla_weight, atr)
        target_tsll_weight = 1.0 - target_tsla_weight

        target_tsla_shares = math.floor((current_value * target_tsla_weight) / price_tsla / 100) * 100
        target_tsll_shares = math.floor((current_value * target_tsll_weight) / price_tsll / 100) * 100

        if target_tsla_shares > holdings['TSLA']:
            shares_to_buy = target_tsla_shares - holdings['TSLA']
            cost = shares_to_buy * price_tsla * (1 + TRANSACTION_COST_RATE)
            if cash >= cost:
                holdings['TSLA'] += shares_to_buy
                cash -= cost
        elif target_tsla_shares < holdings['TSLA']:
            shares_to_sell = holdings['TSLA'] - target_tsla_shares
            revenue = shares_to_sell * price_tsla * (1 - TRANSACTION_COST_RATE)
            holdings['TSLA'] -= shares_to_sell
            cash += revenue

        if target_tsll_shares > holdings['TSLL']:
            shares_to_buy = target_tsll_shares - holdings['TSLL']
            cost = shares_to_buy * price_tsll * (1 + TRANSACTION_COST_RATE)
            if cash >= cost:
                holdings['TSLL'] += shares_to_buy
                cash -= cost
        elif target_tsll_shares < holdings['TSLL']:
            shares_to_sell = holdings['TSLL'] - target_tsll_shares
            revenue = shares_to_sell * price_tsll * (1 - TRANSACTION_COST_RATE)
            holdings['TSLL'] -= shares_to_sell
            cash += revenue

        total_value = cash + holdings['TSLA'] * price_tsla + holdings['TSLL'] * price_tsll
        tsla_weight = (holdings['TSLA'] * price_tsla) / total_value * 100 if total_value > 0 else 0.0
        tsll_weight = (holdings['TSLL'] * price_tsll) / total_value * 100 if total_value > 0 else 0.0
        portfolio_values.append({
            'Date': date,
            'Portfolio Value': total_value,
            'Portfolio Weight': f"TSLA:{tsla_weight:.1f}% TSLL:{tsll_weight:.1f}%",
            'Signals': f"Buy Signals: {buy_points} points, Sell Signals: {sell_points} points"
        })

    portfolio_df = pd.DataFrame(portfolio_values)
    final_value = portfolio_df['Portfolio Value'].iloc[-1]
    return_rate = (final_value - initial_cash) / initial_cash * 100
    simulation_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    print(f"\n{'='*50}\nSimulation Results:\n{'='*50}")
    print(f"Simulation Duration: {start_date} ~ {end_date} ({simulation_days} Days)")
    print(f"Initial Investment: ${initial_cash:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return Rate: {return_rate:.2f}%")
    print("\nDaily Portfolio Values (Last 15 Days):")
    if len(portfolio_df) > 15:
        print(tabulate(portfolio_df.tail(15), headers='keys', tablefmt='fancy_grid', showindex=False))
    else:
        print(tabulate(portfolio_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    portfolio_df.to_csv('simulation_portfolio_values.csv', index=False)
    print("Full portfolio values saved to 'simulation_portfolio_values.csv'")

# 데이터베이스 관리 함수
def save_indicators_to_db(df, ticker, conn):
    indicators = ['RSI_14', 'RSI_5', 'MACD', 'Signal', 'MACD_Histogram', 'Short_MACD', 'Short_Signal',
                  'Short_MACD_Histogram', 'SMA20', 'Upper_Band', 'Lower_Band', 'BB_Width', 'SMA5',
                  'SMA10', 'SMA50', 'SMA200', 'Weekly_RSI', 'Percent_K', 'Percent_D', 'OBV', 'ATR', 'VWAP', 'Volume_Change']
    df_indicators = df.reset_index()[['Date'] + [col for col in indicators if col in df.columns]].copy()
    df_indicators['Ticker'] = ticker
    df_indicators['Date'] = df_indicators['Date'].dt.strftime('%Y-%m-%d')
    existing_dates = pd.read_sql(f"SELECT Date FROM technical_indicators WHERE Ticker='{ticker}'", conn)['Date'].tolist()
    df_indicators = df_indicators[~df_indicators['Date'].isin(existing_dates)]
    if not df_indicators.empty:
        df_indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)

def load_trade_log_to_db(conn):
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        df['Shares'] = df['Shares'].astype(int)
        df.to_sql('trade_log', conn, if_exists='replace', index=False)

def process_portfolio(conn):
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
    if not os.path.exists(TRADE_LOG_FILE):
        return INITIAL_INVESTMENT
    trade_log = pd.read_csv(TRADE_LOG_FILE)
    hold_trades = trade_log[trade_log['Action'].str.lower() == 'hold']
    return sum(trade['Shares'] * trade['Price'] for _, trade in hold_trades.iterrows()) if not hold_trades.empty else INITIAL_INVESTMENT

# 시각화 함수
def visualize_portfolio_text(conn, tickers, initial_investment):
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
    if 'TSLA' not in df_dict or df_dict['TSLA'].empty:
        print("No data available for TSLA.")
        return

    latest_tsla = df_dict['TSLA'].iloc[-1]
    fear_greed = conn.cursor().execute("SELECT fear_greed_index FROM fear_greed_index ORDER BY date DESC LIMIT 1").fetchone()
    fear_greed = fear_greed[0] if fear_greed else 'N/A'

    vix_row = conn.cursor().execute('SELECT VIX_Close FROM vix_data ORDER BY Date DESC LIMIT 1').fetchone()
    vix_value = vix_row[0] if vix_row else 'N/A'
    vix_trend = 'High' if vix_row and vix_row[0] > 20 else 'Low' if vix_row else 'N/A'

    indicators = {
        'Momentum': [
            {'Indicator': 'Fear & Greed Index', 'Value': f"{fear_greed:.2f}" if isinstance(fear_greed, float) else fear_greed, 'Trend/Notes': 'Fear' if isinstance(fear_greed, float) and fear_greed < 30 else 'Greed' if isinstance(fear_greed, float) and fear_greed > 70 else 'Neutral'},
            {'Indicator': 'MACD Histogram', 'Value': f"{latest_tsla['MACD_Histogram']:.2f}", 'Trend/Notes': 'Bullish' if latest_tsla['MACD_Histogram'] > 0 else 'Bearish'},
            {'Indicator': 'Daily RSI', 'Value': f"{latest_tsla['RSI_14']:.2f}", 'Trend/Notes': 'Increasing' if len(df_dict['TSLA']) > 1 and latest_tsla['RSI_14'] > df_dict['TSLA']['RSI_14'].iloc[-2] else 'Decreasing'},
            {'Indicator': 'Weekly RSI', 'Value': f"{latest_tsla['Weekly_RSI']:.2f}", 'Trend/Notes': 'Overbought' if latest_tsla['Weekly_RSI'] > 70 else 'Oversold' if latest_tsla['Weekly_RSI'] < 30 else 'Neutral'},
            {'Indicator': 'Short RSI', 'Value': f"{latest_tsla['RSI_5']:.2f}", 'Trend/Notes': 'Overbought' if latest_tsla['RSI_5'] > 70 else 'Oversold' if latest_tsla['RSI_5'] < 30 else 'Neutral'},
        ],
        'Volatility': [
            {'Indicator': 'ATR', 'Value': f"${latest_tsla['ATR']:.2f}", 'Trend/Notes': 'High Volatility' if latest_tsla['ATR'] > 10 else 'Low Volatility'},
            {'Indicator': 'BB Width', 'Value': f"{latest_tsla['BB_Width']:.2f}", 'Trend/Notes': 'High' if latest_tsla['BB_Width'] > 0.3 else 'Low'},
            {'Indicator': 'VIX', 'Value': f"{vix_value:.2f}" if isinstance(vix_value, (int, float)) else vix_value, 'Trend/Notes': vix_trend},
        ],
        'Trend': [
            {'Indicator': 'TSLA Close', 'Value': f"${latest_tsla['Close']:.2f}", 'Trend/Notes': f"{'Above' if latest_tsla['Close'] > latest_tsla['SMA50'] else 'Below'} SMA50"},
            {'Indicator': 'SMA5', 'Value': f"${latest_tsla['SMA5']:.2f}", 'Trend/Notes': f"{'Above' if latest_tsla['SMA5'] > latest_tsla['SMA10'] else 'Below'} SMA10"},
            {'Indicator': 'SMA50', 'Value': f"${latest_tsla['SMA50']:.2f}", 'Trend/Notes': 'Above' if latest_tsla['Close'] > latest_tsla['SMA50'] else 'Below'},
            {'Indicator': 'SMA200', 'Value': f"${latest_tsla['SMA200']:.2f}", 'Trend/Notes': 'Above' if latest_tsla['Close'] > latest_tsla['SMA200'] else 'Below'},
        ],
        'Volume': [
            {'Indicator': 'Volume Change', 'Value': f"{latest_tsla['Volume_Change']:.2%}", 'Trend/Notes': 'Increasing' if latest_tsla['Volume_Change'] > 0 else 'Decreasing'},
            {'Indicator': 'OBV', 'Value': f"{latest_tsla['OBV']:.0f}", 'Trend/Notes': 'Increasing' if latest_tsla['OBV'] > df_dict['TSLA']['OBV'].iloc[-2] else 'Decreasing'},
            {'Indicator': 'VWAP', 'Value': f"${latest_tsla['VWAP']:.2f}", 'Trend/Notes': 'Above' if latest_tsla['Close'] > latest_tsla['VWAP'] else 'Below'},
        ]
    }

    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*50}\n📈 **Market Indicators Summary ({current_date})**\n{'='*50}")
    for category, ind_list in indicators.items():
        print(f"\n**{category}**")
        print(tabulate(pd.DataFrame(ind_list), headers=['Indicator', 'Value', 'Trend/Notes'], tablefmt='fancy_grid', showindex=False))

# 포트폴리오 조정 제안 함수
def check_weight_adjustment(df_dict, tickers, total_value, prices, conn, holdings):
    if 'TSLA' not in df_dict or df_dict['TSLA'].empty:
        print("No data available for TSLA.")
        return

    tsla_df = df_dict['TSLA']
    if len(tsla_df) < 2:
        print("Insufficient data for TSLA to generate signals.")
        return
    current = tsla_df.iloc[-1]
    fear_greed = conn.cursor().execute("SELECT fear_greed_index FROM fear_greed_index ORDER BY date DESC LIMIT 1").fetchone()
    fear_greed = fear_greed[0] if fear_greed else 50.0
    vix = conn.cursor().execute("SELECT VIX_Close FROM vix_data ORDER BY Date DESC LIMIT 1").fetchone()[0] if conn.cursor().execute("SELECT VIX_Close FROM vix_data ORDER BY Date DESC LIMIT 1").fetchone() else 20.0

    tsla_value = holdings.get('TSLA', 0) * prices.get('TSLA', 0.0)
    tsll_value = holdings.get('TSLL', 0) * prices.get('TSLL', 0.0)
    current_tsla_weight = tsla_value / total_value if total_value > 0 else 0.0
    current_tsll_weight = tsll_value / total_value if total_value > 0 else 0.0

    thresholds, weights, existing_return_rate = load_optimal_parameters(conn, 'TSLA')
    if thresholds is None or weights is None:
        thresholds, weights, return_rate = genetic_algorithm_optimize(tsla_df, 'TSLA')
        save_optimal_parameters(conn, 'TSLA', thresholds, weights, return_rate)
        print(f"Initial optimization completed for TSLA with return rate: {return_rate*100:.2f}%")
    else:
        print("Loaded existing optimal parameters for TSLA.")

    buy_signals, sell_signals, buy_points, sell_points = generate_signals(tsla_df, thresholds, fear_greed, vix)
    target_tsla_weight = adjust_portfolio(tsla_df, buy_signals, sell_signals, thresholds, weights, current_tsla_weight, current['ATR'])
    target_tsll_weight = 1.0 - target_tsla_weight

    target_tsla_shares = math.floor((total_value * target_tsla_weight) / prices.get('TSLA', 1)) if prices.get('TSLA', 0) > 0 else 0
    target_tsll_shares = math.floor((total_value * target_tsll_weight) / prices.get('TSLL', 1)) if prices.get('TSLL', 0) > 0 else 0

    print(f"\n{'='*50}\n📈 **Portfolio Adjustment Suggestions**\n{'='*50}")
    print(f"**TSLA Suggestion:** {'Increase' if target_tsla_weight > current_tsla_weight else 'Decrease' if target_tsla_weight < current_tsla_weight else 'Hold'} ({abs(target_tsla_weight - current_tsla_weight)*100:.1f}%)")
    print(f"**TSLL Suggestion:** {'Increase' if target_tsll_weight > current_tsll_weight else 'Decrease' if target_tsll_weight < current_tsll_weight else 'Hold'} ({abs(target_tsll_weight - current_tsll_weight)*100:.1f}%)")
    proposed = [
        {'Asset': 'TSLA', 'Current Weight': f"{current_tsla_weight*100:.1f}%", 'Target Weight': f"{target_tsla_weight*100:.1f}%", 'Target Shares': target_tsla_shares},
        {'Asset': 'TSLL', 'Current Weight': f"{current_tsll_weight*100:.1f}%", 'Target Weight': f"{target_tsll_weight*100:.1f}%", 'Target Shares': target_tsll_shares}
    ]
    print("\n**Proposed Portfolio Changes**")
    print(tabulate(pd.DataFrame(proposed), headers='keys', tablefmt='fancy_grid', showindex=False))
    print("\n**Adjustment Rationale**")
    print(f"Buy Signals (points: {buy_points}):")
    for signal in buy_signals if buy_signals else ["None"]:
        print(f"  . {signal}")
    print(f"Sell Signals (points: {sell_points}):")
    for signal in sell_signals if sell_signals else ["None"]:
        print(f"  . {signal}")
    print(f"- ATR: {current['ATR']:.2f}, VIX: {vix:.2f}")

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description='Manage portfolio and perform backtesting/simulation')
    parser.add_argument('--tickers', type=str, default='TSLA,TSLL', help='Comma-separated stock tickers')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--simulate', action='store_true', help='Run simulation with optimized thresholds')
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    tickers = args.tickers.split(',')

    global conn
    conn = sqlite3.connect(DB_FILE)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS fear_greed_index (date TEXT PRIMARY KEY, fear_greed_index REAL);
        CREATE TABLE IF NOT EXISTS stock_data (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, PRIMARY KEY (Ticker, Date));
        CREATE TABLE IF NOT EXISTS vix_data (Date TEXT PRIMARY KEY, VIX_Close REAL);
        CREATE TABLE IF NOT EXISTS technical_indicators (Ticker TEXT, Date TEXT, RSI_14 REAL, RSI_5 REAL, MACD REAL, Signal REAL, MACD_Histogram REAL, Short_MACD REAL, Short_Signal REAL, Short_MACD_Histogram REAL, SMA20 REAL, Upper_Band REAL, Lower_Band REAL, BB_Width REAL, SMA5 REAL, SMA10 REAL, SMA50 REAL, SMA200 REAL, Weekly_RSI REAL, Percent_K REAL, Percent_D REAL, OBV REAL, ATR REAL, VWAP REAL, Volume_Change REAL, PRIMARY KEY (Ticker, Date));
        CREATE TABLE IF NOT EXISTS trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Ticker TEXT, Action TEXT, Shares INTEGER, Price REAL);
        CREATE TABLE IF NOT EXISTS portfolio_state (id INTEGER PRIMARY KEY AUTOINCREMENT, Date TEXT, Cash REAL, Ticker TEXT, Shares INTEGER);
        CREATE TABLE IF NOT EXISTS optimal_parameters (ticker TEXT PRIMARY KEY, thresholds TEXT, weights TEXT, return_rate REAL);
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

    if args.backtest:
        df = pd.read_sql(f"SELECT * FROM stock_data WHERE Ticker='TSLA' AND Date BETWEEN '{args.start_date}' AND '{args.end_date}'", conn)
        if not df.empty and len(df) >= 200:  # 최소 200일 데이터 필요
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df = calculate_all_indicators(df)
            thresholds, weights, return_rate = genetic_algorithm_optimize(df, 'TSLA')
            existing_thresholds, existing_weights, existing_return_rate = load_optimal_parameters(conn, 'TSLA')
            if existing_return_rate is None or return_rate > existing_return_rate:
                save_optimal_parameters(conn, 'TSLA', thresholds, weights, return_rate)
                print(f"\nUpdated optimal parameters for TSLA with return rate: {return_rate*100:.2f}%")
                print("Optimized Thresholds:")
                for key, value in thresholds.items():
                    print(f"  - {key}: {value:.2f}")
                print("Optimized Weights:")
                for key, value in weights.items():
                    print(f"  - {key}: {value:.2f}")
            else:
                print(f"\nExisting parameters are better or equal with return rate: {existing_return_rate*100:.2f}%. No update.")
            portfolio_df, trade_df = backtest_strategy(df, 'TSLA', thresholds=thresholds, weights=weights)
            metrics = {
                'Total Return (%)': (portfolio_df['Portfolio Value'].iloc[-1] - INITIAL_INVESTMENT) / INITIAL_INVESTMENT * 100,
                'Sharpe Ratio': (portfolio_df['Portfolio Value'].pct_change().mean() / portfolio_df['Portfolio Value'].pct_change().std()) * np.sqrt(252) if portfolio_df['Portfolio Value'].pct_change().std() != 0 else 0
            }
            print(f"\nBacktest Results for TSLA:")
            print(tabulate(portfolio_df.tail(), headers='keys', tablefmt='fancy_grid', showindex=False))
            if not trade_df.empty:
                print(tabulate(trade_df.tail(), headers='keys', tablefmt='fancy_grid', showindex=False))
            print(tabulate(pd.DataFrame([metrics]), headers='keys', tablefmt='fancy_grid', showindex=False))
        else:
            print("Insufficient data for backtest (minimum 200 days required).")
    elif args.simulate:
        simulate_portfolio(conn, args.start_date, args.end_date)
    else:
        check_weight_adjustment(df_dict, tickers, total_value, prices, conn, holdings)

    conn.close()

if __name__ == "__main__":
    main()
