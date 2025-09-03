import os
import warnings
import numpy as np
import pandas as pd

first_period_start_date = "2014-09-17"
first_period_end_date = "2020-12-31"
second_period_start_date = "2021-01-01"
second_period_end_date = "2025-08-31"

warnings.filterwarnings("ignore")
os.makedirs("data/data_processed_hmm", exist_ok=True)

def preprocess(input_file, output_file):
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_5'] = df['log_returns'].rolling(window=5).std()
    df['momentum_5'] = df['log_returns'].rolling(window=5).mean()
    df = df.dropna(subset=['log_returns','volatility_5','momentum_5']).copy()
    df.to_csv(output_file, index=False)
    print("processed rows:", len(df))

preprocess(
    f"data/data_ohlc/BTC-USD_yfinance_{first_period_start_date}_{first_period_end_date}.csv",
    "data/data_processed_hmm/1-btc_processed_train.csv"
)

preprocess(
    f"data/data_ohlc/BTC-USD_yfinance_{second_period_start_date}_{second_period_end_date}.csv",
    "data/data_processed_hmm/1-btc_processed_test.csv"
)
