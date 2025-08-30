import pandas as pd
import mplfinance as mpf
import os

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data_ohlc', 'btc_usdt_1d_2020_2024.csv')
csv_path = os.path.abspath(csv_path)

df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
mpf.plot(df, type='candle', volume=True, style='charles')
