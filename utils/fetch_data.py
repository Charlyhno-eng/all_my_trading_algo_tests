import ccxt
import pandas as pd
import time
import os

if not os.path.exists('data_ohlc'):
    os.makedirs('data_ohlc')

exchange = ccxt.binance()
symbol = 'BTC/USDT'
token_pair = 'btc_usdt'
timeframes = ['1d', '4h', '1h', '30m']
start_date = '2017-01-01T00:00:00Z'
end_date = '2025-08-31T23:59:59Z'

for timeframe in timeframes:
    since = exchange.parse8601(start_date)
    end = exchange.parse8601(end_date)
    all_ohlcv = []
    save_data_filename = f'{token_pair}_{timeframe}_2017_2025.csv'
    while since < end:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
            print(f"{timeframe} jusqu'à : {pd.to_datetime(ohlcv[-1][0], unit='ms')} - total bougies {len(all_ohlcv)}")
        except Exception as e:
            print("Erreur:", e)
            time.sleep(10)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    csv_path = os.path.join('data_ohlc', save_data_filename)
    df.to_csv(csv_path, index=False)
    print(f"Téléchargement terminé pour {timeframe} - fichier sauvegardé : {csv_path}")
