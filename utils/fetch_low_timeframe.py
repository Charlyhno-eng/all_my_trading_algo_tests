import ccxt
import pandas as pd
import time
import os

if not os.path.exists('data/data_ohlc'):
    os.makedirs('data/data_ohlc')

exchange = ccxt.mexc({
    "options": {"defaultType": "future"}
})

symbol = 'PUMP/USDT'
token_pair = 'PUMP_USDT'
timeframe = '1m'

start_date = '2025-09-06T00:00:00Z'
end_date = '2025-09-06T23:59:59Z'

since = exchange.parse8601(start_date)
end = exchange.parse8601(end_date)
all_ohlcv = []
save_data_filename = f'{token_pair}_mexc_future_{timeframe}_2025-09-06_2025-09-07.csv'

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

csv_path = os.path.join('data/data_ohlc', save_data_filename)
df.to_csv(csv_path, index=False)
print(f"Téléchargement terminé - fichier sauvegardé : {csv_path}")
