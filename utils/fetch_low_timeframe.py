import ccxt
import pandas as pd
import time
import os

# --- Config générale ---
if not os.path.exists('data/data_ohlc'):
    os.makedirs('data/data_ohlc')

exchange = ccxt.mexc({"options": {"defaultType": "future"}})

symbol = 'BTC/USDT'
token_pair = 'BTC_USDT'
timeframe = '1h'

# --- Périodes à télécharger ---
train_start = '2020-01-01T00:00:00Z'
train_end   = '2021-12-31T23:59:59Z'

test_start  = '2022-01-01T00:00:00Z'
test_end    = '2025-08-31T23:59:59Z'


def fetch_ohlcv_to_csv(symbol, timeframe, start_date, end_date, save_filename):
    """Télécharge l'historique OHLCV pour une période et sauvegarde en CSV."""
    since = exchange.parse8601(start_date)
    end   = exchange.parse8601(end_date)
    all_ohlcv = []

    while since < end:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # continuer à partir de la dernière bougie
            time.sleep(exchange.rateLimit / 1000)
            print(f"{timeframe} jusqu'à : {pd.to_datetime(ohlcv[-1][0], unit='ms')} "
                  f"- total bougies {len(all_ohlcv)}")
        except Exception as e:
            print("Erreur:", e)
            time.sleep(10)

    # Transformer en DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])

    # Sauvegarde CSV
    csv_path = os.path.join('data/data_ohlc', save_filename)
    df.to_csv(csv_path, index=False)
    print(f"Téléchargement terminé - fichier sauvegardé : {csv_path}")


# --- Télécharger train et test ---
train_file = f"{token_pair}_{timeframe}_mexc_future_{train_start[:10]}_{train_end[:10]}.csv"
test_file  = f"{token_pair}_{timeframe}_mexc_future_{test_start[:10]}_{test_end[:10]}.csv"

fetch_ohlcv_to_csv(symbol, timeframe, train_start, train_end, train_file)
fetch_ohlcv_to_csv(symbol, timeframe, test_start, test_end, test_file)
