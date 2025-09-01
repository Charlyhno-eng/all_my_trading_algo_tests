import ccxt
import csv
import os

DATA_FOLDER = "data_delta_neutral"
os.makedirs(DATA_FOLDER, exist_ok=True)

def normalize_pair(symbol):
    """Normalise la pair USDT pour comparaison entre MEXC et KuCoin."""
    base, quote = symbol.split(":")[0].split("/")
    return f"{base}/{quote}"

def get_common_pairs():
    mexc = ccxt.mexc({'enableRateLimit': True})
    kucoin = ccxt.kucoinfutures({'enableRateLimit': True})

    mexc_markets = mexc.fetch_markets()
    kucoin_markets = kucoin.fetch_markets()

    mexc_pairs = {normalize_pair(m['symbol']) for m in mexc_markets if ":USDT" in m['symbol']}
    kucoin_pairs = {normalize_pair(m['symbol']) for m in kucoin_markets if ":USDT" in m['symbol']}

    common_pairs = sorted(list(mexc_pairs & kucoin_pairs))
    return common_pairs

def save_to_csv(pairs, filename="common_pairs_mexc_kucoin.csv"):
    filepath = os.path.join(DATA_FOLDER, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pair"])
        for p in pairs:
            writer.writerow([p])
    print(f"{len(pairs)} paires communes enregistrées dans {filepath}")

if __name__ == "__main__":
    pairs = get_common_pairs()
    save_to_csv(pairs)
