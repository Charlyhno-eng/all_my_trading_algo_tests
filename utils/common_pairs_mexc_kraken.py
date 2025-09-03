import ccxt
import csv
import os

DATA_FOLDER = "data/data_delta_neutral"
os.makedirs(DATA_FOLDER, exist_ok=True)

def normalize_pair(symbol, exchange):
    """Transforme la pair pour comparer MEXC USDT et Kraken USD."""
    base, quote = symbol.split(":")[0].split("/")
    if exchange == "mexc" and quote == "USDT":
        quote = "USD"
    return f"{base}/{quote}"

def to_usdt(pair):
    """Remet USDT pour l'affichage dans le CSV."""
    base, quote = pair.split("/")
    if quote == "USD":
        quote = "USDT"
    return f"{base}/{quote}"

def get_common_pairs():
    mexc = ccxt.mexc({'enableRateLimit': True})
    kraken = ccxt.krakenfutures({'enableRateLimit': True})

    mexc_markets = mexc.fetch_markets()
    kraken_markets = kraken.fetch_markets()

    mexc_pairs = {normalize_pair(m['symbol'], "mexc") for m in mexc_markets if ":USDT" in m['symbol']}
    kraken_pairs = {normalize_pair(m['symbol'], "kraken") for m in kraken_markets if ":USD" in m['symbol']}

    common_pairs = sorted(list(mexc_pairs & kraken_pairs))
    common_pairs_usdt = [to_usdt(p) for p in common_pairs]
    return common_pairs_usdt

def save_to_csv(pairs, filename="common_pairs_mexc_kraken.csv"):
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
