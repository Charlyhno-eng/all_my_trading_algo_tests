import ccxt
from tabulate import tabulate
import csv
import os

COMMON_PAIRS_FILE = "data/data_delta_neutral/common_pairs_mexc_binance.csv"

def get_funding_rate_mexc(symbol):
    """Récupère le funding rate pour une paire perp sur MEXC."""
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        data = exchange.fetch_funding_rate(symbol + ":USDT")
    except Exception:
        return None
    fr = float(data['fundingRate']) * 100
    interval_hours = 8
    fundings_per_day = 24 / interval_hours
    return {
        "platform": "MEXC (Perp)",
        "funding_rate_percent": fr,
        "funding_24h": fr * fundings_per_day,
        "beneficiary": "Shorts" if fr > 0 else "Longs" if fr < 0 else "Aucun",
        "apy": fr * fundings_per_day * 365
    }

def compute_delta_neutral(mexc_pos):
    """Comme Binance est spot, le rendement net vient uniquement du funding MEXC."""
    net_daily = mexc_pos['funding_24h']
    net_apy = net_daily * 365
    return net_daily, net_apy

def read_common_pairs(filename):
    pairs = []
    if os.path.exists(filename):
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pairs.append(row['Pair'])
    return pairs

if __name__ == "__main__":
    pairs = read_common_pairs(COMMON_PAIRS_FILE)

    for pair in pairs:
        mexc = get_funding_rate_mexc(pair)
        if not mexc:
            continue

        net_daily, net_apy = compute_delta_neutral(mexc)

        # Filtrer uniquement APY positifs et >= 15%
        if net_apy < 20:
            continue

        table = [
            ["Long", "Binance (Spot)", "0.000000%", "Aucun"],
            ["Short", mexc['platform'], f"{mexc['funding_rate_percent']:.6f}%", mexc['beneficiary']],
        ]
        headers = ["Position", "Plateforme", "Funding actuel", "Bénéficiaire"]

        print(f"\n--- {pair} ---")
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print(f"Rendement net approximatif delta neutre: {net_daily:.6f}% / jour, APY ≈ {net_apy:.2f}%")
