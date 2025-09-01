import ccxt
from tabulate import tabulate
from datetime import datetime, timedelta
import csv
import os

COMMON_PAIRS_FILE = "data_delta_neutral/common_pairs_mexc_kucoin.csv"

def get_next_funding_time():
    """Retourne la prochaine heure de funding (00:00, 08:00, 16:00 UTC)."""
    now = datetime.utcnow()
    funding_hours = [0, 8, 16]
    for hour in funding_hours:
        next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if next_time > now:
            return next_time.strftime('%Y-%m-%d %H:%M UTC')
    next_time = now.replace(hour=funding_hours[0], minute=0, second=0, microsecond=0) + timedelta(days=1)
    return next_time.strftime('%Y-%m-%d %H:%M UTC')

def get_funding_rate_mexc(symbol):
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        data = exchange.fetch_funding_rate(symbol + ":USDT")
    except Exception:
        return None
    fr = float(data['fundingRate']) * 100
    interval_hours = 8
    fundings_per_day = 24 / interval_hours
    return {
        "platform": "MEXC",
        "funding_rate_percent": fr,
        "funding_24h": fr * fundings_per_day,
        "beneficiary": "Shorts" if fr > 0 else "Longs" if fr < 0 else "Aucun",
        "next_funding": get_next_funding_time(),
        "interval_hours": interval_hours,
        "apy": fr * fundings_per_day * 365
    }

def get_funding_rate_kucoin(symbol):
    exchange = ccxt.kucoinfutures({'enableRateLimit': True})
    try:
        data = exchange.fetch_funding_rate(symbol + ":USDT")
    except Exception:
        return None
    fr = float(data['fundingRate']) * 100
    interval_hours = 8
    fundings_per_day = 24 / interval_hours
    return {
        "platform": "KuCoin",
        "funding_rate_percent": fr,
        "funding_24h": fr * fundings_per_day,
        "beneficiary": "Shorts" if fr > 0 else "Longs" if fr < 0 else "Aucun",
        "next_funding": get_next_funding_time(),
        "interval_hours": interval_hours,
        "apy": fr * fundings_per_day * 365
    }

def compute_delta_neutral(long_pos, short_pos):
    long_effective = -long_pos['funding_24h']
    short_effective = short_pos['funding_24h']
    net_daily = long_effective + short_effective
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
        kucoin = get_funding_rate_kucoin(pair)
        if not mexc or not kucoin:
            continue

        if mexc['funding_rate_percent'] < kucoin['funding_rate_percent']:
            long_pos, short_pos = mexc, kucoin
        else:
            long_pos, short_pos = kucoin, mexc

        net_daily, net_apy = compute_delta_neutral(long_pos, short_pos)

        # Filtrer uniquement les APY >= 20%
        if abs(net_apy) < 20:
            continue

        table = [
            ["Long", long_pos['platform'], f"{long_pos['funding_rate_percent']:.6f}%", long_pos['beneficiary'], long_pos['next_funding']],
            ["Short", short_pos['platform'], f"{short_pos['funding_rate_percent']:.6f}%", short_pos['beneficiary'], short_pos['next_funding']],
        ]
        headers = ["Position", "Plateforme", "Funding actuel", "Bénéficiaire", "Prochain funding"]

        print(f"\n--- {pair} ---")
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print(f"Rendement net approximatif delta neutre: {net_daily:.6f}% / jour, APY ≈ {net_apy:.2f}%")

        # Vérifier si les fundings vont dans le même sens
        if long_pos['beneficiary'] == short_pos['beneficiary']:
            print("⚠️ Attention : le funding est dans le même sens sur les deux plateformes. La stratégie delta neutre n'est probablement pas rentable.")
