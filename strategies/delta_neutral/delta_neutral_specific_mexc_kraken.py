import ccxt
from tabulate import tabulate
from datetime import datetime, timedelta

PAIR_USDT = "BTC/USDT"

def pair_for_kraken(pair_usdt):
    """Transforme la pair USDT en USD pour Kraken (enlève juste le 'T')."""
    return pair_usdt.replace("USDT", "USD")

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

def get_funding_rate_mexc(symbol=PAIR_USDT + ":USDT"):
    exchange = ccxt.mexc({'enableRateLimit': True})
    data = exchange.fetch_funding_rate(symbol)
    funding_rate_percent = float(data['fundingRate']) * 100
    interval = data.get('interval', '8h')
    try:
        interval_hours = int(interval.replace("h", "")) if "h" in interval else 8
    except:
        interval_hours = 8
    fundings_per_day = 24 / interval_hours
    funding_24h = funding_rate_percent * fundings_per_day
    funding_14d = funding_24h * 14
    funding_365d = funding_24h * 365
    beneficiary = "Shorts" if funding_rate_percent > 0 else "Longs" if funding_rate_percent < 0 else "Aucun"
    next_funding_dt = get_next_funding_time()
    return {
        "platform": "MEXC",
        "funding_rate_percent": funding_rate_percent,
        "funding_24h": funding_24h,
        "funding_14d": funding_14d,
        "beneficiary": beneficiary,
        "next_funding": next_funding_dt,
        "interval_hours": interval_hours,
        "apy": funding_365d,
    }

def get_funding_rate_kraken(symbol=None):
    if symbol is None:
        symbol = pair_for_kraken(PAIR_USDT) + ":USD"
    exchange = ccxt.krakenfutures({'enableRateLimit': True})
    data = exchange.fetch_funding_rate(symbol)
    funding_rate_percent = float(data['fundingRate']) * 100
    interval_hours = 8
    fundings_per_day = 24 / interval_hours
    funding_24h = funding_rate_percent * fundings_per_day
    funding_14d = funding_24h * 14
    funding_365d = funding_24h * 365
    beneficiary = "Shorts" if funding_rate_percent > 0 else "Longs" if funding_rate_percent < 0 else "Aucun"
    next_funding_dt = get_next_funding_time()
    return {
        "platform": "Kraken",
        "funding_rate_percent": funding_rate_percent,
        "funding_24h": funding_24h,
        "funding_14d": funding_14d,
        "beneficiary": beneficiary,
        "next_funding": next_funding_dt,
        "interval_hours": interval_hours,
        "apy": funding_365d,
    }

def compute_delta_neutral(long_pos, short_pos):
    long_effective = -long_pos['funding_24h']
    short_effective = short_pos['funding_24h']
    net_daily = long_effective + short_effective
    net_apy = net_daily * 365
    return net_daily, net_apy


if __name__ == "__main__":
    mexc = get_funding_rate_mexc()
    kraken = get_funding_rate_kraken()

    if mexc['funding_rate_percent'] < kraken['funding_rate_percent']:
        long_pos = mexc
        short_pos = kraken
    else:
        long_pos = kraken
        short_pos = mexc

    net_daily, net_apy = compute_delta_neutral(long_pos, short_pos)

    table = [
        ["Long", long_pos['platform'], f"{long_pos['funding_rate_percent']:.6f}%", long_pos['beneficiary'], long_pos['next_funding']],
        ["Short", short_pos['platform'], f"{short_pos['funding_rate_percent']:.6f}%", short_pos['beneficiary'], short_pos['next_funding']],
    ]

    headers = ["Position", "Plateforme", "Funding actuel", "Bénéficiaire", "Prochain funding"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print(f"\nRendement net approximatif delta neutre: {net_daily:.6f}% / jour, APY ≈ {net_apy:.2f}%")

    # Vérifier si les fundings vont dans le même sens
    if (mexc['beneficiary'] == kraken['beneficiary']):
        print("⚠️ Attention : le funding est dans le même sens sur les deux plateformes. La stratégie delta neutre n'est probablement pas rentable.")
