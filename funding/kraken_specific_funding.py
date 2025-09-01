import ccxt
from tabulate import tabulate
from datetime import datetime, timedelta

PAIR = "BTC/USD:USD"

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

def get_funding_rate_kraken(symbol=PAIR):
    exchange = ccxt.krakenfutures({'enableRateLimit': True})
    data = exchange.fetch_funding_rate(symbol)

    fr = float(data['fundingRate'])
    funding_rate_percent = fr * 100
    interval = '8h'
    interval_hours = 8
    fundings_per_day = 24 / interval_hours

    funding_24h = funding_rate_percent * fundings_per_day
    funding_14d = funding_24h * 14
    funding_365d = funding_24h * 365

    beneficiary = "Shorts" if funding_rate_percent > 0 else "Longs" if funding_rate_percent < 0 else "Aucun"
    next_funding_dt = get_next_funding_time()

    return {
        "interval": interval,
        "funding_rate_percent": funding_rate_percent,
        "funding_24h": funding_24h,
        "funding_14d": funding_14d,
        "beneficiary": beneficiary,
        "next_funding": next_funding_dt,
        "apr": funding_365d,
    }

if __name__ == "__main__":
    result = get_funding_rate_kraken()
    table = [[
        PAIR.replace(":USD", ""),
        f"{result['funding_rate_percent']:.6f}% ({result['interval']})",
        f"{result['funding_24h']:.6f}%",
        f"{result['funding_14d']:.6f}%",
        result['next_funding'],
        result['beneficiary'],
        f"{result['apr']:.2f}%",
    ]]
    headers = ["Paire", "Funding actuel", "24h", "14j", "Prochain funding", "Bénéficiaires", "APR"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
