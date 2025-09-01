import ccxt
from tabulate import tabulate
from datetime import datetime

PAIR = "BTC/USDT:USDT"

def get_funding_rate_mexc(symbol=PAIR):
    exchange = ccxt.mexc({'enableRateLimit': True})
    data = exchange.fetch_funding_rate(symbol)

    funding_rate_percent = float(data['fundingRate']) * 100
    interval = data['interval']

    # conversion interval -> heures
    try:
        interval_hours = int(interval.replace("h", "")) if "h" in interval else 8
    except:
        interval_hours = 8
    fundings_per_day = 24 / interval_hours

    funding_24h = funding_rate_percent * fundings_per_day
    funding_14d = funding_24h * 14
    funding_365d = funding_24h * 365

    if funding_rate_percent > 0:
        beneficiary = "Shorts"
    elif funding_rate_percent < 0:
        beneficiary = "Longs"
    else:
        beneficiary = "Aucun"

    # Prochain funding datetime
    next_funding_ts = data.get('fundingDatetime')
    if next_funding_ts:
        try:
            next_funding_dt = datetime.fromisoformat(next_funding_ts.replace('Z','')).strftime('%Y-%m-%d %H:%M UTC')
        except:
            next_funding_dt = next_funding_ts
    else:
        next_funding_dt = 'N/A'

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
    result = get_funding_rate_mexc()

    table = [[
        PAIR.replace(":USDT",""),
        f"{result['funding_rate_percent']:.6f}% ({result['interval']})",
        f"{result['funding_24h']:.6f}%",
        f"{result['funding_14d']:.6f}%",
        result['next_funding'],
        result['beneficiary'],
        f"{result['apr']:.2f}%",
    ]]

    headers = ["Paire", "Funding actuel", "24h", "14j", "Prochain funding", "Bénéficiaires", "APR"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
