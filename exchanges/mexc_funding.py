import ccxt
from tabulate import tabulate
from datetime import datetime

PAIR = "BTC/USDT:USDT"

def get_funding_rate_mexc(symbol=PAIR):
    exchange = ccxt.mexc({'enableRateLimit': True})
    data = exchange.fetch_funding_rate(symbol)

    funding_rate_percent = float(data['fundingRate']) * 100
    interval = data['interval']

    if funding_rate_percent > 0:
        beneficiary = "Shorts"
    elif funding_rate_percent < 0:
        beneficiary = "Longs"
    else:
        beneficiary = "Aucun"

    funding_24h = funding_rate_percent * 3
    funding_14d = funding_rate_percent * 42
    funding_365d = funding_rate_percent * 1095

    # Amplitude possible
    min_fr = float(data['info'].get('minFundingRate', 0)) * 100
    max_fr = float(data['info'].get('maxFundingRate', 0)) * 100
    amplitude = f"[{min_fr:.4f}%, {max_fr:.4f}%]"

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
        "funding_365d": funding_365d,
        "beneficiary": beneficiary,
        "next_funding": next_funding_dt,
        "apr": funding_365d,
        "amplitude": amplitude
    }

if __name__ == "__main__":
    result = get_funding_rate_mexc()

    table = [[
        PAIR.replace(":USDT",""),
        result['amplitude'],
        f"{result['funding_rate_percent']:.6f}% ({result['interval']})",
        f"{result['funding_24h']:.6f}%",
        f"{result['funding_14d']:.6f}%",
        f"{result['funding_365d']:.6f}%",
        result['next_funding'],
        result['beneficiary'],
        f"{result['apr']:.2f}%",
    ]]

    headers = ["Paire", "Amplitude possible", "Funding actuel", "24h", "14j", "365j", "Prochain funding", "Bénéficiaires", "APR"]
    print(tabulate(table, headers=headers, tablefmt="grid"))
