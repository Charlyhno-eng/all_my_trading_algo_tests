import ccxt
from tabulate import tabulate
from datetime import datetime

def get_all_funding_rates():
    exchange = ccxt.mexc({'enableRateLimit': True})
    markets = exchange.load_markets()

    # On filtre uniquement les contrats perpétuels USDT
    perp_symbols = [s for s in markets if ":USDT" in s and markets[s].get("swap", False)]

    results = []
    for symbol in perp_symbols:
        try:
            data = exchange.fetch_funding_rate(symbol)
            fr_percent = float(data['fundingRate']) * 100
            interval = data['interval']

            # conversion interval -> heures
            try:
                interval_hours = int(interval.replace("h", "")) if "h" in interval else 8
            except:
                interval_hours = 8
            fundings_per_day = 24 / interval_hours

            funding_24h = fr_percent * fundings_per_day
            funding_14d = funding_24h * 14
            funding_365d = funding_24h * 365

            if fr_percent > 0:
                beneficiary = "Shorts"
            elif fr_percent < 0:
                beneficiary = "Longs"
            else:
                beneficiary = "Aucun"

            # prochain funding
            next_funding_ts = data.get('fundingDatetime')
            if next_funding_ts:
                try:
                    next_funding_dt = datetime.fromisoformat(next_funding_ts.replace('Z','')).strftime('%Y-%m-%d %H:%M UTC')
                except:
                    next_funding_dt = next_funding_ts
            else:
                next_funding_dt = 'N/A'

            results.append({
                "pair": symbol.replace(":USDT", ""),
                "funding_rate_percent": fr_percent,
                "interval": interval,
                "funding_24h": funding_24h,
                "funding_14d": funding_14d,
                "beneficiary": beneficiary,
                "next_funding": next_funding_dt,
                "apr": funding_365d,
            })
        except Exception:
            continue

    return results

if __name__ == "__main__":
    results = get_all_funding_rates()

    top_10 = sorted(results, key=lambda x: x['apr'], reverse=True)[:10]
    bottom_10 = sorted(results, key=lambda x: x['apr'])[:10]

    def format_table(data, title):
        table = [[
            r['pair'],
            f"{r['funding_rate_percent']:.6f}% ({r['interval']})",
            f"{r['funding_24h']:.6f}%",
            f"{r['funding_14d']:.6f}%",
            r['next_funding'],
            r['beneficiary'],
            f"{r['apr']:.2f}%",
        ] for r in data]

        headers = ["Paire", "Funding actuel", "24h", "14j", "Prochain funding", "Bénéficiaires", "APR"]
        print(f"\n=== {title} ===")
        print(tabulate(table, headers=headers, tablefmt="grid"))

    format_table(top_10, "Top 10 APR (plus grand au plus petit)")
    format_table(bottom_10, "Bottom 10 APR (plus petit au plus grand)")
