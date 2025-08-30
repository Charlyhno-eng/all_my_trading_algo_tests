import pandas as pd
import numpy as np
import os
import vectorbt as vbt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

csv_path = os.path.join(os.path.dirname(__file__), '../..', 'data_ohlc', 'btc_usdt_1d_2020_2024.csv')
csv_path = os.path.abspath(csv_path)
df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

close = df['close']

short_window = 20
long_window = 200
sma_window = 30
short_up_level = 2.0
short_down_level = -2.0
long_min_threshold = 0

def rolling_linreg_tstat(series, window):
    x = np.arange(window)
    out = [np.nan] * (window - 1)
    for i in range(window, len(series) + 1):
        y = series.iloc[i-window:i].values
        X = np.vstack([x, np.ones(len(x))]).T
        beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - (beta * x + alpha)
        se = np.sqrt(np.sum(residuals**2) / (len(x) - 2)) / np.sqrt(np.sum((x - x.mean())**2))
        tstat = beta / se if se > 0 else 0
        out.append(tstat)
    return pd.Series(out, index=series.index)

short_tstat = rolling_linreg_tstat(close, short_window)
long_tstat = rolling_linreg_tstat(close, long_window)
long_tstat_sma = long_tstat.rolling(sma_window).mean()

signal = pd.Series(0, index=close.index)
signal[(short_tstat > short_up_level) & (long_tstat_sma > long_min_threshold)] = 1
signal[(short_tstat < short_down_level) & (long_tstat_sma > long_min_threshold)] = -1

entries = (signal == 1) & (signal.shift(1) != 1)
exits = (signal == -1) & (signal.shift(1) != -1)

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    init_cash=1000,
    fees=0.001,
    slippage=0.001,
    freq='1D'
)

print("Buy & Hold Final Equity:", round((1000 / close.iloc[0]) * close.iloc[-1], 2))
print(pf.stats())

try:
    equity = pf.value
except Exception:
    try:
        equity = pf.portfolio_value
    except Exception:
        cash = 1000.0
        units = 0.0
        fee = 0.001
        slippage = 0.001
        equity_list = []
        in_pos = False
        idxs = close.index
        for i in range(len(idxs)):
            price = close.iloc[i]
            if entries.iloc[i] and not in_pos:
                trade_cost = cash
                buy_price = price * (1 + slippage)
                fee_paid = trade_cost * fee
                units = (trade_cost - fee_paid) / buy_price
                cash = 0.0
                in_pos = True
            elif exits.iloc[i] and in_pos:
                sell_price = price * (1 - slippage)
                proceeds = units * sell_price
                fee_paid = proceeds * fee
                cash = proceeds - fee_paid
                units = 0.0
                in_pos = False
            equity_now = cash + units * price
            equity_list.append(equity_now)
        equity = pd.Series(equity_list, index=close.index)

buy_hold = 1000 * close / close.iloc[0]

plt.figure(figsize=(14,7))
plt.plot(buy_hold.index, buy_hold.values, label='Buy & Hold', linewidth=1.5)
plt.plot(equity.index, equity.values, label='Strategy Equity', linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Equity (€)')
plt.title('Buy & Hold vs Regression Linear Strategy (BTC 1D)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
