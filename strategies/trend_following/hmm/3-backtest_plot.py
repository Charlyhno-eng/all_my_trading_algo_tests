import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = "data/data_processed_hmm"
TEST_FILE_SIGNALS = "2-btc_with_signals_test.csv"     # => déjà no-repaint
MODEL_PATH = "2-models/hmm_model.pkl"

# Paramètres backtest
USE_SHORT_CURVE = False   # False = long/flat (comparaison la plus “juste” vs B&H)
FEES_BPS = 0.0            # frais aller-retour en proportion (ex: 0.001 = 10 bps)
TITLE = "BTC vs HMM (no-repaint, 2022–2025)"

with open(f"{BASE_DIR}/{MODEL_PATH}", "rb") as f:
    saved = pickle.load(f)

signal_map = saved['signal_map']
params = saved.get('params', {})

df = pd.read_csv(f"{BASE_DIR}/{TEST_FILE_SIGNALS}")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Sécurité: si 'signal' n’est pas rempli (NaN aux débuts), remplir à 0
df['signal'] = df['signal'].fillna(0)

# Position
if USE_SHORT_CURVE:
    df['position'] = df['signal'].clip(-1, 1).astype(float)   # -1/0/1
else:
    df['position'] = (df['signal'] == 1).astype(float)        # long/flat

# Rendements stratégiques avec coûts de transaction (si activés)
df['log_returns'] = df['log_returns'].astype(float)

# Variation de position pour compter les trades
pos_change = df['position'].diff().abs().fillna(0.0)
# Coût simple “per trade”: on applique un coût proportionnel FEES_BPS au retour du jour d’entrée/sortie.
# En log-returns, approx: retrancher FEES_BPS * pos_change
fees_series = FEES_BPS * pos_change

df['strategy_log_ret'] = df['position'] * df['log_returns'] - fees_series

# Courbes cumulées
df['cum_strategy'] = np.exp(df['strategy_log_ret'].cumsum())
df['cum_bh'] = np.exp(df['log_returns'].cumsum())

# Stats rapides
def stats(series):
    mu = series.mean()
    sd = series.std(ddof=0)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252) if sd > 0 else 0.0
    return float(np.exp(series.cumsum().iloc[-1])), float(sharpe)

cum_s, shp_s = stats(df['strategy_log_ret'])
cum_b, shp_b = stats(df['log_returns'])

print("Hyperparamètres retenus:", params)
print(f"Performance stratégie: x{cum_s:.2f} | Sharpe {shp_s:.2f}")
print(f"Buy & Hold BTC      : x{cum_b:.2f} | Sharpe {shp_b:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['cum_bh'], label="Buy & Hold BTC")
label_strat = "HMM long/flat" if not USE_SHORT_CURVE else "HMM long/short"
plt.plot(df['timestamp'], df['cum_strategy'], label=label_strat)
plt.title(f"{TITLE}\n"
          f"n_components={params.get('n_components')} | "
          f"delay={params.get('momentum_delay')} | "
          f"n_moms={params.get('n_momentums')} | "
          f"fees={FEES_BPS:.3f}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
