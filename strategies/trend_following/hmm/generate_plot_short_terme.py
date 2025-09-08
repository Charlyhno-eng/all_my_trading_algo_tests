import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# --- Paramètres globaux ---
ticker = "BTC_USDT_4h"  # ou ton ticker actuel
BASE_DIR = f"data/data_processed_hmm/{ticker}"
TEST_FILE_SIGNALS = f"{ticker}_with_signals_test.csv"
MODEL_PATH = "models/hmm_model.pkl"

TITLE = f"HMM Strategy vs {ticker} Buy & Hold"

# --- Frais Binance réalistes ---
FEES_MAKER = 0.1
FEES_TAKER = 0.2

# --- Charger modèle ---
with open(f"{BASE_DIR}/{MODEL_PATH}", "rb") as f:
    saved = pickle.load(f)

params = saved.get('params', {})
model = saved['model']
means = saved['means']
stds = saved['stds']
features = saved['features']
signal_map = saved['signal_map']

# --- Charger CSV enrichi ---
df = pd.read_csv(f"{BASE_DIR}/{TEST_FILE_SIGNALS}")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# --- Construire features test ---
for k in range(params['n_momentums']):
    df[f'momentum_{k}'] = np.log(df['close'] / df['close'].shift(params['momentum_delay'] + k))
feat_cols = ['log_returns'] + [f'momentum_{k}' for k in range(params['n_momentums'])]
df = df.dropna(subset=feat_cols)

# --- Scaling ---
X_test = df[feat_cols].values
X_test_scaled = (X_test - means) / (stds + 1e-8)

# --- Predict HMM ---
states_test = model.predict(X_test_scaled)  # modèle entraîné
df['market_regime'] = states_test
df['signal'] = df['market_regime'].map(signal_map)

# --- Position (long only) ---
df['position'] = (df['signal'] == 1).astype(float)

# --- Rendements ---
df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

# --- Frais de transaction ---
df['pos_change'] = df['position'].diff().fillna(0)
df['fees'] = 0.0
df.loc[df['pos_change'] > 0, 'fees'] = FEES_MAKER / 100
df.loc[df['pos_change'] < 0, 'fees'] = FEES_TAKER / 100

# --- Rendement stratégie ---
df['strategy_log_ret'] = df['position'] * df['log_returns'] - df['fees']
df['strategy_cumret'] = np.exp(df['strategy_log_ret'].cumsum())
df['bh_cumret'] = np.exp(df['log_returns'].cumsum())

# --- Drawdown ---
def compute_drawdown(cumret_series):
    rolling_max = cumret_series.cummax()
    return (cumret_series - rolling_max) / rolling_max

df['strategy_dd'] = compute_drawdown(df['strategy_cumret'])
df['bh_dd'] = compute_drawdown(df['bh_cumret'])

# --- Statistiques ---
def compute_stats(strategy_ret, benchmark_ret):
    mu_s, sd_s = strategy_ret.mean(), strategy_ret.std(ddof=0)
    sharpe = (mu_s / (sd_s + 1e-12)) * np.sqrt(252) if sd_s > 0 else 0.0
    cum_return = np.exp(strategy_ret.cumsum().iloc[-1])
    max_dd = compute_drawdown(np.exp(strategy_ret.cumsum())).min() * 100

    X = benchmark_ret.values.reshape(-1, 1)
    y = strategy_ret.values
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    alpha = reg.intercept_ * 252

    return {
        'cumulative': cum_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'alpha': alpha,
        'beta': beta
    }

stats_strategy = compute_stats(df['strategy_log_ret'], df['log_returns'])
stats_bh = compute_stats(df['log_returns'], df['log_returns'])

# --- Comptage des transactions ---
nb_achats = (df['pos_change'] == 1).sum()
nb_ventes = (df['pos_change'] == -1).sum()

# --- Affichage des stats ---
print("Hyperparamètres retenus:", params)
print("\n--- Stratégie HMM ---")
for k, v in stats_strategy.items():
    print(f"{k}: {v:.4f}")
print(f"Nombre d'achats effectués: {nb_achats}")
print(f"Nombre de ventes effectuées: {nb_ventes}")

print(f"\n--- Buy & Hold {ticker} ---")
for k, v in stats_bh.items():
    print(f"{k}: {v:.4f}")

# --- Plot cumulatif return + drawdown + positions ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(df['timestamp'], df['bh_cumret'], label=f"Buy & Hold {ticker}", color='blue')
axes[0].plot(df['timestamp'], df['strategy_cumret'], label="HMM Strategy", color='orange')
axes[0].set_ylabel("Cumulative Return")
axes[0].set_title(f"{TITLE}\n"
                  f"n_components={params.get('n_components')} | "
                  f"delay={params.get('momentum_delay')} | "
                  f"n_moms={params.get('n_momentums')} | "
                  f"fees maker={FEES_MAKER}% | fees taker={FEES_TAKER}%")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df['timestamp'], df['strategy_dd'] * 100, label="HMM Strategy DD", color='orange')
axes[1].plot(df['timestamp'], df['bh_dd'] * 100, label=f"{ticker} Buy & Hold DD", color='blue')
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend()
axes[1].grid(True)

axes[2].step(df['timestamp'], df['position'], label="Position (1=Long,0=Flat)", color='green', where='post')
axes[2].set_ylabel("Position")
axes[2].set_xlabel("Date")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
