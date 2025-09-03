import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_file = "1-btc_processed_test.csv"
model_path = "2-models/hmm_model.pkl"

warnings.filterwarnings("ignore")

with open(f"data/data_processed_hmm/{model_path}", "rb") as f:
    saved = pickle.load(f)

model = saved['model']
means = saved['means']
stds = saved['stds']
signal_map = saved['signal_map']

df = pd.read_csv(f"data/data_processed_hmm/{test_file}")
features = df[['log_returns','volatility_5','momentum_5']].values
features = (features - means) / (stds + 1e-8)

hidden_states = model.predict(features)
df['market_regime'] = hidden_states
df['signal'] = df['market_regime'].map(signal_map)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

df['position'] = (df['signal'] == 1).astype(int)
df['strategy_returns'] = df['position'] * df['log_returns']
df['cum_strategy'] = np.exp(df['strategy_returns'].cumsum())
df['cum_bh'] = np.exp(df['log_returns'].cumsum())

plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], df['cum_bh'], label="Buy & Hold BTC")
plt.plot(df['timestamp'], df['cum_strategy'], label="HMM Strategy")
plt.title("BTC vs HMM Strategy (2022 - 2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
