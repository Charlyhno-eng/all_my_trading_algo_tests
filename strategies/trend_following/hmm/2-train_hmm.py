import os
import warnings
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

train_file = "1-btc_processed_train.csv"

warnings.filterwarnings("ignore")
os.makedirs("data/data_processed_hmm/2-models", exist_ok=True)

df = pd.read_csv(f"data/data_processed_hmm/{train_file}")
features = df[['log_returns','volatility_5','momentum_5']].values
means = np.nanmean(features, axis=0)
stds = np.nanstd(features, axis=0)
features = (features - means) / (stds + 1e-8)

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(features)

hidden_states = model.predict(features)
df['market_regime'] = hidden_states

regime_means = df.groupby('market_regime')['log_returns'].mean()
long_regime = regime_means.idxmax()
short_regime = regime_means.idxmin()

signal_map = {}
for regime in regime_means.index:
    if regime == long_regime:
        signal_map[regime] = 1
    elif regime == short_regime:
        signal_map[regime] = -1
    else:
        signal_map[regime] = 0

df['signal'] = df['market_regime'].map(signal_map)

with open("data/data_processed_hmm/2-models/hmm_model.pkl", "wb") as f:
    pickle.dump({'model': model, 'means': means, 'stds': stds, 'signal_map': signal_map}, f)

df.to_csv("data/data_processed_hmm/2-btc_with_signals_train.csv", index=False)
print("model trained, components:", model.n_components)
