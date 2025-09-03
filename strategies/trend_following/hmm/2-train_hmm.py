import os
import warnings
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
os.makedirs("data/data_processed_hmm/2-models", exist_ok=True)

train_file = "1-btc_processed_train.csv"
test_file = "1-btc_processed_test.csv"
model_path = "2-models/hmm_model.pkl"
train_signals_file = "2-btc_with_signals_train.csv"

df_train = pd.read_csv(f"data/data_processed_hmm/{train_file}")
features_train = ['log_returns', 'volatility_5', 'momentum_5']
X_train = df_train[features_train].values
means = np.nanmean(X_train, axis=0)
stds = np.nanstd(X_train, axis=0)
X_train_scaled = (X_train - means) / (stds + 1e-8)

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X_train_scaled)

hidden_states = model.predict(X_train_scaled)
df_train['market_regime'] = hidden_states

regime_means = df_train.groupby('market_regime')['log_returns'].mean()
long_regime = regime_means.idxmax()
short_regime = regime_means.idxmin()
signal_map = {regime: 1 if regime==long_regime else -1 if regime==short_regime else 0
              for regime in regime_means.index}
df_train['signal'] = df_train['market_regime'].map(signal_map)

with open(f"data/data_processed_hmm/{model_path}", "wb") as f:
    pickle.dump({'model': model, 'means': means, 'stds': stds, 'signal_map': signal_map}, f)

df_train.to_csv(f"data/data_processed_hmm/{train_signals_file}", index=False)
print("model trained, components:", model.n_components)


# --- NO REPAINT PREDICTION ON TEST DATA (2022-2025) ---
df_test = pd.read_csv(f"data/data_processed_hmm/{test_file}")
df_test = df_test.sort_values('timestamp').reset_index(drop=True)

features_test = ['log_returns', 'volatility_5', 'momentum_5']
df_test['market_regime'] = np.nan

for i in range(len(df_test)):
    X_window = df_test[features_test].iloc[:i+1].values
    X_window_scaled = (X_window - means) / (stds + 1e-8)
    pred = model.predict(X_window_scaled)
    df_test.loc[i, 'market_regime'] = pred[-1]

df_test['signal'] = df_test['market_regime'].map(signal_map)
df_test.to_csv("data/data_processed_hmm/2-btc_with_signals_test.csv", index=False)
print("test signals computed with no repaint, rows:", len(df_test))
