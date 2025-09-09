import os
import warnings
import pickle
from itertools import product
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

# --- Paramètres ---
ticker = "BTC_USDT_4h"

# Fichiers cohérents avec le script de téléchargement
train_start_date = "2022-01-01"
train_end_date   = "2023-12-31"
test_start_date  = "2024-01-01"
test_end_date    = "2025-08-31"

csv_train_path = f"data/data_ohlc/{ticker}_mexc_future_{train_start_date}_{train_end_date}.csv"
csv_test_path  = f"data/data_ohlc/{ticker}_mexc_future_{test_start_date}_{test_end_date}.csv"

# Dossier de sortie
BASE_DIR = f"data/data_processed_hmm/{ticker}"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

MODEL_PATH = os.path.join("models", "hmm_model.pkl")
OUT_TRAIN_SIGNALS = f"{ticker}_with_signals_train.csv"
OUT_TEST_SIGNALS = f"{ticker}_with_signals_test.csv"

# Hyperparamètres
N_COMPONENTS_CHOICES = [2, 3, 4]
MOMENTUM_DELAYS = [5, 10, 20]
N_MOMENTUMS_CHOICES = [1, 2, 3]

# --- Fonctions utilitaires ---
def load_csv_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    return df

def compute_basic_features(df):
    df = df.copy()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_5'] = df['log_returns'].rolling(5).std()
    df['momentum_5'] = df['log_returns'].rolling(5).mean()
    df = df.dropna(subset=['log_returns', 'volatility_5', 'momentum_5'])
    return df

def build_hmm_features(df, mom_delay, n_moms):
    df = df.copy()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features = ['log_returns']
    for k in range(n_moms):
        df[f'momentum_{k}'] = np.log(df['close'] / df['close'].shift(mom_delay + k))
        features.append(f'momentum_{k}')
    df = df.dropna(subset=features)
    return df, features

def fit_hmm(X, n_components):
    model = GaussianHMM(n_components=n_components, covariance_type="full",
                        n_iter=10000, tol=1e-4, algorithm='map', random_state=42)
    model.fit(X)
    return model

def assign_signals_by_regime(df):
    regime_means = df.groupby('market_regime')['log_returns'].mean()
    long_regime = regime_means.idxmax()
    short_regime = regime_means.idxmin()
    signal_map = {reg: 1 if reg==long_regime else -1 if reg==short_regime else 0 for reg in regime_means.index}
    df['signal'] = df['market_regime'].map(signal_map)
    return df, signal_map

def predict_no_repaint(model, X_scaled):
    preds = np.empty(len(X_scaled), dtype=int)
    for i in range(len(X_scaled)):
        preds[i] = model.predict(X_scaled[:i+1])[-1]
    return preds

def score_curve(df, use_short=False):
    pos = df['signal'].copy()
    if not use_short:
        pos = (pos == 1).astype(float)
    strat_rets = pos * df['log_returns']
    cum = float(np.exp(strat_rets.cumsum().iloc[-1]))
    mu, sd = strat_rets.mean(), strat_rets.std(ddof=0)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252*24) if sd > 0 else 0.0  # hourly data
    return cum, sharpe


# --- 1) Charger données train/test ---
df_train_raw = compute_basic_features(load_csv_data(csv_train_path))
df_test_raw  = compute_basic_features(load_csv_data(csv_test_path))

# Split interne sur train pour validation hyperparamètres
cut_val = int(len(df_train_raw) * 0.8)
df_fit_raw = df_train_raw.iloc[:cut_val].copy()
df_val_raw = df_train_raw.iloc[cut_val:].copy()

best = {'score': -np.inf, 'params': None, 'signal_map': None, 'means': None, 'stds': None, 'features': None}

for n_comp, mom_delay, n_moms in product(N_COMPONENTS_CHOICES, MOMENTUM_DELAYS, N_MOMENTUMS_CHOICES):
    df_fit, feat = build_hmm_features(df_fit_raw, mom_delay, n_moms)
    df_val, _ = build_hmm_features(df_val_raw, mom_delay, n_moms)
    if len(df_fit) < 50 or len(df_val) < 30:
        continue

    X_fit = df_fit[feat].values
    means = np.nanmean(X_fit, axis=0)
    stds = np.nanstd(X_fit, axis=0)
    X_fit_scaled = (X_fit - means) / (stds + 1e-8)

    try:
        model = fit_hmm(X_fit_scaled, n_comp)
    except:
        continue

    df_fit_tmp = df_fit.copy()
    df_fit_tmp['market_regime'] = model.predict(X_fit_scaled)
    df_fit_tmp, signal_map = assign_signals_by_regime(df_fit_tmp)

    X_val = df_val[feat].values
    X_val_scaled = (X_val - means) / (stds + 1e-8)
    states_val = predict_no_repaint(model, X_val_scaled)
    df_val_tmp = df_val.copy()
    df_val_tmp['market_regime'] = states_val
    df_val_tmp['signal'] = pd.Series(states_val).map(signal_map).values

    cum_lf, shp_lf = score_curve(df_val_tmp)
    score = cum_lf + 0.05 * shp_lf
    if score > best['score']:
        best.update({'score': score, 'params': {'n_components': n_comp, 'momentum_delay': mom_delay, 'n_momentums': n_moms},
                     'signal_map': signal_map, 'means': means, 'stds': stds, 'features': feat})

print("Meilleurs hyperparamètres:", best['params'], "score:", round(best['score'],4))

# --- 2) Réentraîner HMM sur tout le train ---
params = best['params']
df_train_hmm, features = build_hmm_features(df_train_raw, params['momentum_delay'], params['n_momentums'])
X_train = df_train_hmm[features].values
means = np.nanmean(X_train, axis=0)
stds = np.nanstd(X_train, axis=0)
X_train_scaled = (X_train - means) / (stds + 1e-8)
model = fit_hmm(X_train_scaled, params['n_components'])

states_train = model.predict(X_train_scaled)
df_train_hmm['market_regime'] = states_train
df_train_hmm, signal_map = assign_signals_by_regime(df_train_hmm)

# Sauvegarde du modèle
to_save = {'model': model, 'means': means, 'stds': stds,
           'signal_map': signal_map, 'features': features, 'params': params}
with open(os.path.join(BASE_DIR, MODEL_PATH), "wb") as f:
    pickle.dump(to_save, f)

# --- 3) Générer CSV train enrichi ---
df_train_out = df_train_raw.merge(df_train_hmm[['timestamp', 'market_regime', 'signal']], on='timestamp', how='left')
df_train_out.to_csv(os.path.join(BASE_DIR, OUT_TRAIN_SIGNALS), index=False)

# --- 4) Générer CSV test enrichi ---
df_test_hmm, _ = build_hmm_features(df_test_raw, params['momentum_delay'], params['n_momentums'])
X_test = df_test_hmm[features].values
X_test_scaled = (X_test - means) / (stds + 1e-8)
states_test = predict_no_repaint(model, X_test_scaled)
df_test_hmm['market_regime'] = states_test
df_test_hmm['signal'] = pd.Series(states_test).map(signal_map).values

df_test_out = df_test_raw.merge(df_test_hmm[['timestamp', 'market_regime', 'signal']], on='timestamp', how='left')
df_test_out.to_csv(os.path.join(BASE_DIR, OUT_TEST_SIGNALS), index=False)

print("Modèle entraîné et fichiers train/test enrichis générés.")
