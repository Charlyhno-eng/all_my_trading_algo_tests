import os
import warnings
import pickle
from itertools import product
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

# Dossiers / fichiers
BASE_DIR = "data/data_processed_hmm"
os.makedirs(f"{BASE_DIR}/2-models", exist_ok=True)

TRAIN_FILE = "1-btc_processed_train.csv"   # <= doit couvrir jusqu'à ~2021-01-01
TEST_FILE  = "1-btc_processed_test.csv"    # >= 2022
MODEL_PATH = "2-models/hmm_model.pkl"

OUT_TRAIN_SIGNALS = "2-btc_with_signals_train.csv"
OUT_TEST_SIGNALS  = "2-btc_with_signals_test.csv"

# Grille d'hyperparamètres à tester (sobre mais efficace)
N_COMPONENTS_CHOICES = [2, 3, 4]
MOMENTUM_DELAYS      = [5, 10, 20]
N_MOMENTUMS_CHOICES  = [1, 2, 3]   # nombre de momentums (1 => momentum_0 seul; 3 => momentum_0..2)

# Métriques de sélection (on maximise)
def score_curve(df, use_short=False):
    """
    df: dataframe avec colonnes ['log_returns','signal'] trié par temps, sans NaN.
    use_short:
        - False: stratégie long/flat (position = 1 si signal==1, sinon 0)
        - True : stratégie long/short (position = 1, 0 ou -1 selon signal)
    Retourne (final_cum, sharpe_annuelle)
    """
    pos = df['signal'].copy()
    if not use_short:
        pos = (pos == 1).astype(float)

    strat_rets = pos * df['log_returns']
    # Cumul
    cum = float(np.exp(strat_rets.cumsum().iloc[-1]))
    # Sharpe annualisée (en supposant daily data)
    mu = strat_rets.mean()
    sd = strat_rets.std(ddof=0)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252) if sd > 0 else 0.0
    return cum, sharpe

def build_features(df, mom_delay, n_moms):
    """
    Ajoute: log_returns + n_moms colonnes momentum_k (k=0..n_moms-1)
    momentum_k = log(close / close.shift(mom_delay + k))
    """
    df = df.copy()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    for k in range(n_moms):
        df[f'momentum_{k}'] = np.log(df['close'] / df['close'].shift(mom_delay + k))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    features = ['log_returns'] + [f'momentum_{k}' for k in range(n_moms)]
    return df, features

def fit_hmm(X, n_components):
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=10000,
        tol=1e-4,
        algorithm='map',
        random_state=42
    )
    model.fit(X)
    return model

def assign_signals_by_regime(df):
    # Long = régime à meilleure moyenne de log_returns; Short = pire; Autre = neutre
    regime_means = df.groupby('market_regime')['log_returns'].mean()
    long_regime  = regime_means.idxmax()
    short_regime = regime_means.idxmin()
    signal_map = {reg: (1 if reg == long_regime else (-1 if reg == short_regime else 0))
                  for reg in regime_means.index}
    df['signal'] = df['market_regime'].map(signal_map)
    return df, signal_map

def predict_no_repaint(model, X_scaled):
    """
    X_scaled: np.array T x F.
    Retourne un vecteur (T,) des régimes prédits sans repaint (walk-forward).
    """
    preds = np.empty(len(X_scaled), dtype=int)
    for i in range(len(X_scaled)):
        seq = X_scaled[:i+1]
        y = model.predict(seq)
        preds[i] = y[-1]
    return preds

# --- 1) Charger le train / test
df_train_raw = pd.read_csv(f"{BASE_DIR}/{TRAIN_FILE}")
df_test_raw  = pd.read_csv(f"{BASE_DIR}/{TEST_FILE}")

# S'assurer de l'ordre temporel
for df in (df_train_raw, df_test_raw):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

# --- 2) Split interne pour validation walk-forward sur le train
#     On prend ~80% pour fit, 20% pour validation no-repaint.
n_train = len(df_train_raw)
cut = int(n_train * 0.8)
df_fit_raw = df_train_raw.iloc[:cut].copy()
df_val_raw = df_train_raw.iloc[cut:].copy()

best = {
    'score': -np.inf,
    'params': None,
    'signal_map': None,
    'means': None,
    'stds': None,
    'n_features': None
}

# --- 3) Recherche d’hyperparamètres
for n_comp, mom_delay, n_moms in product(N_COMPONENTS_CHOICES, MOMENTUM_DELAYS, N_MOMENTUMS_CHOICES):
    # a) Features sur fit + val
    df_fit, feat = build_features(df_fit_raw, mom_delay, n_moms)
    df_val, _    = build_features(df_val_raw, mom_delay, n_moms)

    if len(df_fit) < 50 or len(df_val) < 30:
        continue

    # b) Standardisation selon FIT uniquement
    X_fit = df_fit[feat].values
    means = np.nanmean(X_fit, axis=0)
    stds  = np.nanstd(X_fit, axis=0)
    X_fit_scaled = (X_fit - means) / (stds + 1e-8)

    # c) Entraîner HMM sur FIT
    try:
        model = fit_hmm(X_fit_scaled, n_comp)
    except Exception:
        continue

    # d) Régimes FIT + signals (pour établir le signal_map)
    states_fit = model.predict(X_fit_scaled)
    df_fit_tmp = df_fit.copy()
    df_fit_tmp['market_regime'] = states_fit
    df_fit_tmp, signal_map = assign_signals_by_regime(df_fit_tmp)

    # e) Prédictions NO-REPAINT sur VAL
    X_val = df_val[feat].values
    X_val_scaled = (X_val - means) / (stds + 1e-8)
    states_val = predict_no_repaint(model, X_val_scaled)
    df_val_tmp = df_val.copy()
    df_val_tmp['market_regime'] = states_val
    df_val_tmp['signal'] = pd.Series(states_val).map(signal_map).values

    # f) Score validation (priorité au rendement, puis Sharpe)
    cum_lf, shp_lf = score_curve(df_val_tmp, use_short=False)  # long/flat
    # On combine légèrement Sharpe pour départager
    score = cum_lf + 0.05 * shp_lf

    if score > best['score']:
        best.update({
            'score': score,
            'params': {'n_components': n_comp, 'momentum_delay': mom_delay, 'n_momentums': n_moms},
            'signal_map': signal_map,
            'means': means,
            'stds': stds,
            'n_features': len(feat)
        })

print("Meilleurs hyperparamètres trouvés:", best['params'], "score:", round(best['score'], 4))

# --- 4) Réentraîner sur TOUT le train avec les meilleurs hyperparamètres
mom_delay = best['params']['momentum_delay']
n_moms    = best['params']['n_momentums']
n_comp    = best['params']['n_components']

df_train, features = build_features(df_train_raw, mom_delay, n_moms)
X_train = df_train[features].values
means = np.nanmean(X_train, axis=0)
stds  = np.nanstd(X_train, axis=0)
X_train_scaled = (X_train - means) / (stds + 1e-8)

model = fit_hmm(X_train_scaled, n_comp)

# Régimes + signals sur le train (info ex-post acceptable pour cartographier les régimes)
states_train = model.predict(X_train_scaled)
df_train['market_regime'] = states_train
df_train, signal_map = assign_signals_by_regime(df_train)

# Sauvegarde du modèle + scaler + signal_map + hyperparamètres
to_save = {
    'model': model,
    'means': means,
    'stds': stds,
    'signal_map': signal_map,
    'features': features,
    'params': best['params']
}
with open(f"{BASE_DIR}/{MODEL_PATH}", "wb") as f:
    pickle.dump(to_save, f)

# Sauvegarde train enrichi
df_train_out = df_train_raw.merge(
    df_train[['timestamp', 'market_regime', 'signal']],
    on='timestamp', how='left'
)
df_train_out.to_csv(f"{BASE_DIR}/{OUT_TRAIN_SIGNALS}", index=False)

print("Modèle entraîné. n_components:", n_comp, "| delay:", mom_delay, "| n_momentums:", n_moms)

# --- 5) NO-REPAINT sur le TEST (2022-2025)
df_test, _ = build_features(df_test_raw, mom_delay, n_moms)
X_test = df_test[features].values
X_test_scaled = (X_test - means) / (stds + 1e-8)

states_test = predict_no_repaint(model, X_test_scaled)
df_test['market_regime'] = states_test
df_test['signal'] = pd.Series(states_test).map(signal_map).values

# Merge pour conserver toutes les colonnes originales
df_test_out = df_test_raw.merge(
    df_test[['timestamp', 'market_regime', 'signal']],
    on='timestamp', how='left'
)
df_test_out.to_csv(f"{BASE_DIR}/{OUT_TEST_SIGNALS}", index=False)

print("Test NO-REPAINT généré. Lignes:", len(df_test_out))
