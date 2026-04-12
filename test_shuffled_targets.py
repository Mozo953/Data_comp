"""
Test de sanité : entraîner le modèle avec targets shufflées
Si le modèle fonctionne correctement, il devrait avoir une RMSE très mauvaise
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from pathlib import Path
from src.odor_competition.metrics import competition_rmse

# Configuration
DATA_DIR = "src/odor_competition/data"
PARAMS_JSON = "artifacts_extratrees_corr_optuna/optuna_objectif_0.04_EX_Type_melchior_20260407T193613Z.json"
CV_FOLDS = 3

print("=" * 70)
print("TEST DE SANITÉ : TARGETS SHUFFLÉS")
print("=" * 70)

# Load params from JSON
with open(PARAMS_JSON, 'r') as f:
    config = json.load(f)

best_params = config['optuna']['best_params']
print(f"\n✓ Paramètres chargés depuis: {PARAMS_JSON}")
print(f"  Params: {best_params}")

# Load data
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")

print(f"\n✓ Données chargées")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

# Get first actual target (skip ID) - prend d01 (colonne 1)
first_target = y_train.columns[1]  # Skip ID column
y_original = y_train[first_target].values
print(f"\n✓ Target utilisée: {first_target}")
print(f"  RMSE baseline (original) attendue: ~0.0416")

# SHUFFLE les targets
np.random.seed(42)
y_shuffled = y_original.copy()
np.random.shuffle(y_shuffled)

print(f"\n✓ Targets shufflés (aléatoires)")
print(f"  Min original: {y_original.min():.6f}, Max: {y_original.max():.6f}")
print(f"  Min shuffled: {y_shuffled.min():.6f}, Max: {y_shuffled.max():.6f}")

# CV avec targets shufflés
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
fold_rmses = []

print(f"\n{'='*70}")
print("CV avec TARGETS SHUFFLÉS:")
print(f"{'='*70}")

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_shuffled[train_idx]
    y_fold_val = y_shuffled[val_idx]
    
    # Entraîner
    model = ExtraTreesRegressor(**best_params)
    model.fit(X_fold_train, y_fold_train)
    
    # Prédire
    y_pred = model.predict(X_fold_val)
    fold_rmse = competition_rmse(y_fold_val, y_pred)
    fold_rmses.append(fold_rmse)
    
    print(f"Fold {fold_idx}: RMSE = {fold_rmse:.6f}")

mean_rmse_shuffled = np.mean(fold_rmses)
std_rmse_shuffled = np.std(fold_rmses)

print(f"\n{'='*70}")
print("RÉSULTATS:")
print(f"{'='*70}")
print(f"Mean RMSE (targets shufflés):  {mean_rmse_shuffled:.6f}")
print(f"Std RMSE (targets shufflés):   {std_rmse_shuffled:.6f}")
print(f"\nMean RMSE (targets originaux): 0.041680 (du test CV=6)")
print(f"Ratio (shuffled / original):   {mean_rmse_shuffled / 0.041680:.2f}x")

print(f"\n{'='*70}")
print("INTERPRÉTATION:")
print(f"{'='*70}")
if mean_rmse_shuffled > 0.05:
    print("✅ BON SIGNE: RMSE sur targets shufflés >> RMSE sur targets originaux")
    print("   Le modèle ne sur-ajuste pas les données aléatoires!")
else:
    print("❌ MAUVAIS SIGNE: RMSE similaire avec targets shufflés")
    print("   Possible problème: le modèle ne fait que sur-ajuster!")

print(f"\n{'='*70}\n")
