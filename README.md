# Gaz Detection Competition

Repo reproductible pour le projet de prediction d'odeurs.

Le meilleur modele est:

`Best_models__2ET_nohumidty_dirichlet_0.1391`

Il utilise deux modeles ExtraTrees, retire `Humidity` des features, applique les poids `model50`, puis fait un blend Dirichlet cible par cible.

## 1. Installation

Depuis la racine du repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Les donnees attendues sont deja dans:

```text
src/odor_competition/data/
```

avec:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`

## 2. Lancer Le Meilleur Modele

Commande simple:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/odor_competition/data
```

Commande avec les arguments principaux:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py `
  --data-dir src/odor_competition/data `
  --output-dir artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391 `
  --submission-prefix best_2et_nohumidity_dirichlet `
  --cv-folds 3 `
  --random-state 42 `
  --n-jobs -1 `
  --tail-quantile 0.01 `
  --ratio-eps 0.001 `
  --dirichlet-samples 5000 `
  --dirichlet-batch-size 1024 `
  --dirichlet-alpha-vector 1.0 1.0 `
  --weight-preset model50
```

Pour voir tous les arguments disponibles:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py --help
```

## 3. Arguments Du Script Final

| Argument | Defaut | Role |
| --- | --- | --- |
| `--data-dir` | `src/odor_competition/data` | Dossier contenant `X_train.csv`, `X_test.csv`, `y_train.csv`. |
| `--output-dir` | `artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391` | Dossier de sortie. |
| `--submission-prefix` | `best_2et_nohumidity_dirichlet` | Prefixe des fichiers generes. |
| `--cv-folds` | `3` | Nombre de folds CV. Le pipeline final est prevu pour 3. |
| `--random-state` | `42` | Seed de reproductibilite. |
| `--n-jobs` | `-1` | Nombre de coeurs utilises par scikit-learn. |
| `--tail-quantile` | `0.01` | Clipping des valeurs extremes. |
| `--ratio-eps` | `0.001` | Petite constante pour les ratios de features. |
| `--dirichlet-samples` | `5000` | Nombre d'essais pour le blend Dirichlet. |
| `--dirichlet-batch-size` | `1024` | Taille de batch pour la recherche Dirichlet. |
| `--dirichlet-alpha-vector` | `1.0 1.0` | Parametres alpha du Dirichlet entre les deux modeles. |
| `--weight-preset` | `model50` | Regle de poids par bins d'humidite. |
| `--max-train-rows` | aucun | Limite de lignes train pour un test rapide. |
| `--max-test-rows` | aucun | Limite de lignes test pour un test rapide. |
| `--verbose` | off | Logs plus detailles. |

## 4. Fichiers Principaux Obtenus

Apres execution, les fichiers importants sont ecrits dans:

```text
artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391/
```

Les principaux fichiers sont:

| Fichier | Description |
| --- | --- |
| `best_2et_nohumidity_dirichlet_*.csv` | Soumission finale au format attendu. |
| `best_2et_nohumidity_dirichlet_*.json` | Resume du run: score CV, parametres, chemins, features. |
| `*_target_simplex_weights.csv` | Poids du blend pour chaque target. |
| `*_env_weight_bins.csv` | Resume des bins `Humidity` et des poids `model50`. |
| `*_feature_manifest.json` | Liste et nombre de features utilisees. |
| `*_oof_blend_modelspace.csv` | Predictions out-of-fold du blend, espace targets compresse. |
| `*_oof_blend_full.csv` | Predictions out-of-fold avec toutes les targets finales. |
| `*_test_blend_modelspace.csv` | Predictions test avant expansion finale. |
| `et_rowagg_mf06_bs_oof.csv` | OOF du premier ExtraTrees. |
| `et_allpool_3_oof.csv` | OOF du second ExtraTrees. |
| `et_rowagg_mf06_bs_test.csv` | Predictions test du premier ExtraTrees. |
| `et_allpool_3_test.csv` | Predictions test du second ExtraTrees. |

## 5. Test Rapide Sans Gros Entrainement

Pour verifier que le pipeline tourne sans relancer tout le calcul:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py `
  --data-dir src/odor_competition/data `
  --output-dir artifacts_extratrees_corr_optuna/_smoke_check `
  --submission-prefix smoke_best_2et `
  --max-train-rows 120 `
  --max-test-rows 20 `
  --dirichlet-samples 8 `
  --dirichlet-batch-size 8 `
  --n-jobs 1
```

Ce test sert seulement a verifier les imports, le chargement des donnees, les features et l'ecriture des fichiers.

## 6. Structure Du Repo

| Chemin | Role |
| --- | --- |
| `scripts/train_best_2et_nohumidity_dirichlet.py` | Script final a lancer. |
| `scripts/best_2et_nohumidity_core.py` | Fonctions centrales du meilleur modele. |
| `scripts/README.md` | Resume court des scripts. |
| `src/odor_competition/` | Chargement des donnees, schema targets, metriques, reporting. |
| `src/odor_competition/data/` | CSV d'entree. |
| `Best_models__2ET_nohumidty_dirichlet_0.1391/` | README visible du meilleur modele. |
| `artifacts_extratrees_corr_optuna/` | Artefacts generes localement. |
| `archive_scripts/` | Anciennes experiences conservees pour trace. |
| `notebooks/` | Wrappers notebooks et profiling. |
| `feature_corr_y/` | Analyses exploratoires de correlations et distributions. |

## 7. Notes

- Le script final utilise `model50` par defaut.
- `Humidity` sert aux poids d'echantillons, mais elle est retiree des features du modele.
- Les gros artefacts sont ignores par Git.
- Pour reproduire la livraison, utiliser seulement `scripts/train_best_2et_nohumidity_dirichlet.py`.
