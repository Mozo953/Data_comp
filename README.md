# Gas Detection Competition

Repo reproductible pour un projet de detection de gaz a partir de mesures capteurs.

Le meilleur modele est:

`Best_models__2ET_nohumidty_dirichlet_0.1391`

Il combine deux modeles ExtraTrees, retire `Humidity` des features d'entree, applique les poids `model50`, puis fait un blend Dirichlet cible par cible.

> Note: le package Python s'appelle encore `odor_competition` pour garder les imports stables. Dans ce repo, il correspond bien au workflow de detection de gaz.

## 1. Installation

Depuis la racine du repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Les donnees attendues sont dans:

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

Commande complete avec les principaux arguments:

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

Voir tous les arguments:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py --help
```

## 3. Arguments Du Script Final

| Argument | Defaut | Role |
| --- | --- | --- |
| `--data-dir` | `src/odor_competition/data` | Dossier contenant les CSV. |
| `--output-dir` | `artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391` | Dossier de sortie. |
| `--submission-prefix` | `best_2et_nohumidity_dirichlet` | Prefixe des fichiers generes. |
| `--cv-folds` | `3` | Nombre de folds. Le pipeline final est prevu pour 3. |
| `--random-state` | `42` | Seed de reproductibilite. |
| `--n-jobs` | `-1` | Nombre de coeurs utilises. |
| `--tail-quantile` | `0.01` | Clipping des valeurs extremes des capteurs. |
| `--ratio-eps` | `0.001` | Petite constante pour les ratios de features. |
| `--dirichlet-samples` | `5000` | Nombre d'essais pour optimiser le blend. |
| `--dirichlet-batch-size` | `1024` | Taille des batches Dirichlet. |
| `--dirichlet-alpha-vector` | `1.0 1.0` | Parametres alpha entre les deux modeles. |
| `--weight-preset` | `model50` | Poids par bins d'humidite. |
| `--max-train-rows` | aucun | Limite train pour un test rapide. |
| `--max-test-rows` | aucun | Limite test pour un test rapide. |
| `--verbose` | off | Logs plus detailles. |

## 4. Fichiers Principaux Obtenus

Le run ecrit principalement:

| Fichier | Description |
| --- | --- |
| `best_2et_nohumidity_dirichlet_*.csv` | Soumission finale. |
| `best_2et_nohumidity_dirichlet_*.json` | Resume complet du run. |
| `*_target_simplex_weights.csv` | Poids du blend par cible gaz. |
| `*_humidity_weight_bins.csv` | Bins d'humidite et poids `model50`. |
| `*_feature_manifest.json` | Liste et nombre de features. |
| `*_oof_blend_modelspace.csv` | Predictions out-of-fold du blend. |
| `*_oof_blend_full.csv` | OOF avec toutes les cibles finales. |
| `*_test_blend_modelspace.csv` | Predictions test avant expansion finale. |
| `et_rowagg_mf06_bs_oof.csv` | OOF du modele ExtraTrees row-aggregate. |
| `et_allpool_3_oof.csv` | OOF du modele ExtraTrees all-pool. |
| `et_rowagg_mf06_bs_test.csv` | Predictions test du premier modele. |
| `et_allpool_3_test.csv` | Predictions test du second modele. |

## 5. Test Rapide

Pour verifier que le pipeline tourne sans gros entrainement:

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

Ce test sert seulement a verifier les imports, les features, les modeles et l'ecriture des fichiers.

## 6. Structure Du Repo

| Chemin | Role |
| --- | --- |
| `scripts/train_best_2et_nohumidity_dirichlet.py` | Script final a lancer. |
| `scripts/best_2et_nohumidity_core.py` | Fonctions centrales du meilleur modele. |
| `scripts/README.md` | Resume court des scripts. |
| `src/odor_competition/` | Package historique: chargement des donnees, metriques, soumission. |
| `src/odor_competition/data/` | CSV d'entree. |
| `Best_models__2ET_nohumidty_dirichlet_0.1391/` | Dossier visible du meilleur modele. |
| `archive_scripts/` | Anciennes experiences conservees. |
| `notebooks/` | Notebooks de reproduction et profiling. |
| `feature_corr_y/` | Analyses exploratoires. |

## 7. Notes

- Le sujet du repo est la detection de gaz.
- `Humidity` est utilisee pour la ponderation, mais retiree des features du modele.
- Les gros artefacts sont ignores par Git.
- Pour reproduire la livraison, utiliser `scripts/train_best_2et_nohumidity_dirichlet.py`.

