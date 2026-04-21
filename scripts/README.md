# Scripts

Ce dossier contient les scripts d'entrainement, de test et de visualisation.

## A lancer en priorite

### `train_best_2et_nohumidity_dirichlet.py`

Script final. Il reproduit le meilleur modele:

- 2 modeles ExtraTrees;
- `Humidity` retiree des features;
- poids `model50`;
- blend Dirichlet par target;
- generation de la soumission.

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/odor_competition/data
```

### `best_2et_nohumidity_core.py`

Code commun du modele final:

- creation des features;
- definition des modeles;
- metrique;
- blend;
- helpers de prediction.

Plusieurs scripts dependent de ce fichier.

### `check_shuffled_targets.py`

Test rapide pour verifier que le modele n'apprend pas du bruit.

```powershell
python scripts/check_shuffled_targets.py --data-dir src/odor_competition/data --target c01
```

## Autres scripts

| Type | Scripts |
| --- | --- |
| Experiences modeles | fichiers dans `archive_scripts/` |
| Diagnostics | `adversarial_validation_train_test.py`, `compare_two_models_humidity_bin_loss.py` |
| Visualisations | tous les scripts `plot_*.py` |

## Dependances internes

Les scripts utilisent surtout:

- `src/odor_competition/data.py` pour charger les donnees et creer la soumission;
- `src/odor_competition/metrics.py` pour la metrique;
- `src/odor_competition/reporting.py` pour certains graphes;
- `src/odor_competition/data_shift.py` pour les analyses de shift train/test.

## Dependances Python

Voir `requirements.txt`.

Principales libs:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `optuna`
