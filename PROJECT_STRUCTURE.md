# Structure Du Projet

Ce repo est organise autour du meilleur modele de detection de gaz.

## Fichiers Prioritaires

| Chemin | Role |
| --- | --- |
| `README.md` | Guide global: installation, commande finale, sorties. |
| `requirements.txt` | Dependances Python. |
| `scripts/train_best_2et_nohumidity_dirichlet.py` | Script final a lancer. |
| `scripts/best_2et_nohumidity_core.py` | Fonctions du meilleur modele. |
| `scripts/README.md` | Resume court des scripts. |
| `Best_models__2ET_nohumidty_dirichlet_0.1391/README.md` | Explication du meilleur modele. |

## Donnees

Les CSV d'entree sont dans:

```text
src/odor_competition/data/
```

Fichiers attendus:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`


## Artefacts

Les predictions, OOF, poids de blend et resumes JSON sont generes dans un dossier d'artefacts, en general:

```text
artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391/
```

## Archives

`archive_scripts/` contient les anciennes experiences conservees pour trace. Elles ne sont pas necessaires pour refaire la soumission finale.

