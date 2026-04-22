# Best_models__2ET_nohumidty_dirichlet_0.1391

Ce dossier met en avant le meilleur modele final pour la detection de gaz.

Le nom du dossier est conserve tel quel pour rester coherent avec les artefacts deja generes.

## Resume

- Tache: prediction des cibles gaz a partir de mesures capteurs.
- Modele final: blend de deux `ExtraTreesRegressor`.
- Score repere: `0.1391`.
- Validation: CV a 3 folds.
- Ponderation: preset `model50`.
- Particularite: `Humidity` est retiree des features, mais utilisee pour ponderer les lignes.
- Script principal: `scripts/train_best_2et_nohumidity_dirichlet.py`.

## Pipeline Selectionne

Le modele combine deux vues des donnees capteurs:

1. `et_rowagg_mf06_bs`: features agregees par ligne.
2. `et_allpool_3`: ensemble plus large de features derivees.

Les deux modeles produisent des predictions out-of-fold. Ensuite, un blend Dirichlet/simplex choisit des poids differents pour chaque cible gaz.

## Pourquoi Traiter `Humidity` A Part

`Humidity` aide a comprendre le shift entre train et test, mais elle peut rendre le modele trop dependant d'une variable d'humidite.

Dans ce pipeline:

- `Humidity` est supprimee des features;
- les colonnes derivees de `Humidity` sont aussi interdites;
- `Humidity` sert seulement a creer des poids d'echantillons.

## Poids `model50`

| Intervalle Humidity | Poids |
| --- | ---: |
| `[0.00, 0.20)` | `1.00` |
| `[0.20, 0.39)` | `1.10` |
| `[0.39, 0.50)` | `1.35` |
| `[0.50, 0.68)` | `1.00` |
| `[0.68, 0.80)` | `1.25` |
| `[0.80, 1.00]` | `1.25` |

## Reproduire

Depuis la racine du repo:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/odor_competition/data
```

## Fichiers A Regarder

Les artefacts principaux du run sont:

| Fichier | Role |
| --- | --- |
| `*_submission*.csv` ou `best_2et_nohumidity_dirichlet_*.csv` | Soumission finale. |
| `*.json` | Resume du run et parametres. |
| `*_target_simplex_weights.csv` | Poids du blend par cible gaz. |
| `*_feature_manifest.json` | Liste des features utilisees. |
| `*_humidity_weight_bins.csv` | Bins d'humidite et poids. |
| `et_allpool_3_oof.csv` | OOF du modele all-pool. |
| `et_rowagg_mf06_bs_oof.csv` | OOF du modele row-aggregate. |

## Note Sur Le Nom Du Package

Le code Python utilise encore le chemin `src/odor_competition/`. C'est un nom historique du projet. Les donnees et la documentation de livraison concernent bien la detection de gaz.

