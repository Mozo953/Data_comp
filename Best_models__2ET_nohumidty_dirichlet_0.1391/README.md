# Best_models__2ET_nohumidty_dirichlet_0.1391

Ce dossier sert de coin bien visible pour le meilleur modèle final du projet.

Le nom est volontairement conservé tel quel pour rester cohérent avec les artefacts déjà générés:

`Best_models__2ET_nohumidty_dirichlet_0.1391`

Il correspond au modèle final sélectionné pour la livraison: un blend de deux modèles ExtraTrees, entraîné sans utiliser `Humidity` comme feature directe, puis combiné cible par cible avec une recherche Dirichlet/simplex.

## Résumé Rapide

- Modèle final: blend de deux `ExtraTreesRegressor`.
- Score repère: `0.1391`.
- Pondération retenue: preset `model50`.
- Validation: CV à 3 folds.
- Données attendues: `X_train.csv`, `X_test.csv`, `y_train.csv`.
- Script principal: `scripts/train_best_2et_nohumidity_dirichlet.py`.
- Dossier des gros artefacts: `artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391/`.

## Idée Générale

Le modèle final combine deux visions complémentaires des données:

1. Un modèle `et_rowagg_mf06_bs`, basé sur des agrégats calculés ligne par ligne.
2. Un modèle `et_allpool_3`, basé sur un ensemble plus large de features dérivées.

Les deux modèles sont entraînés séparément, puis leurs prédictions sont combinées avec des poids différents selon les targets. Ces poids sont optimisés avec une recherche Dirichlet/simplex sur les prédictions out-of-fold.

L’objectif est de garder un modèle robuste, simple à relancer, et moins dépendant d’un seul type de feature engineering.

## Pourquoi Retirer Humidity Des Features

La variable `Humidity` est très importante pour comprendre le décalage entre train et test, mais elle peut aussi créer une dépendance trop directe dans les features.

Dans ce pipeline, `Humidity` est donc utilisée pour définir des poids d’échantillons, mais elle est retirée des features avant le feature engineering. Cela permet de tenir compte du shift train/test sans laisser le modèle apprendre une règle trop fragile directement sur l’humidité.

Le script vérifie aussi que les colonnes liées à l’environnement ne réapparaissent pas dans les features finales:

- `Humidity`
- `humidity_*`
- `humidity_times_*`
- `support_gap`
- `env_*`
- `env_times_*`

## Pondération `model50`

Le preset final utilisé par défaut est `model50`.

Il applique une pondération par intervalles de `Humidity`:

| Intervalle Humidity | Poids |
| --- | ---: |
| `[0.00, 0.20)` | `1.00` |
| `[0.20, 0.39)` | `1.10` |
| `[0.39, 0.50)` | `1.35` |
| `[0.50, 0.68)` | `1.00` |
| `[0.68, 0.80)` | `1.25` |
| `[0.80, 1.00]` | `1.25` |

Ces poids donnent plus d’importance à certaines zones d’humidité pendant l’entraînement, afin de mieux gérer le décalage de distribution observé entre train et test.

## Modèles De Base

### `et_rowagg_mf06_bs`

Ce modèle utilise une vue compacte des données.

Les features principales sont des statistiques par ligne:

- moyenne des capteurs;
- écart-type;
- quantiles;
- IQR;
- somme absolue;
- norme L2;
- moyennes par blocs de capteurs;
- quelques ratios et écarts ciblés.

Ce modèle est plus simple et plus stable. Il sert de composant robuste dans le blend.

### `et_allpool_3`

Ce modèle utilise une vue plus large.

Il reprend les features brutes nettoyées, ajoute les agrégats, puis ajoute davantage de transformations par capteur et par bloc.

Il est plus expressif que `et_rowagg_mf06_bs`, et dans les artefacts du run final il gagne la majorité des targets dans la pondération simplex.

## Blend Dirichlet/Simplex

Après la génération des prédictions out-of-fold, le script cherche les meilleurs poids de combinaison entre les deux modèles.

Cette optimisation se fait cible par cible:

- certaines targets peuvent préférer presque uniquement `et_allpool_3`;
- d’autres peuvent bénéficier d’une contribution plus forte de `et_rowagg_mf06_bs`;
- le blend final applique donc des poids différents selon la cible.

Le fichier suivant résume ces poids:

`blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z_target_simplex_weights.csv`

## Commande De Reproduction

Depuis la racine du repo:

```powershell
python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/odor_competition/data
```

Cette commande relance le pipeline complet avec les valeurs par défaut:

- `--weight-preset model50`
- `--cv-folds 3`
- `--random-state 42`
- sortie dans `artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391`

## Smoke Test Rapide

Pour vérifier que le script fonctionne sans relancer l’entraînement complet:

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

Ce test ne sert pas à produire un score significatif. Il sert uniquement à vérifier que:

- les imports fonctionnent;
- le module core est bien chargé;
- les features sont construites;
- les modèles s’entraînent;
- le JSON de résumé est bien généré;
- les chemins de sortie sont corrects.

## Fichiers Importants Dans Les Artefacts

Les gros fichiers générés sont stockés ici:

`artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391/`

Les fichiers les plus utiles sont:

| Fichier | Rôle |
| --- | --- |
| `blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z.csv` | Soumission finale générée pour le test set. |
| `blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z.json` | Résumé complet du run: paramètres, scores CV, chemins, features. |
| `blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z_target_simplex_weights.csv` | Poids du blend par target. |
| `blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z_env_weight_bins.csv` | Répartition des poids par bins d’humidité. |
| `blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z_feature_manifest.json` | Liste et nombre de features utilisées. |
| `et_allpool_3_oof.csv` | Prédictions out-of-fold du modèle `et_allpool_3`. |
| `et_rowagg_mf06_bs_oof.csv` | Prédictions out-of-fold du modèle `et_rowagg_mf06_bs`. |
| `et_allpool_3_test.csv` | Prédictions test du modèle `et_allpool_3`. |
| `et_rowagg_mf06_bs_test.csv` | Prédictions test du modèle `et_rowagg_mf06_bs`. |
| `model50_piecewise_humidity_loss_20260420T164429Z.svg` | Visualisation des pertes par bins d’humidité. |
| `model50_piecewise_weight_bins_20260420T164517Z.svg` | Visualisation des poids `model50`. |

## Ce Qu’il Faut Regarder En Premier

Pour comprendre rapidement le run final:

1. Ouvrir le fichier JSON de résumé.
2. Vérifier `simplex.oof_weighted_wrmse`.
3. Vérifier `base_models` pour comparer les deux modèles seuls.
4. Ouvrir `target_simplex_weights.csv` pour voir quelles targets utilisent quel modèle.
5. Ouvrir `feature_manifest.json` pour vérifier que `Humidity` n’est pas dans les features finales.

## Structure Du Code

Le pipeline est séparé en deux fichiers principaux.

### Script final

`scripts/train_best_2et_nohumidity_dirichlet.py`

Ce fichier contient:

- le parsing des arguments;
- le choix du preset de pondération;
- la CV;
- l’optimisation du blend;
- l’écriture des prédictions;
- l’écriture du JSON de résumé.

### Module core

`scripts/best_2et_nohumidity_core.py`

Ce fichier contient:

- les fonctions de feature engineering;
- la construction des vues `rowagg` et `allpool`;
- les définitions des deux modèles ExtraTrees;
- le calcul de la métrique pondérée;
- les fonctions de blend Dirichlet/simplex.

Cette séparation rend le script final plus lisible et permet aux anciens scripts d’expérimentation de réutiliser les mêmes helpers.

## Résultats Du Run Conservé

Le run conservé dans les artefacts est:

`20260420T162942Z`

Dans le résumé JSON, les points clés sont:

- `et_rowagg_mf06_bs` CV3 weighted WRMSE: environ `0.02997`.
- `et_allpool_3` CV3 weighted WRMSE: environ `0.02556`.
- Blend final OOF weighted WRMSE: environ `0.02465`.
- La majorité des targets sont gagnées par `et_allpool_3`.

Le label `0.1391` dans le nom du dossier correspond au score/repère de sélection côté livraison, tandis que les scores internes du JSON sont les scores CV pondérés calculés pendant la reproduction.

## Notes De Livraison

Ce dossier racine ne contient volontairement qu’un README pour rendre le meilleur modèle visible immédiatement.

Les gros fichiers restent dans `artifacts_extratrees_corr_optuna/` parce qu’ils sont volumineux et ignorés par Git. Cela évite d’alourdir le dépôt tout en gardant les artefacts disponibles localement.

Si quelqu’un ouvre le repo sans connaître l’historique, il doit pouvoir comprendre rapidement:

- quel modèle est le bon;
- quel script lancer;
- quels artefacts inspecter;
- pourquoi `Humidity` est traitée à part;
- où se trouvent les prédictions finales.
