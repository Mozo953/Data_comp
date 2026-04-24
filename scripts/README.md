README Scripts

Ce dossier contient les scripts pour entrainer, analyser et visualiser le pipeline de detection de gaz.

Commande principale :

python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/gaz_competition/data

Commande SHAP du meilleur pipeline :

python scripts/shap_best_2et_nohumidity_dirichlet.py --data-dir src/gaz_competition/data

Scripts cles

train_best_2et_nohumidity_dirichlet.py : pipeline final complet (2 ExtraTrees + blend Dirichlet + submission)
best_2et_nohumidity_core.py : coeur du pipeline final et feature engineering sans humidite
shap_best_2et_nohumidity_dirichlet.py : explication SHAP des 2 ExtraTrees du meilleur pipeline
check_shuffled_targets.py : sanity check anti-fuite
adversarial_validation_train_test.py : analyse du shift train/test

A retenir

Le blend final est un ensemble de 2 modeles, donc le script SHAP explique chaque ExtraTrees separement.
Les sorties volumineuses restent dans artifacts_extratrees_corr_optuna/
