Best_models__2ET_nohumidty_dirichlet_0.1391

Run final conserve pour la detection de gaz:
blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z

Pipeline:
- 2 modeles ExtraTrees
- Humidity retiree avant le feature engineering
- poids model50 par bins d'humidite
- blend Dirichlet/simplex cible par cible

Commande de reproduction:
python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/odor_competition/data
