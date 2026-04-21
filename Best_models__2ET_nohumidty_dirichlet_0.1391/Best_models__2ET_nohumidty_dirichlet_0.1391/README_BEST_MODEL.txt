Best_models__2ET_nohumidty_dirichlet_0.1391

Selected run kept in this folder:
blender_et3_rowaggbs_piecewise_model50_cv3_20260420T162942Z

Pipeline:
- 2 ExtraTrees models
- Humidity removed before feature engineering
- model50 piecewise sample weights
- target-wise Dirichlet/simplex blend

Use this command to regenerate:
python scripts/train_best_2et_nohumidity_dirichlet.py --data-dir src/odor_competition/data
