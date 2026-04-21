# Final Delivery Package

This package contains the final reproducible workflow for the odor-detection competition.

## Included files

- `requirements.txt`
- `scripts/train_final_rf300.py`
- `scripts/generate_profile_report.py`
- `src/odor_competition/data.py`
- `src/odor_competition/metrics.py`
- `src/odor_competition/reporting.py`
- `src/odor_competition/__init__.py`
- `notebooks/13_final_rf300_submission.ipynb`
- `reports/final_submission_report.tex`
- `reports/final_submission_methodology.md`

## Required data files

Place these files in one folder:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`

You can keep that folder as the package root, or pass it explicitly with `--data-dir`.

## Install

```powershell
pip install -r requirements.txt
```

## Generate the profiling report

```powershell
python scripts/generate_profile_report.py --data-dir . --dataset raw_train_targets --sample-size 50000
```

## Run diagnostics only

```powershell
python scripts/train_final_rf300.py --data-dir . --skip-submission
```

## Generate the final submission

```powershell
python scripts/train_final_rf300.py --data-dir .
```

## Notebook option

```powershell
jupyter notebook notebooks/13_final_rf300_submission.ipynb
```

The final selected submission model is the raw-feature RandomForest with 300 trees and deterministic handling of `c15` plus the known duplicate target groups.
