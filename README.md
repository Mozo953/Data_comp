# Odor Detection Competition

Final project code and reports for Competition 1.

## Task

Predict `d01` to `d23` from the 13 sensor measurements in `X_train.csv`. The evaluation uses a weighted RMSE-style metric, and the test set is known to be shifted relative to training conditions.

## Final Deliverables

- Final training/submission script: `scripts/train_final_rf300.py`
- Final notebook wrapper: `notebooks/13_final_rf300_submission.ipynb`
- Methodology report: `reports/final_submission_methodology.md`
- Profiling entry point: `scripts/generate_profile_report.py`

## Main Commands

```powershell
python scripts/generate_profile_report.py --data-dir . --dataset raw_train_targets --sample-size 50000
python scripts/train_final_rf300.py --data-dir . --skip-submission
python scripts/train_final_rf300.py --data-dir .
```

If the CSV files are stored somewhere else, replace `.` with that folder path.

## Project Structure

- `src/odor_competition/` shared data, metric, and reporting utilities
- `scripts/` reproducible training and reporting entry points
- `notebooks/` notebook wrappers and archived experiments
- `reports/` written methodology and generated analysis
- `submissions/` exported prediction files

## Notes

- The final submission path is the raw-feature `RandomForestRegressor` with 300 trees.
- Archived experiment notebooks are kept for traceability, but the client-facing final workflow is the RF300 pipeline above.
- Raw CSV files, generated outputs, and competition PDFs stay local and are ignored by Git where appropriate.
