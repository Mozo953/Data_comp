from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, infer_target_schema, load_competition_data, raw_features  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


TRIAL11_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 22,
    "min_samples_split": 0.0010335242134112178,
    "min_samples_leaf": 50,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a second final RandomForest submission model using Optuna Trial 11 parameters."
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing X_train.csv, X_test.csv, and y_train.csv.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the holdout split.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Validation fraction for diagnostics.")
    parser.add_argument("--output-dir", default="artifacts_final_rf500_trial11", help="Directory for reports and outputs.")
    parser.add_argument("--submission-prefix", default="final_rf500_trial11", help="Prefix used for the submission file.")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the fitted full model to disk for later test-only inference.",
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Run diagnostics only and skip generating the final CSV.",
    )
    args = parser.parse_args()

    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be between 0 and 1.")
    return args


def prepare_targets(y_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target_frame = y_frame.drop(columns=["ID"]) if "ID" in y_frame.columns else y_frame.copy()
    train_targets = [column for column in target_frame.columns if column != "d15"]
    return target_frame[train_targets].copy(), train_targets


def enforce_known_constraints(predictions: pd.DataFrame) -> pd.DataFrame:
    constrained = predictions.copy()
    duplicate_groups = [
        ["d02", "d07", "d20"],
        ["d03", "d04", "d22"],
        ["d05", "d06", "d21"],
    ]
    for group in duplicate_groups:
        group_mean = constrained[group].mean(axis=1)
        for column in group:
            constrained[column] = group_mean
    constrained["d15"] = 0.0
    final_columns = [f"d{i:02d}" for i in range(1, 24)]
    return constrained[final_columns]


def make_model(random_state: int | None = None) -> RandomForestRegressor:
    params = TRIAL11_RF_PARAMS.copy()
    if random_state is not None:
        params["random_state"] = random_state
    return RandomForestRegressor(**params)


def summarize_feature_importance(model: RandomForestRegressor, feature_columns: list[str]) -> dict[str, float]:
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    return {feature: float(value) for feature, value in importance.items()}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train = raw_features(data.x_train)
    X_test = raw_features(data.x_test)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train_full,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )

    diagnostic_model = make_model(random_state=args.random_state)
    diagnostic_model.fit(X_fit, y_fit)
    fit_pred = pd.DataFrame(diagnostic_model.predict(X_fit), columns=train_target_columns, index=X_fit.index)
    valid_pred = pd.DataFrame(diagnostic_model.predict(X_valid), columns=train_target_columns, index=X_valid.index)

    fit_full = enforce_known_constraints(fit_pred)
    valid_full = enforce_known_constraints(valid_pred)
    y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
    y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

    train_rmse = competition_rmse(y_fit_full, fit_full)
    validation_rmse = competition_rmse(y_valid_full, valid_full)

    full_model = make_model(random_state=args.random_state)
    full_model.fit(X_train, y_train_full)
    full_train_pred = pd.DataFrame(full_model.predict(X_train), columns=train_target_columns, index=X_train.index)
    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train_full.assign(d15=0.0)),
        enforce_known_constraints(full_train_pred),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        test_pred = pd.DataFrame(full_model.predict(X_test), columns=train_target_columns, index=X_test.index)
        final_predictions = enforce_known_constraints(test_pred)
        submission = build_submission_frame(data.x_test["ID"], final_predictions)
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    model_path = None
    if args.save_model:
        model_file = output_dir / f"{args.submission_prefix}_{timestamp}.joblib"
        joblib.dump(
            {
                "model": full_model,
                "feature_columns": list(X_train.columns),
                "train_target_columns": train_target_columns,
                "rf_params": TRIAL11_RF_PARAMS,
                "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
                "constant_targets": schema.constant_targets,
            },
            model_file,
        )
        model_path = str(model_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "toxic_gas_multioutput_regression",
        "final_model_name": "random_forest_trial11_500",
        "data_dir": str(data_dir),
        "feature_columns": list(X_train.columns),
        "feature_count": int(X_train.shape[1]),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "holdout_fraction": float(args.holdout_fraction),
        "holdout_rows": {
            "fit": int(len(X_fit)),
            "validation": int(len(X_valid)),
        },
        "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
        "constant_targets": schema.constant_targets,
        "target_strategy": {
            "trained_targets": train_target_columns,
            "d15_handling": "removed_from_training_and_reinserted_as_zero",
            "duplicate_groups_handling": "predictions_averaged_within_known_duplicate_groups_before_export",
        },
        "rf_params": TRIAL11_RF_PARAMS,
        "diagnostic_scores": {
            "train_rmse": float(train_rmse),
            "validation_rmse": float(validation_rmse),
            "full_train_rmse": float(full_train_rmse),
        },
        "feature_importance": summarize_feature_importance(full_model, list(X_train.columns)),
        "model_path": model_path,
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
