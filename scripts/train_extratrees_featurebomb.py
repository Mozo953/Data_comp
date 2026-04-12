from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ExtraTrees with heavy feature engineering and implicit sensor normalization."
    )
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, y_train.csv.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts_extratrees_featurebomb")
    parser.add_argument("--submission-prefix", default="extratrees_featurebomb")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--skip-submission", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
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
    return constrained[[f"d{i:02d}" for i in range(1, 24)]]


def build_feature_bomb(features: pd.DataFrame) -> pd.DataFrame:
    base = raw_features(features)
    env = base["Env"]
    sensors = [c for c in base.columns if c != "Env"]

    engineered: dict[str, pd.Series] = {}

    # Implicit normalization by environmental factor (humidity-like driver).
    for column in sensors:
        engineered[f"{column}_div_env"] = base[column] / (env + EPS)
        engineered[f"{column}_mul_env"] = base[column] * env
        engineered[f"{column}_minus_env"] = base[column] - env

    # Block statistics to capture relative patterns.
    blocks = {
        "m_block": ["X12", "X13", "X14", "X15", "X4", "X5", "X6", "X7"],
        "support": ["Z", "Y1", "Y2", "Y3"],
        "all_no_env": sensors,
    }
    for name, cols in blocks.items():
        local = base[cols]
        local_max = local.max(axis=1)
        local_min = local.min(axis=1)
        engineered[f"{name}_mean"] = local.mean(axis=1)
        engineered[f"{name}_std"] = local.std(axis=1)
        engineered[f"{name}_min"] = local_min
        engineered[f"{name}_max"] = local_max
        engineered[f"{name}_range"] = local_max - local_min
        engineered[f"{name}_energy"] = np.square(local).sum(axis=1)

    # Combinatorial pairwise ratios and differences among sensor channels.
    pair_cols = sensors
    for i in range(len(pair_cols)):
        for j in range(i + 1, len(pair_cols)):
            a = pair_cols[i]
            b = pair_cols[j]
            engineered[f"{a}_div_{b}"] = base[a] / (base[b] + EPS)
            engineered[f"{a}_minus_{b}"] = base[a] - base[b]

    engineered_df = pd.DataFrame(engineered, index=base.index)
    out = pd.concat([base, engineered_df], axis=1)
    return out.astype(np.float32)


def build_model(args: argparse.Namespace, random_state: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=False,
        random_state=random_state,
        n_jobs=args.n_jobs,
    )


def summarize_feature_importance(model: ExtraTreesRegressor, feature_columns: list[str]) -> dict[str, float]:
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    return {feature: float(value) for feature, value in importance.items()}


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train_full: pd.DataFrame,
    train_target_columns: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    fold_scores: list[float] = []

    for fold_index, (fit_idx, valid_idx) in enumerate(cv.split(X_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train_full.iloc[fit_idx]
        y_valid = y_train_full.iloc[valid_idx]

        model = build_model(args, args.random_state + fold_index)
        model.fit(X_fit, y_fit)
        valid_pred = pd.DataFrame(model.predict(X_valid), columns=train_target_columns, index=X_valid.index)

        score = competition_rmse(
            enforce_known_constraints(y_valid.assign(d15=0.0)),
            enforce_known_constraints(valid_pred),
        )
        fold_scores.append(float(score))

    return {
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train = build_feature_bomb(data.x_train)
    X_test = build_feature_bomb(data.x_test)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train_full,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )

    diagnostic_model = build_model(args, args.random_state)
    diagnostic_model.fit(X_fit, y_fit)
    fit_pred = pd.DataFrame(diagnostic_model.predict(X_fit), columns=train_target_columns, index=X_fit.index)
    valid_pred = pd.DataFrame(diagnostic_model.predict(X_valid), columns=train_target_columns, index=X_valid.index)

    fit_full = enforce_known_constraints(fit_pred)
    valid_full = enforce_known_constraints(valid_pred)
    y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
    y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

    train_rmse = competition_rmse(y_fit_full, fit_full)
    validation_rmse = competition_rmse(y_valid_full, valid_full)

    cv_scores = run_cross_validation(X_train, y_train_full, train_target_columns, args)

    full_model = build_model(args, args.random_state)
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

    summary = {
        "generated_at_utc": timestamp,
        "task": "extratrees_featurebomb_regression",
        "data_dir": str(data_dir),
        "feature_count": int(X_train.shape[1]),
        "feature_columns": list(X_train.columns),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "holdout_fraction": float(args.holdout_fraction),
        "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
        "constant_targets": schema.constant_targets,
        "target_strategy": {
            "trained_targets": train_target_columns,
            "d15_handling": "removed_from_training_and_reinserted_as_zero",
            "duplicate_groups_handling": "predictions_averaged_within_known_duplicate_groups_before_export",
        },
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "bootstrap": False,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
        },
        "diagnostic_scores": {
            "train_rmse": float(train_rmse),
            "validation_rmse": float(validation_rmse),
            "full_train_rmse": float(full_train_rmse),
        },
        "cross_validation": {
            "n_folds": args.cv_folds,
            "scores": cv_scores,
        },
        "feature_importance": summarize_feature_importance(full_model, list(X_train.columns)),
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
