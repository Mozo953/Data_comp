from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    infer_target_schema,
    load_competition_data,
    prepare_feature_frames,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


FINAL_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_split": 0.005,
    "min_samples_leaf": 20,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}


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


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    n_trials: int = 50,
) -> dict:
    """Optimize RandomForest hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets (without d15)
        train_target_columns: List of target column names
        n_trials: Number of trials for Optuna optimization
    
    Returns:
        Dictionary of best hyperparameters found
    """
    if not OPTUNA_AVAILABLE:
        print("⚠️  Optuna not available, using default parameters")
        return FINAL_RF_PARAMS.copy()
    
    print("🔍 Starting Optuna hyperparameter optimization...")
    
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 30, step=2),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.02, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50, step=5),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0, step=0.1),
            "random_state": 42,
            "n_jobs": -1,
        }
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_fit, y_fit)
        
        # Predict and evaluate
        valid_pred = pd.DataFrame(
            model.predict(X_valid),
            columns=train_target_columns,
            index=X_valid.index,
        )
        valid_pred_full = enforce_known_constraints(valid_pred)
        y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))
        
        # Compute metric (lower is better)
        rmse = competition_rmse(y_valid_full, valid_pred_full)
        
        # Report intermediate result for pruning
        trial.report(rmse, step=0)
        
        return rmse
    
    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="rf_optimization",
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    best_trial = study.best_trial
    print(f"\n✅ Optimization complete!")
    print(f"   Best RMSE: {best_trial.value:.6f}")
    print(f"   Best parameters: {best_trial.params}")
    
    # Convert back to integer types if needed
    best_params = best_trial.params.copy()
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    
    return best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final full-data RandomForest submission model with optional Optuna hyperparameter tuning."
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing X_train.csv, X_test.csv, and y_train.csv.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization before training.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (only used with --optimize).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the holdout split.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Validation fraction for diagnostics.")
    parser.add_argument("--output-dir", default="artifacts_final_rf300", help="Directory for reports and outputs.")
    parser.add_argument("--submission-prefix", default="final_rf300", help="Prefix used for the submission file.")
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


def make_model(params: dict) -> RandomForestRegressor:
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

    X_train, X_test, _ = prepare_feature_frames(data.x_train, data.x_test)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    # Optimize hyperparameters if requested
    if args.optimize:
        rf_params = optimize_hyperparams(X_train, y_train_full, train_target_columns, n_trials=args.n_trials)
    else:
        rf_params = FINAL_RF_PARAMS.copy()
        print("📊 Using default parameters (use --optimize to search for best params)")

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train_full,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )

    diagnostic_model = make_model(rf_params)
    diagnostic_model.fit(X_fit, y_fit)
    fit_pred = pd.DataFrame(diagnostic_model.predict(X_fit), columns=train_target_columns, index=X_fit.index)
    valid_pred = pd.DataFrame(diagnostic_model.predict(X_valid), columns=train_target_columns, index=X_valid.index)

    fit_full = enforce_known_constraints(fit_pred)
    valid_full = enforce_known_constraints(valid_pred)
    y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
    y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

    train_rmse = competition_rmse(y_fit_full, fit_full)
    validation_rmse = competition_rmse(y_valid_full, valid_full)

    full_model = make_model(rf_params)
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
        "task": "toxic_gas_multioutput_regression",
        "final_model_name": "random_forest_raw_300_full_data",
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
        "hyperparameter_optimization": {
            "enabled": args.optimize,
            "n_trials": args.n_trials if args.optimize else 0,
        },
        "rf_params": rf_params,
        "diagnostic_scores": {
            "train_rmse": float(train_rmse),
            "validation_rmse": float(validation_rmse),
            "full_train_rmse": float(full_train_rmse),
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
