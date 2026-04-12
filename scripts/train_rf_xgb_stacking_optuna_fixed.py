"""
RF + XGBoost Stacking with Optuna optimization.
Tests with/without environment variable to assess its utility.
Trains a Ridge meta-learner on OOF predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    engineer_features,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF + XGBoost Stacking with Optuna and environment variable utility check."
    )
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv and y_train.csv.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3, help="Number of KFold splits for OOF (lightweight).")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials (lightweight).")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Final holdout for validation.")
    parser.add_argument("--output-dir", default="artifacts_rf_xgb_stacking")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="Progress print interval.")
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
    return constrained[[f"d{i:02d}" for i in range(1, 24)]]


def build_feature_sets(x_train: pd.DataFrame, x_test: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build feature sets with and without environment variable (lightweight version)."""
    raw_train = raw_features(x_train)
    raw_test = raw_features(x_test)

    eng_train = engineer_features(x_train)
    eng_test = engineer_features(x_test)

    return {
        "raw_with_env": (raw_train, raw_test),
        "raw_no_env": (raw_train.drop(columns=["Env"]), raw_test.drop(columns=["Env"])),
        "engineered_with_env": (eng_train, eng_test),
        "engineered_no_env": (eng_train.drop(columns=["Env"]), eng_test.drop(columns=["Env"])),
    }


def generate_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_splits: int,
    rf_params: dict,
    xgb_params: dict,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Out-of-Fold predictions for RF and XGBoost.
    Returns: (rf_oof, xgb_oof)
    """
    n_targets = y_train.shape[1]
    rf_oof = np.zeros((X_train.shape[0], n_targets), dtype=float)
    xgb_oof = np.zeros((X_train.shape[0], n_targets), dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]

        # RF
        rf = RandomForestRegressor(
            **rf_params,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_fold_train, y_fold_train)
        rf_oof[valid_idx] = rf.predict(X_fold_valid)

        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb = MultiOutputRegressor(
                XGBRegressor(
                    **xgb_params,
                    random_state=random_state,
                    tree_method="hist",
                    objective="reg:squarederror",
                    verbosity=0,
                ),
                n_jobs=-1,
            )
            xgb.fit(X_fold_train, y_fold_train)
            xgb_oof[valid_idx] = xgb.predict(X_fold_valid)

    return rf_oof, xgb_oof


def create_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    n_splits: int,
    random_state: int,
):
    """Create an Optuna objective function for RF + XGBoost stacking."""

    def objective(trial: optuna.Trial) -> float:
        # RF hyperparameters (lightweight)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 200, 400, step=50)
        rf_max_depth = trial.suggest_int("rf_max_depth", 18, 26)
        rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 10, 40)
        rf_max_features = trial.suggest_float("rf_max_features", 0.65, 0.9)

        # XGBoost hyperparameters (lightweight)
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 150, 300, step=50)
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 4, 8)
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.02, 0.15)
        xgb_subsample = trial.suggest_float("xgb_subsample", 0.7, 1.0)
        xgb_colsample = trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0)
        xgb_reg_alpha = trial.suggest_float("xgb_reg_alpha", 1e-4, 1e-2, log=True)
        xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.5, 3.0)

        # Meta-learner hyperparameters
        meta_alpha = trial.suggest_float("meta_alpha", 0.01, 10.0, log=True)

        rf_params = {
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "min_samples_leaf": rf_min_samples_leaf,
            "max_features": rf_max_features,
        }

        xgb_params = {
            "n_estimators": xgb_n_estimators,
            "max_depth": xgb_max_depth,
            "learning_rate": xgb_learning_rate,
            "subsample": xgb_subsample,
            "colsample_bytree": xgb_colsample,
            "reg_alpha": xgb_reg_alpha,
            "reg_lambda": xgb_reg_lambda,
        }

        try:
            # Generate OOF predictions
            rf_oof, xgb_oof = generate_oof_predictions(
                X_train, y_train, n_splits, rf_params, xgb_params, random_state
            )

            # Stack OOF predictions
            meta_features = np.hstack([rf_oof, xgb_oof])

            # Train meta-learner (Ridge)
            meta_model = Ridge(alpha=meta_alpha)
            meta_model.fit(meta_features, y_train.values)

            # Evaluate on OOF
            meta_pred = meta_model.predict(meta_features)
            meta_pred_df = pd.DataFrame(meta_pred, columns=train_target_columns, index=y_train.index)
            meta_pred_constrained = enforce_known_constraints(meta_pred_df)

            y_train_constrained = enforce_known_constraints(y_train.assign(d15=0.0))
            score = competition_rmse(y_train_constrained, meta_pred_constrained)

            return float(score)

        except Exception as e:
            print(f"    Trial error: {e}")
            return float("inf")

    return objective


def evaluate_feature_set(
    feature_mode: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    n_splits: int,
    n_trials: int,
    random_state: int,
    heartbeat_seconds: int,
) -> dict:
    """Optimize and evaluate a single feature set with Optuna."""
    print(f"\n=== Testing feature mode: {feature_mode} ({X_train.shape[1]} features) ===")

    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available. Install with: pip install optuna")
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost not available. Install with: pip install xgboost")

    # Create Optuna study
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=0)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = create_optuna_objective(X_train, y_train, train_target_columns, n_splits, random_state)

    t0 = time.time()
    last_heartbeat = t0

    def callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        nonlocal last_heartbeat
        now = time.time()
        best_val = study.best_value
        elapsed = (now - t0) / 60.0
        trial_num = trial.number + 1
        
        # Print every few trials or at heartbeat interval
        if trial_num <= 2 or (now - last_heartbeat) >= heartbeat_seconds:
            print(f"  Trial {trial_num:2d}/{n_trials}: rmse={trial.value:.6f}, best={best_val:.6f}, {elapsed:.1f}m")
            last_heartbeat = now

    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best_trial = study.best_trial
    best_val = best_trial.value
    elapsed = (time.time() - t0) / 60.0

    print(f"  ✓ Best trial #{best_trial.number}: rmse={best_val:.6f} ({elapsed:.1f}m)")

    # Retrain with best parameters on full training data
    best_params = best_trial.params

    rf_params = {
        "n_estimators": best_params["rf_n_estimators"],
        "max_depth": best_params["rf_max_depth"],
        "min_samples_leaf": best_params["rf_min_samples_leaf"],
        "max_features": best_params["rf_max_features"],
    }

    xgb_params = {
        "n_estimators": best_params["xgb_n_estimators"],
        "max_depth": best_params["xgb_max_depth"],
        "learning_rate": best_params["xgb_learning_rate"],
        "subsample": best_params["xgb_subsample"],
        "colsample_bytree": best_params["xgb_colsample_bytree"],
        "reg_alpha": best_params["xgb_reg_alpha"],
        "reg_lambda": best_params["xgb_reg_lambda"],
    }

    meta_alpha = best_params["meta_alpha"]

    # Generate OOF on full training data
    rf_oof, xgb_oof = generate_oof_predictions(X_train, y_train, n_splits, rf_params, xgb_params, random_state)
    meta_features = np.hstack([rf_oof, xgb_oof])

    # Train final meta-learner
    meta_model = Ridge(alpha=meta_alpha)
    meta_model.fit(meta_features, y_train.values)

    # Predict on test set (need full training RF and XGBoost)
    rf_full = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)
    rf_full.fit(X_train, y_train)

    xgb_full = MultiOutputRegressor(
        XGBRegressor(
            **xgb_params,
            random_state=random_state,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        ),
        n_jobs=-1,
    )
    xgb_full.fit(X_train, y_train)

    rf_test_pred = rf_full.predict(X_test)
    xgb_test_pred = xgb_full.predict(X_test)

    test_meta_features = np.hstack([rf_test_pred, xgb_test_pred])
    test_pred_raw = meta_model.predict(test_meta_features)
    test_pred_df = pd.DataFrame(test_pred_raw, columns=train_target_columns)
    test_pred_constrained = enforce_known_constraints(test_pred_df)

    return {
        "feature_mode": feature_mode,
        "best_trial_number": best_trial.number,
        "best_validation_rmse": best_val,
        "best_params": best_trial.params,
        "elapsed_minutes": elapsed,
        "n_trials": n_trials,
        "test_predictions": test_pred_constrained,
        "meta_model": meta_model,
        "rf_model": rf_full,
        "xgb_model": xgb_full,
    }


def main() -> None:
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not available. Install with: pip install optuna")
        sys.exit(1)

    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost not available. Install with: pip install xgboost")
        sys.exit(1)

    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    feature_sets = build_feature_sets(data.x_train, data.x_test)

    # Evaluate each feature set
    results = []
    overall_best = None
    overall_best_score = float("inf")

    t0_global = time.time()

    for feature_mode in sorted(feature_sets.keys()):
        X_train, X_test = feature_sets[feature_mode]

        result = evaluate_feature_set(
            feature_mode,
            X_train,
            X_test,
            y_train_full,
            train_target_columns,
            args.n_splits,
            args.n_trials,
            args.random_state,
            args.heartbeat_seconds,
        )

        results.append(result)

        if result["best_validation_rmse"] < overall_best_score:
            overall_best_score = result["best_validation_rmse"]
            overall_best = result

    elapsed_global = (time.time() - t0_global) / 60.0
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (Total time: {elapsed_global:.1f}m)")
    print(f"{'='*70}")
    
    print("\nAll feature modes:")
    for res in sorted(results, key=lambda r: r["best_validation_rmse"]):
        env_status = "WITH Env" if "with_env" in res["feature_mode"] else "NO Env"
        print(f"  {res['feature_mode']:30s} [{env_status:10s}]: rmse={res['best_validation_rmse']:.6f}")

    print(f"\n🏆 BEST: {overall_best['feature_mode']} with rmse={overall_best_score:.6f}")
    
    # Check environment variable utility
    with_env_results = [r for r in results if "with_env" in r["feature_mode"]]
    no_env_results = [r for r in results if "no_env" in r["feature_mode"]]
    
    if with_env_results and no_env_results:
        best_with_env = min(with_env_results, key=lambda r: r["best_validation_rmse"])
        best_no_env = min(no_env_results, key=lambda r: r["best_validation_rmse"])
        improvement = ((best_no_env["best_validation_rmse"] - best_with_env["best_validation_rmse"]) 
                       / best_no_env["best_validation_rmse"] * 100)
        print(f"\n📊 Environment variable utility:")
        print(f"  Best WITH Env:  {best_with_env['feature_mode']:28s} rmse={best_with_env['best_validation_rmse']:.6f}")
        print(f"  Best NO Env:    {best_no_env['feature_mode']:28s} rmse={best_no_env['best_validation_rmse']:.6f}")
        if improvement > 0:
            print(f"  → Env is USEFUL: {improvement:.2f}% improvement with Env")
        else:
            print(f"  → Env has MINIMAL impact (difference: {abs(improvement):.2f}%)")

    # Save metadata
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_splits": args.n_splits,
        "n_trials": args.n_trials,
        "random_state": args.random_state,
        "total_elapsed_minutes": elapsed_global,
        "overall_best_feature_mode": overall_best["feature_mode"],
        "overall_best_validation_rmse": overall_best_score,
        "all_results": [
            {
                "feature_mode": r["feature_mode"],
                "best_trial_number": r["best_trial_number"],
                "best_validation_rmse": r["best_validation_rmse"],
                "elapsed_minutes": r["elapsed_minutes"],
                "best_params": r["best_params"],
            }
            for r in results
        ],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata_path = output_dir / f"rf_xgb_stacking_optuna_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")

    # Generate submission from best model
    if not args.skip_submission:
        test_ids = data.x_test.index if hasattr(data.x_test, "index") else np.arange(len(data.x_test))
        submission = build_submission_frame(test_ids, overall_best["test_predictions"])

        sub_path = output_dir / f"submission_rf_xgb_stacking_{timestamp}.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    if args.save_model:
        import joblib

        model_path = output_dir / f"models_rf_xgb_stacking_{timestamp}.joblib"
        models_dict = {
            "rf": overall_best["rf_model"],
            "xgb": overall_best["xgb_model"],
            "meta": overall_best["meta_model"],
            "feature_mode": overall_best["feature_mode"],
        }
        joblib.dump(models_dict, model_path)
        print(f"Models saved to {model_path}")


if __name__ == "__main__":
    main()
