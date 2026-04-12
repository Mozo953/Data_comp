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
from sklearn.multioutput import MultiOutputRegressor

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, engineer_features, load_competition_data, raw_features  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF+XGB Optuna training with anti-overfit objective.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, y_train.csv.")
    parser.add_argument("--n-trials", type=int, default=12, help="Number of Optuna trials.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--overfit-penalty", type=float, default=0.8, help="Penalty multiplier on (val_rmse - train_rmse).")
    parser.add_argument("--output-dir", default="artifacts_rf_xgb_optuna_antioverfit")
    parser.add_argument("--submission-prefix", default="rf_xgb_optuna_antioverfit")
    parser.add_argument("--skip-submission", action="store_true")
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
    raw_train = raw_features(x_train)
    raw_test = raw_features(x_test)

    eng_train = engineer_features(x_train)
    eng_test = engineer_features(x_test)

    return {
        "raw_with_env": (raw_train, raw_test),
        "raw_no_env": (raw_train.drop(columns=["Env"]), raw_test.drop(columns=["Env"])),
        "engineered_no_env": (eng_train.drop(columns=["Env"]), eng_test.drop(columns=["Env"])),
    }


def fit_models_and_blend(
    X_fit: pd.DataFrame,
    y_fit: pd.DataFrame,
    X_eval: pd.DataFrame,
    rf_params: dict,
    xgb_params: dict,
    w_rf: float,
    random_state: int,
) -> np.ndarray:
    rf = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)
    rf.fit(X_fit, y_fit)
    pred_rf = np.asarray(rf.predict(X_eval), dtype=float)

    xgb = MultiOutputRegressor(
        XGBRegressor(
            **xgb_params,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        ),
        n_jobs=-1,
    )
    xgb.fit(X_fit, y_fit)
    pred_xgb = np.asarray(xgb.predict(X_eval), dtype=float)

    return w_rf * pred_rf + (1.0 - w_rf) * pred_xgb


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    y_train_full, train_target_columns = prepare_targets(data.y_train)
    feature_sets = build_feature_sets(data.x_train, data.x_test)

    # Fixed split for robust trial comparisons.
    fit_idx, valid_idx = train_test_split(
        y_train_full.index,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        feature_mode = trial.suggest_categorical("feature_mode", list(feature_sets.keys()))
        X_train_all, _ = feature_sets[feature_mode]
        X_fit = X_train_all.loc[fit_idx]
        X_valid = X_train_all.loc[valid_idx]
        y_fit = y_train_full.loc[fit_idx]
        y_valid = y_train_full.loc[valid_idx]

        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 180, 420, step=40),
            "max_depth": trial.suggest_int("rf_max_depth", 14, 24),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 10, 45),
            "min_samples_split": trial.suggest_float("rf_min_samples_split", 0.002, 0.012),
            "max_features": trial.suggest_float("rf_max_features", 0.6, 0.9),
        }

        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 160, 360, step=40),
            "max_depth": trial.suggest_int("xgb_max_depth", 4, 8),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.03, 0.12),
            "subsample": trial.suggest_float("xgb_subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.65, 0.9),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 0.02, log=True),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.5, 3.0),
        }

        w_rf = trial.suggest_float("weight_rf", 0.2, 0.8)

        pred_fit = fit_models_and_blend(X_fit, y_fit, X_fit, rf_params, xgb_params, w_rf, args.random_state)
        pred_valid = fit_models_and_blend(X_fit, y_fit, X_valid, rf_params, xgb_params, w_rf, args.random_state)

        fit_df = pd.DataFrame(pred_fit, columns=train_target_columns, index=X_fit.index)
        valid_df = pd.DataFrame(pred_valid, columns=train_target_columns, index=X_valid.index)
        fit_full = enforce_known_constraints(fit_df)
        valid_full = enforce_known_constraints(valid_df)
        y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
        y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

        train_rmse = competition_rmse(y_fit_full, fit_full)
        valid_rmse = competition_rmse(y_valid_full, valid_full)

        overfit_gap = max(0.0, valid_rmse - train_rmse)
        score = valid_rmse + args.overfit_penalty * overfit_gap

        trial.set_user_attr("train_rmse", float(train_rmse))
        trial.set_user_attr("valid_rmse", float(valid_rmse))
        trial.set_user_attr("overfit_gap", float(overfit_gap))
        return float(score)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=0),
        study_name="rf_xgb_antioverfit",
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_feature_mode = best.params["feature_mode"]
    X_train, X_test = feature_sets[best_feature_mode]

    rf_params = {
        "n_estimators": int(best.params["rf_n_estimators"]),
        "max_depth": int(best.params["rf_max_depth"]),
        "min_samples_leaf": int(best.params["rf_min_samples_leaf"]),
        "min_samples_split": float(best.params["rf_min_samples_split"]),
        "max_features": float(best.params["rf_max_features"]),
    }
    xgb_params = {
        "n_estimators": int(best.params["xgb_n_estimators"]),
        "max_depth": int(best.params["xgb_max_depth"]),
        "learning_rate": float(best.params["xgb_learning_rate"]),
        "subsample": float(best.params["xgb_subsample"]),
        "colsample_bytree": float(best.params["xgb_colsample_bytree"]),
        "reg_alpha": float(best.params["xgb_reg_alpha"]),
        "reg_lambda": float(best.params["xgb_reg_lambda"]),
    }
    w_rf = float(best.params["weight_rf"])

    fit_idx, valid_idx = train_test_split(
        y_train_full.index,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )
    X_fit = X_train.loc[fit_idx]
    X_valid = X_train.loc[valid_idx]
    y_fit = y_train_full.loc[fit_idx]
    y_valid = y_train_full.loc[valid_idx]

    pred_valid = fit_models_and_blend(X_fit, y_fit, X_valid, rf_params, xgb_params, w_rf, args.random_state)
    valid_df = pd.DataFrame(pred_valid, columns=train_target_columns, index=X_valid.index)
    valid_rmse = competition_rmse(
        enforce_known_constraints(y_valid.assign(d15=0.0)),
        enforce_known_constraints(valid_df),
    )

    pred_train_full = fit_models_and_blend(X_train, y_train_full, X_train, rf_params, xgb_params, w_rf, args.random_state)
    pred_test_full = fit_models_and_blend(X_train, y_train_full, X_test, rf_params, xgb_params, w_rf, args.random_state)

    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train_full.assign(d15=0.0)),
        enforce_known_constraints(pd.DataFrame(pred_train_full, columns=train_target_columns, index=X_train.index)),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        test_pred_df = pd.DataFrame(pred_test_full, columns=train_target_columns, index=X_test.index)
        submission = build_submission_frame(data.x_test["ID"], enforce_known_constraints(test_pred_df))
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "rf_xgb_optuna_antioverfit",
        "data_dir": str(data_dir),
        "n_trials": args.n_trials,
        "holdout_fraction": args.holdout_fraction,
        "overfit_penalty": args.overfit_penalty,
        "best_feature_mode": best_feature_mode,
        "best_objective": float(best.value),
        "best_trial": int(best.number),
        "best_trial_metrics": {
            "train_rmse": float(best.user_attrs.get("train_rmse", np.nan)),
            "valid_rmse": float(best.user_attrs.get("valid_rmse", np.nan)),
            "overfit_gap": float(best.user_attrs.get("overfit_gap", np.nan)),
        },
        "selected_params": {
            "rf": rf_params,
            "xgb": xgb_params,
            "weight_rf": w_rf,
            "weight_xgb": 1.0 - w_rf,
        },
        "diagnostic_scores": {
            "validation_rmse": float(valid_rmse),
            "full_train_rmse": float(full_train_rmse),
        },
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
