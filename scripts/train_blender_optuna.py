from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

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
    engineer_env_focus_features,
    engineer_features,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


FEATURE_MODES = ["raw", "raw_no_env", "engineered", "engineered_no_env", "env_focus", "env_focus_no_env"]


@dataclass(frozen=True)
class DatasetBundle:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    feature_mode: str
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a blender ensemble with Optuna over RandomForest, ExtraTrees and XGBoost/boosting models."
    )
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, and y_train.csv.")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna search before final training.")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits and models.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Validation fraction for diagnostics.")
    parser.add_argument("--output-dir", default="artifacts_blender_optuna", help="Directory for reports and outputs.")
    parser.add_argument("--submission-prefix", default="blender_optuna", help="Prefix for generated files.")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="raw",
        help="Feature recipe to use. *_no_env variants drop the Env column.",
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Run diagnostics only and skip generating the final CSV.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the fitted blender to a joblib file for later test-only inference.",
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
    return constrained[[f"d{i:02d}" for i in range(1, 24)]]


def build_features(features: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "raw":
        return raw_features(features)
    if mode == "raw_no_env":
        frame = raw_features(features)
        return frame.drop(columns=["Env"])
    if mode == "engineered":
        return engineer_features(features)
    if mode == "engineered_no_env":
        frame = engineer_features(features)
        return frame.drop(columns=["Env"])
    if mode == "env_focus":
        return engineer_env_focus_features(features)
    if mode == "env_focus_no_env":
        frame = engineer_env_focus_features(features)
        return frame.drop(columns=["Env"])
    raise ValueError(f"Unknown feature mode: {mode}")


def make_base_models(trial: optuna.Trial | None, random_state: int) -> dict[str, object]:
    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 200, 600, step=100) if trial else 400,
        "max_depth": trial.suggest_int("rf_max_depth", 12, 28, step=2) if trial else 22,
        "min_samples_split": trial.suggest_float("rf_min_samples_split", 0.001, 0.02, log=True) if trial else 0.005,
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 5, 60, step=5) if trial else 20,
        "max_features": trial.suggest_float("rf_max_features", 0.5, 1.0, step=0.1) if trial else 0.7,
        "random_state": random_state,
        "n_jobs": -1,
    }
    et_params = {
        "n_estimators": trial.suggest_int("et_n_estimators", 200, 700, step=100) if trial else 500,
        "max_depth": trial.suggest_int("et_max_depth", 12, 30, step=2) if trial else 24,
        "min_samples_split": trial.suggest_float("et_min_samples_split", 0.001, 0.02, log=True) if trial else 0.003,
        "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 5, 60, step=5) if trial else 15,
        "max_features": trial.suggest_float("et_max_features", 0.5, 1.0, step=0.1) if trial else 0.7,
        "random_state": random_state,
        "n_jobs": -1,
    }

    models: dict[str, object] = {
        "random_forest": RandomForestRegressor(**rf_params),
        "extra_trees": ExtraTreesRegressor(**et_params),
    }

    if XGBOOST_AVAILABLE:
        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 700, step=100) if trial else 400,
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10) if trial else 6,
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True) if trial else 0.05,
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0, step=0.1) if trial else 0.8,
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0, step=0.1) if trial else 0.8,
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 1.0, log=True) if trial else 1e-4,
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 10.0, log=True) if trial else 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
        models["xgboost"] = MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=-1)

    return models


def make_weight_vector(trial: optuna.Trial | None, model_names: list[str]) -> dict[str, float]:
    if trial is None:
        return {name: 1.0 / len(model_names) for name in model_names}
    raw_weights = [trial.suggest_float(f"weight_{name}", 0.1, 3.0) for name in model_names]
    total = float(sum(raw_weights))
    return {name: float(value / total) for name, value in zip(model_names, raw_weights)}


def fit_predict_model(model: object, X_fit: pd.DataFrame, y_fit: pd.DataFrame, X_valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model.fit(X_fit, y_fit)
    fit_pred = np.asarray(model.predict(X_fit), dtype=float)
    valid_pred = np.asarray(model.predict(X_valid), dtype=float)
    return fit_pred, valid_pred


def blend_predictions(prediction_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = None
    for name, prediction in prediction_map.items():
        weighted = weights[name] * prediction
        blended = weighted if blended is None else blended + weighted
    return np.asarray(blended, dtype=float)


def optimize_blender(
    x_train_raw: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    random_state: int,
    n_trials: int,
) -> tuple[dict[str, float], str]:
    if not OPTUNA_AVAILABLE:
        print("Optuna n'est pas installé, retour sur les valeurs par défaut du blender.")
        return {"random_forest": 1 / 2, "extra_trees": 1 / 2}, "raw"

    X_fit_raw, X_valid_raw, y_fit, y_valid = train_test_split(
        x_train_raw,
        y_train,
        test_size=0.2,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        feature_mode = trial.suggest_categorical("feature_mode", FEATURE_MODES)
        X_fit_variant = build_features(X_fit_raw, feature_mode)
        X_valid_variant = build_features(X_valid_raw, feature_mode)

        models = make_base_models(trial, random_state=random_state)
        model_names = list(models.keys())
        weights = make_weight_vector(trial, model_names)

        valid_predictions: dict[str, np.ndarray] = {}
        for name, model in models.items():
            _, valid_pred = fit_predict_model(model, X_fit_variant, y_fit, X_valid_variant)
            valid_predictions[name] = valid_pred

        blended_valid = blend_predictions(valid_predictions, weights)
        blended_valid_df = pd.DataFrame(blended_valid, columns=train_target_columns, index=X_valid.index)
        blended_valid_full = enforce_known_constraints(blended_valid_df)
        y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

        rmse = competition_rmse(y_valid_full, blended_valid_full)
        trial.report(rmse, step=0)
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name="blender_optuna",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    feature_mode = best.params["feature_mode"]
    model_names = ["random_forest", "extra_trees"] + (["xgboost"] if XGBOOST_AVAILABLE else [])
    weights = make_weight_vector(best, model_names)
    print(f"Meilleur score Optuna: {best.value:.6f}")
    print(f"Meilleur feature_mode: {feature_mode}")
    print(f"Poids blender: {weights}")
    return weights, feature_mode


def fit_final_blender(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    feature_mode: str,
    random_state: int,
    weights: dict[str, float],
) -> tuple[dict[str, object], pd.DataFrame]:
    use_env = feature_mode not in {"raw_no_env", "engineered_no_env", "env_focus_no_env"}
    X_train_variant = X_train if use_env else X_train.drop(columns=["Env"])

    models = make_base_models(None, random_state=random_state)
    prediction_map: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_train_variant, y_train)
        prediction_map[name] = np.asarray(model.predict(X_train_variant), dtype=float)

    blended_train = blend_predictions(prediction_map, weights)
    blended_train_df = pd.DataFrame(blended_train, columns=train_target_columns, index=X_train.index)
    return models, blended_train_df


def summarize_feature_importance(models: dict[str, object], feature_columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
            summary[name] = {feature: float(value) for feature, value in importance.items()}
    return summary


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    y_train_full, train_target_columns = prepare_targets(data.y_train)

    if args.optimize:
        weights, feature_mode = optimize_blender(
            data.x_train,
            y_train_full,
            train_target_columns,
            random_state=args.random_state,
            n_trials=args.n_trials,
        )
    else:
        weights = {"random_forest": 0.45, "extra_trees": 0.35}
        if XGBOOST_AVAILABLE:
            weights["xgboost"] = 0.20
        feature_mode = args.feature_mode

    use_env = feature_mode not in {"raw_no_env", "engineered_no_env", "env_focus_no_env"}
    X_train = build_features(data.x_train, feature_mode)
    X_test = build_features(data.x_test, feature_mode)

    if not use_env and "Env" in X_train.columns:
        X_train = X_train.drop(columns=["Env"])
        X_test = X_test.drop(columns=["Env"])

    feature_columns = list(X_train.columns)

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train_full,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )

    models = make_base_models(None, random_state=args.random_state)
    fit_pred_map: dict[str, np.ndarray] = {}
    valid_pred_map: dict[str, np.ndarray] = {}
    for name, model in models.items():
        fit_pred_map[name], valid_pred_map[name] = fit_predict_model(model, X_fit, y_fit, X_valid)

    fit_blend = blend_predictions(fit_pred_map, weights)
    valid_blend = blend_predictions(valid_pred_map, weights)
    fit_full = enforce_known_constraints(pd.DataFrame(fit_blend, columns=train_target_columns, index=X_fit.index))
    valid_full = enforce_known_constraints(pd.DataFrame(valid_blend, columns=train_target_columns, index=X_valid.index))
    y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
    y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

    train_rmse = competition_rmse(y_fit_full, fit_full)
    validation_rmse = competition_rmse(y_valid_full, valid_full)

    final_models = make_base_models(None, random_state=args.random_state)
    final_train_pred_map: dict[str, np.ndarray] = {}
    final_test_pred_map: dict[str, np.ndarray] = {}
    for name, model in final_models.items():
        model.fit(X_train, y_train_full)
        final_train_pred_map[name] = np.asarray(model.predict(X_train), dtype=float)
        final_test_pred_map[name] = np.asarray(model.predict(X_test), dtype=float)

    blended_train = blend_predictions(final_train_pred_map, weights)
    blended_test = blend_predictions(final_test_pred_map, weights)

    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train_full.assign(d15=0.0)),
        enforce_known_constraints(pd.DataFrame(blended_train, columns=train_target_columns, index=X_train.index)),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        final_predictions = enforce_known_constraints(pd.DataFrame(blended_test, columns=train_target_columns, index=X_test.index))
        submission = build_submission_frame(data.x_test["ID"], final_predictions)
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    model_path = None
    if args.save_model:
        model_file = output_dir / f"{args.submission_prefix}_{timestamp}.joblib"
        joblib.dump(
            {
                "feature_mode": feature_mode,
                "use_env": use_env,
                "weights": weights,
                "models": final_models,
                "feature_columns": feature_columns,
                "train_target_columns": train_target_columns,
                "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
                "constant_targets": schema.constant_targets,
            },
            model_file,
        )
        model_path = str(model_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "blender_multioutput_regression",
        "data_dir": str(data_dir),
        "feature_mode": feature_mode,
        "use_env": use_env,
        "feature_count": int(X_train.shape[1]),
        "feature_columns": list(X_train.columns),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
        "constant_targets": schema.constant_targets,
        "target_strategy": {
            "trained_targets": train_target_columns,
            "d15_handling": "removed_from_training_and_reinserted_as_zero",
            "duplicate_groups_handling": "predictions_averaged_within_known_duplicate_groups_before_export",
        },
        "optuna": {
            "enabled": args.optimize,
            "n_trials": args.n_trials if args.optimize else 0,
            "xgboost_available": XGBOOST_AVAILABLE,
            "feature_modes": FEATURE_MODES,
        },
        "blender_weights": weights,
        "diagnostic_scores": {
            "train_rmse": float(train_rmse),
            "validation_rmse": float(validation_rmse),
            "full_train_rmse": float(full_train_rmse),
        },
        "feature_importance": summarize_feature_importance(final_models, list(X_train.columns)),
        "model_path": model_path,
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()