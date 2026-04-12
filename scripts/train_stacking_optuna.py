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
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split

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
class StackResult:
    feature_mode: str
    use_env: bool
    base_model_names: list[str]
    meta_name: str
    base_params: dict[str, dict[str, float | int]]
    meta_params: dict[str, float | int | bool]
    weights: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot stacking ensemble with Optuna, combining RF, ExtraTrees, boosting and XGBoost when available."
    )
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, and y_train.csv.")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna search before final training.")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode: fewer trials and lighter folds.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of folds for OOF stacking.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits and models.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Validation fraction for diagnostics.")
    parser.add_argument("--output-dir", default="artifacts_stacking_optuna", help="Directory for reports and outputs.")
    parser.add_argument("--submission-prefix", default="stacking_optuna", help="Prefix for generated files.")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="raw",
        help="Feature recipe to use. *_no_env variants drop the Env column.",
    )
    parser.add_argument("--skip-submission", action="store_true", help="Run diagnostics only and skip generating the CSV.")
    parser.add_argument("--save-model", action="store_true", help="Save the fitted stacking bundle to joblib.")
    args = parser.parse_args()

    if args.pilot and not args.optimize:
        args.optimize = True
    if args.pilot and args.n_trials == 20:
        args.n_trials = 5
    if args.pilot and args.n_folds == 3:
        args.n_folds = 3
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be between 0 and 1.")
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
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


def get_use_env(feature_mode: str) -> bool:
    return feature_mode not in {"raw_no_env", "engineered_no_env", "env_focus_no_env"}


def make_base_model_factories(trial: optuna.Trial | None, random_state: int) -> dict[str, callable]:
    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 150, 450, step=50) if trial else 250,
        "max_depth": trial.suggest_int("rf_max_depth", 10, 28, step=2) if trial else 22,
        "min_samples_split": trial.suggest_float("rf_min_samples_split", 0.001, 0.02, log=True) if trial else 0.005,
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 5, 50, step=5) if trial else 20,
        "max_features": trial.suggest_float("rf_max_features", 0.5, 1.0, step=0.1) if trial else 0.7,
        "random_state": random_state,
        "n_jobs": -1,
    }
    et_params = {
        "n_estimators": trial.suggest_int("et_n_estimators", 150, 500, step=50) if trial else 300,
        "max_depth": trial.suggest_int("et_max_depth", 10, 28, step=2) if trial else 24,
        "min_samples_split": trial.suggest_float("et_min_samples_split", 0.001, 0.02, log=True) if trial else 0.003,
        "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 5, 50, step=5) if trial else 15,
        "max_features": trial.suggest_float("et_max_features", 0.5, 1.0, step=0.1) if trial else 0.7,
        "random_state": random_state,
        "n_jobs": -1,
    }
    gb_params = {
        "n_estimators": trial.suggest_int("gb_n_estimators", 80, 250, step=25) if trial else 150,
        "learning_rate": trial.suggest_float("gb_learning_rate", 0.02, 0.15, log=True) if trial else 0.05,
        "max_depth": trial.suggest_int("gb_max_depth", 2, 4) if trial else 3,
        "subsample": trial.suggest_float("gb_subsample", 0.6, 1.0, step=0.1) if trial else 0.8,
        "random_state": random_state,
    }
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300, step=25) if trial else 240,
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 8) if trial else 5,
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.15, log=True) if trial else 0.05,
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0, step=0.1) if trial else 0.8,
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0, step=0.1) if trial else 0.8,
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 1.0, log=True) if trial else 1e-4,
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 10.0, log=True) if trial else 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }

    factories: dict[str, callable] = {
        "random_forest": lambda: RandomForestRegressor(**rf_params),
        "extra_trees": lambda: ExtraTreesRegressor(**et_params),
        "boosting": lambda: MultiOutputRegressor(GradientBoostingRegressor(**gb_params)),
    }
    if XGBOOST_AVAILABLE:
        factories["xgboost"] = lambda: MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=-1)
    return factories


def make_meta_model(trial: optuna.Trial | None) -> Ridge:
    alpha = trial.suggest_float("meta_alpha", 1e-3, 100.0, log=True) if trial else 1.0
    fit_intercept = trial.suggest_categorical("meta_fit_intercept", [True, False]) if trial else True
    return Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)


def make_weights(trial: optuna.Trial | None, model_names: list[str]) -> dict[str, float]:
    if trial is None:
        return {name: 1.0 / len(model_names) for name in model_names}
    raw_weights = [trial.suggest_float(f"weight_{name}", 0.1, 3.0) for name in model_names]
    total = float(sum(raw_weights))
    return {name: float(value / total) for name, value in zip(model_names, raw_weights)}


def oof_stack_predictions(
    X: pd.DataFrame,
    y: pd.DataFrame,
    base_factories: dict[str, callable],
    weights: dict[str, float],
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target_columns = list(y.columns)
    oof_blocks = {
        name: np.zeros((len(X), len(target_columns)), dtype=float)
        for name in base_factories.keys()
    }
    fitted_models: dict[str, object] = {}

    for train_idx, valid_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]

        for name, factory in base_factories.items():
            model = factory()
            model.fit(X_train_fold, y_train_fold)
            oof_blocks[name][valid_idx] = np.asarray(model.predict(X_valid_fold), dtype=float)

    stacked_oof = np.concatenate([weights[name] * oof_blocks[name] for name in base_factories.keys()], axis=1)
    meta_model = make_meta_model(None)
    meta_model.fit(stacked_oof, y)

    for name, factory in base_factories.items():
        fitted_models[name] = factory()
        fitted_models[name].fit(X, y)

    return pd.DataFrame(stacked_oof, index=X.index), {"base_models": fitted_models, "meta_model": meta_model}


def make_stacked_features(prediction_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    stacked = [weights[name] * prediction_map[name] for name in prediction_map.keys()]
    return np.concatenate(stacked, axis=1)


def predict_from_bundle(bundle: dict[str, object], X: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    base_models: dict[str, object] = bundle["base_models"]
    meta_model: Ridge = bundle["meta_model"]
    prediction_map: dict[str, np.ndarray] = {}
    for name, model in base_models.items():
        prediction_map[name] = np.asarray(model.predict(X), dtype=float)
    stacked = make_stacked_features(prediction_map, weights)
    return np.asarray(meta_model.predict(stacked), dtype=float)


def optimize_stack(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    random_state: int,
    n_trials: int,
    n_folds: int,
) -> StackResult:
    if not OPTUNA_AVAILABLE:
        base_factories = make_base_model_factories(None, random_state=random_state)
        base_model_names = list(base_factories.keys())
        weights = {name: 1.0 / len(base_model_names) for name in base_model_names}
        return StackResult("raw", True, base_model_names, "ridge", {}, {}, weights)

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        feature_mode = trial.suggest_categorical("feature_mode", FEATURE_MODES)
        use_env = get_use_env(feature_mode)
        X_fit_variant = build_features(X_fit, feature_mode)
        X_valid_variant = build_features(X_valid, feature_mode)
        if not use_env and "Env" in X_fit_variant.columns:
            X_fit_variant = X_fit_variant.drop(columns=["Env"])
            X_valid_variant = X_valid_variant.drop(columns=["Env"])

        base_factories = make_base_model_factories(trial, random_state=random_state)
        model_names = list(base_factories.keys())
        weights = make_weights(trial, model_names)

        _, bundle = oof_stack_predictions(
            X_fit_variant,
            y_fit,
            base_factories,
            weights,
            n_splits=n_folds,
            random_state=random_state,
        )

        valid_prediction_map: dict[str, np.ndarray] = {}
        for name, model in bundle["base_models"].items():
            valid_prediction_map[name] = np.asarray(model.predict(X_valid_variant), dtype=float)

        stacked_valid = make_stacked_features(valid_prediction_map, weights)
        meta_model: Ridge = bundle["meta_model"]
        valid_pred = np.asarray(meta_model.predict(stacked_valid), dtype=float)
        valid_df = pd.DataFrame(valid_pred, columns=train_target_columns, index=X_valid.index)
        valid_full = enforce_known_constraints(valid_df)
        y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))
        rmse = competition_rmse(y_valid_full, valid_full)
        trial.report(rmse, step=0)
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=0),
        study_name="stacking_optuna",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_feature_mode = best.params["feature_mode"]
    base_factories = make_base_model_factories(best, random_state=random_state)
    base_model_names = list(base_factories.keys())
    weights = make_weights(best, base_model_names)
    meta_name = "ridge"
    base_params = {name: {} for name in base_model_names}
    meta_params = {"alpha": best.params.get("meta_alpha", 1.0), "fit_intercept": best.params.get("meta_fit_intercept", True)}
    return StackResult(best_feature_mode, get_use_env(best_feature_mode), base_model_names, meta_name, base_params, meta_params, weights)


def summarize_feature_importance(model: object, feature_columns: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
        return {feature: float(value) for feature, value in importance.items()}
    return {}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    if args.optimize:
        result = optimize_stack(
            data.x_train,
            y_train_full,
            train_target_columns,
            random_state=args.random_state,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
        )
        feature_mode = result.feature_mode
        use_env = result.use_env
        weights = result.weights
    else:
        feature_mode = args.feature_mode
        use_env = get_use_env(feature_mode)
        model_names = ["random_forest", "extra_trees", "boosting"] + (["xgboost"] if XGBOOST_AVAILABLE else [])
        weights = {name: 1.0 / len(model_names) for name in model_names}
        result = StackResult(feature_mode, use_env, model_names, "ridge", {}, {"alpha": 1.0, "fit_intercept": True}, weights)

    X_train = build_features(data.x_train, feature_mode)
    X_test = build_features(data.x_test, feature_mode)
    if not use_env and "Env" in X_train.columns:
        X_train = X_train.drop(columns=["Env"])
        X_test = X_test.drop(columns=["Env"])

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train_full,
        test_size=args.holdout_fraction,
        random_state=args.random_state,
    )

    base_factories = make_base_model_factories(None, random_state=args.random_state)
    base_model_names = list(base_factories.keys())
    weights = {name: weights.get(name, 1.0 / len(base_model_names)) for name in base_model_names}
    total_weight = sum(weights.values())
    weights = {name: value / total_weight for name, value in weights.items()}

    oof_stack, bundle = oof_stack_predictions(
        X_fit,
        y_fit,
        base_factories,
        weights,
        n_splits=args.n_folds,
        random_state=args.random_state,
    )
    meta_model: Ridge = bundle["meta_model"]
    stack_feature_names = [f"{name}__d{i:02d}" for name in base_model_names for i in range(1, 24) if i != 15]
    oof_meta_pred = np.asarray(meta_model.predict(oof_stack), dtype=float)
    oof_meta_df = pd.DataFrame(oof_meta_pred, columns=train_target_columns, index=X_fit.index)
    oof_full = enforce_known_constraints(oof_meta_df)
    y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
    train_rmse = competition_rmse(y_fit_full, oof_full)

    valid_prediction_map: dict[str, np.ndarray] = {}
    for name, model in bundle["base_models"].items():
        valid_prediction_map[name] = np.asarray(model.predict(X_valid), dtype=float)
    stacked_valid = make_stacked_features(valid_prediction_map, weights)
    valid_meta_pred = np.asarray(meta_model.predict(stacked_valid), dtype=float)
    valid_meta_df = pd.DataFrame(valid_meta_pred, columns=train_target_columns, index=X_valid.index)
    valid_full = enforce_known_constraints(valid_meta_df)
    y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))
    validation_rmse = competition_rmse(y_valid_full, valid_full)

    final_base_factories = make_base_model_factories(None, random_state=args.random_state)
    final_base_models: dict[str, object] = {}
    final_prediction_map_test: dict[str, np.ndarray] = {}
    final_prediction_map_train: dict[str, np.ndarray] = {}
    for name, factory in final_base_factories.items():
        model = factory()
        model.fit(X_train, y_train_full)
        final_base_models[name] = model
        final_prediction_map_test[name] = np.asarray(model.predict(X_test), dtype=float)
        final_prediction_map_train[name] = np.asarray(model.predict(X_train), dtype=float)

    final_train_stack = make_stacked_features(final_prediction_map_train, weights)
    final_meta_model = Ridge(alpha=result.meta_params.get("alpha", 1.0), fit_intercept=result.meta_params.get("fit_intercept", True), random_state=42)
    final_meta_model.fit(final_train_stack, y_train_full)

    blended_train_pred = np.asarray(final_meta_model.predict(final_train_stack), dtype=float)
    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train_full.assign(d15=0.0)),
        enforce_known_constraints(pd.DataFrame(blended_train_pred, columns=train_target_columns, index=X_train.index)),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        final_test_stack = make_stacked_features(final_prediction_map_test, weights)
        test_pred = np.asarray(final_meta_model.predict(final_test_stack), dtype=float)
        final_predictions = enforce_known_constraints(pd.DataFrame(test_pred, columns=train_target_columns, index=X_test.index))
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
                "base_models": final_base_models,
                "meta_model": final_meta_model,
                "train_target_columns": train_target_columns,
                "feature_columns": list(X_train.columns),
                "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
                "constant_targets": schema.constant_targets,
            },
            model_file,
        )
        model_path = str(model_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "stacking_multioutput_regression",
        "data_dir": str(data_dir),
        "feature_mode": feature_mode,
        "use_env": use_env,
        "base_model_names": base_model_names,
        "meta_model": "ridge",
        "xgboost_available": XGBOOST_AVAILABLE,
        "pilot_mode": bool(args.pilot),
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
            "n_folds": args.n_folds,
            "feature_modes": FEATURE_MODES,
        },
        "stacking_weights": weights,
        "diagnostic_scores": {
            "train_rmse": float(train_rmse),
            "validation_rmse": float(validation_rmse),
            "full_train_rmse": float(full_train_rmse),
        },
        "feature_importance": {
            name: summarize_feature_importance(model, list(X_train.columns)) for name, model in final_base_models.items() if hasattr(model, "feature_importances_")
        },
        "model_path": model_path,
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()