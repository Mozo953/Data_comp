from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    load_modeling_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402

VERBOSE = False


DEFAULT_MODEL_SPECS = [
    {
        "name": "et_allpool_3",
        "role": "local_deep",
        "seed_offset": 41,
        "bootstrap_choices": [False],
        "n_estimators_min": 260,
        "n_estimators_max": 420,
        "min_samples_split_min": 8,
        "min_samples_split_max": 16,
        "max_depth_min": 15,
        "max_depth_max": 18,
        "max_features_min": 0.12,
        "max_features_max": 0.28,
        "min_samples_leaf_min": 3,
        "min_samples_leaf_max": 5,
        "max_samples_min": None,
        "max_samples_max": None,
    },
]

FIXED_ET_PARAMS = {
    "et_allpool_3": {
        "n_estimators": 300,
        "max_depth": 17,
        "min_samples_split": 13,
        "min_samples_leaf": 5,
        "max_features": 0.2523786605336733,
        "bootstrap": False,
        "random_state": 83,
        "n_jobs": -1,
        "max_samples": None,
    },
}

FIXED_XGB_RAW_COLUMNS = [
    "Env",
    "X12",
    "X13",
    "X14",
    "X15",
    "X4",
    "X5",
    "X6",
    "X7",
    "Z",
    "Y1",
    "Y2",
    "Y3",
]

FIXED_XGB_PARAMS = {
    "n_estimators": 900,
    "max_depth": 7,
    "learning_rate": 0.07243678751068877,
    "subsample": 0.866081532578582,
    "colsample_bytree": 0.7880215372104394,
    "min_child_weight": 18.863695617192224,
    "reg_alpha": 0.04513481969616506,
    "reg_lambda": 0.013208356641814422,
    "gamma": 0.03882400016731788,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    seed_offset: int
    bootstrap_choices: list[bool]
    n_estimators_min: int
    n_estimators_max: int
    min_samples_split_min: int
    min_samples_split_max: int
    max_depth_min: int
    max_depth_max: int
    max_features_min: float
    max_features_max: float
    min_samples_leaf_min: int
    min_samples_leaf_max: int
    max_samples_min: float | None
    max_samples_max: float | None


@dataclass(frozen=True)
class FeaturePreprocessor:
    all_feature_columns: list[str]
    selected_columns: list[str]
    dropped_constant_columns: list[str]
    total_feature_count: int

    def transform(self, features: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
        expanded = build_global_feature_pool(features, ratio_eps=ratio_eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class CalibrationParams:
    slope: float
    intercept: float
    shrinkage: float
    target_mean: float


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_verbose(message: str) -> None:
    if VERBOSE:
        log_progress(f"VERBOSE: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse fixed et_allpool_3 and raw XGB configs, recompute CV=3 OOFs, then apply a fixed 0.7/0.3 blend and calibration."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/17_fixed_et3_xgbraw_blender")
    parser.add_argument("--submission-prefix", default="fixed_et3_xgbraw_blender")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--eval-cv-folds", type=int, default=3)
    parser.add_argument("--et-weight", type=float, default=0.7)
    parser.add_argument("--xgb-weight", type=float, default=0.3)
    parser.add_argument("--calibration-optuna-trials", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-n-jobs", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.eval_cv_folds < 2:
        raise ValueError("--eval-cv-folds must be >= 2.")
    if args.et_weight < 0.0 or args.xgb_weight < 0.0:
        raise ValueError("--et-weight and --xgb-weight must be >= 0.")
    weight_sum = args.et_weight + args.xgb_weight
    if weight_sum <= 0.0:
        raise ValueError("--et-weight + --xgb-weight must be > 0.")
    args.et_weight = float(args.et_weight / weight_sum)
    args.xgb_weight = float(args.xgb_weight / weight_sum)
    if args.calibration_optuna_trials < 1:
        raise ValueError("--calibration-optuna-trials must be >= 1.")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs must be != 0.")
    if args.xgb_n_jobs == 0:
        raise ValueError("--xgb-n-jobs must be != 0.")
    if args.max_train_rows is not None and args.max_train_rows < 200:
        raise ValueError("--max-train-rows must be >= 200 when provided.")
    if args.max_test_rows is not None and args.max_test_rows < 1:
        raise ValueError("--max-test-rows must be >= 1 when provided.")

    return args


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=np.float32)
    sign = np.where(raw >= 0.0, 1.0, -1.0).astype(np.float32)
    adjusted = raw + (sign * np.float32(eps))
    tiny = np.abs(raw) < eps
    adjusted[tiny] = np.where(raw[tiny] >= 0.0, eps, -eps)
    return adjusted


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def _ordered_ratio_pairs(columns: list[str]) -> list[tuple[str, str]]:
    ordered_pairs: list[tuple[str, str]] = []
    for left in columns:
        for right in columns:
            if left == right:
                continue
            ordered_pairs.append((left, right))
    return ordered_pairs


def build_feature_family(
    features: pd.DataFrame,
    *,
    family: str,
    ratio_eps: float,
    ratio_limit: int,
) -> pd.DataFrame:
    base = raw_features(features).astype(np.float32)
    sensor_columns = [column for column in base.columns if column != "Env"]
    sensor_values = base[sensor_columns].to_numpy(dtype=np.float32)

    row_mean = sensor_values.mean(axis=1)
    row_std = sensor_values.std(axis=1)
    row_p10 = np.percentile(sensor_values, 10, axis=1)
    row_p25 = np.percentile(sensor_values, 25, axis=1)
    row_p50 = np.percentile(sensor_values, 50, axis=1)
    row_p75 = np.percentile(sensor_values, 75, axis=1)
    row_p90 = np.percentile(sensor_values, 90, axis=1)
    row_iqr = row_p75 - row_p25
    row_mad = np.median(np.abs(sensor_values - row_p50[:, None]), axis=1)
    row_l1 = np.linalg.norm(sensor_values, ord=1, axis=1)
    row_l2 = np.linalg.norm(sensor_values, ord=2, axis=1)

    payload: dict[str, np.ndarray | pd.Series] = {"Env": base["Env"].to_numpy(dtype=np.float32)}
    for column in sensor_columns:
        payload[column] = base[column].to_numpy(dtype=np.float32)

    payload["row_mean"] = row_mean
    payload["row_std"] = row_std
    payload["row_p10"] = row_p10
    payload["row_p25"] = row_p25
    payload["row_p50"] = row_p50
    payload["row_p75"] = row_p75
    payload["row_p90"] = row_p90
    payload["row_iqr"] = row_iqr
    payload["row_mad"] = row_mad
    payload["row_l1"] = row_l1
    payload["row_l2"] = row_l2

    block_a = ["X12", "X13", "X14", "X15"]
    block_b = ["X4", "X5", "X6", "X7"]
    block_support = ["Y1", "Y2", "Y3", "Z"]

    block_a_mean = base[block_a].mean(axis=1).astype(np.float32)
    block_b_mean = base[block_b].mean(axis=1).astype(np.float32)
    support_mean = base[block_support].mean(axis=1).astype(np.float32)

    payload["block_a_mean"] = block_a_mean.to_numpy(dtype=np.float32)
    payload["block_b_mean"] = block_b_mean.to_numpy(dtype=np.float32)
    payload["support_mean"] = support_mean.to_numpy(dtype=np.float32)
    payload["block_gap"] = (block_a_mean - block_b_mean).to_numpy(dtype=np.float32)
    payload["support_gap"] = (support_mean - base["Env"]).to_numpy(dtype=np.float32)
    payload["env_times_row_mean"] = (base["Env"] * row_mean).to_numpy(dtype=np.float32)

    if "logs" in family:
        for column in sensor_columns:
            payload[f"log_{column}"] = _signed_log1p(base[column].to_numpy(dtype=np.float32))
        payload["log_row_l1"] = _signed_log1p(row_l1)
        payload["log_row_l2"] = _signed_log1p(row_l2)

    if "ratios" in family:
        ratio_pairs = _ordered_ratio_pairs(sensor_columns)
        if ratio_limit is not None and ratio_limit > 0:
            ratio_pairs = ratio_pairs[:ratio_limit]
        for left, right in ratio_pairs:
            numerator = base[left].to_numpy(dtype=np.float32)
            denominator = _safe_denominator(base[right], ratio_eps)
            payload[f"{left}_over_{right}"] = (numerator / denominator).astype(np.float32)

        row_mean_den = np.where(np.abs(row_mean) < ratio_eps, ratio_eps, row_mean).astype(np.float32)
        for column in sensor_columns:
            payload[f"{column}_over_row_mean"] = (
                base[column].to_numpy(dtype=np.float32) / row_mean_den
            ).astype(np.float32)

    if family.endswith("_block"):
        for column in block_a:
            payload[f"{column}_minus_block_a_mean"] = (base[column] - block_a_mean).to_numpy(dtype=np.float32)
        for column in block_b:
            payload[f"{column}_minus_block_b_mean"] = (base[column] - block_b_mean).to_numpy(dtype=np.float32)
        for column in block_support:
            payload[f"{column}_minus_support_mean"] = (base[column] - support_mean).to_numpy(dtype=np.float32)
        payload["block_a_over_block_b"] = (
            block_a_mean.to_numpy(dtype=np.float32)
            / np.where(np.abs(block_b_mean.to_numpy(dtype=np.float32)) < ratio_eps, ratio_eps, block_b_mean)
        ).astype(np.float32)
        payload["support_over_block_a"] = (
            support_mean.to_numpy(dtype=np.float32)
            / np.where(np.abs(block_a_mean.to_numpy(dtype=np.float32)) < ratio_eps, ratio_eps, block_a_mean)
        ).astype(np.float32)

    return pd.DataFrame(payload, index=base.index).astype(np.float32)


def build_global_feature_pool(features: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
    return build_feature_family(
        features,
        family="stats_logs_ratios_block",
        ratio_eps=ratio_eps,
        ratio_limit=-1,
    )


def fit_feature_preprocessor(
    X_train: pd.DataFrame,
    *,
    ratio_eps: float,
) -> FeaturePreprocessor:
    expanded = build_global_feature_pool(X_train, ratio_eps=ratio_eps)
    total_feature_count = int(expanded.shape[1])

    constant_columns = [column for column in expanded.columns if expanded[column].nunique(dropna=False) <= 1]
    if constant_columns:
        expanded = expanded.drop(columns=constant_columns)

    selected_columns = list(expanded.columns)

    return FeaturePreprocessor(
        all_feature_columns=selected_columns,
        selected_columns=selected_columns,
        dropped_constant_columns=constant_columns,
        total_feature_count=total_feature_count,
    )


def suggest_et_params(trial: optuna.Trial, spec: ModelSpec, *, random_state: int, n_jobs: int) -> dict:
    bootstrap_choices = list(spec.bootstrap_choices)
    bootstrap = (
        bootstrap_choices[0]
        if len(bootstrap_choices) == 1
        else trial.suggest_categorical("bootstrap", bootstrap_choices)
    )
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 180, 340, step=40),
        "max_depth": trial.suggest_int("max_depth", spec.max_depth_min, spec.max_depth_max),
        "min_samples_split": trial.suggest_int("min_samples_split", 8, 24, step=4),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            spec.min_samples_leaf_min,
            spec.min_samples_leaf_max,
        ),
        "max_features": trial.suggest_float("max_features", spec.max_features_min, spec.max_features_max),
        "bootstrap": bootstrap,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    params["max_samples"] = trial.suggest_float("max_samples", 0.65, 0.95) if bootstrap else None
    return params


def make_et_model(params: dict) -> ExtraTreesRegressor:
    clean = dict(params)
    if not clean.get("bootstrap", False):
        clean["max_samples"] = None
    return ExtraTreesRegressor(**clean)


def evaluate_model_cv(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    spec: ModelSpec,
    params: dict,
    *,
    ratio_eps: float,
    n_splits: int,
    random_state: int,
    trial: optuna.Trial | None = None,
    collect_predictions: bool = False,
) -> dict:
    log_progress(f"{spec.name}: starting CV evaluation with {n_splits} folds")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []
    fold_reports: list[dict] = []

    oof_predictions = None
    if collect_predictions:
        oof_predictions = pd.DataFrame(
            np.zeros((len(X_train_raw), y_train_model.shape[1]), dtype=np.float32),
            columns=y_train_model.columns,
            index=X_train_raw.index,
        )

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        log_progress(f"{spec.name}: CV fold {fold_idx}/{n_splits} feature prep")
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        preprocessor = fit_feature_preprocessor(
            X_fit,
            ratio_eps=ratio_eps,
        )
        X_fit_model = preprocessor.transform(X_fit, ratio_eps=ratio_eps)
        X_valid_model = preprocessor.transform(X_valid, ratio_eps=ratio_eps)

        log_progress(
            f"{spec.name}: CV fold {fold_idx}/{n_splits} training ExtraTrees on {X_fit_model.shape[1]} selected features"
        )
        model = make_et_model(params)
        model.fit(X_fit_model, y_fit_model)

        valid_pred_model = pd.DataFrame(
            model.predict(X_valid_model),
            columns=y_train_model.columns,
            index=X_valid.index,
        ).astype(np.float32)
        valid_pred_full = schema.expand_predictions(valid_pred_model)
        fold_score = float(competition_rmse(y_valid_full, valid_pred_full))
        fold_scores.append(fold_score)
        log_progress(f"{spec.name}: CV fold {fold_idx}/{n_splits} done - RMSE {fold_score:.6f}")

        if collect_predictions and oof_predictions is not None:
            oof_predictions.iloc[valid_idx] = valid_pred_model.to_numpy(dtype=np.float32)

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse": fold_score,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "all_feature_count": int(preprocessor.total_feature_count),
                "selected_feature_count": int(len(preprocessor.selected_columns)),
                "dropped_constant_feature_count": int(len(preprocessor.dropped_constant_columns)),
            }
        )

        if trial is not None:
            running_score = float(np.mean(fold_scores))
            trial.report(running_score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    result = {
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "min_rmse": float(np.min(fold_scores)),
        "max_rmse": float(np.max(fold_scores)),
        "fold_reports": fold_reports,
    }

    if collect_predictions and oof_predictions is not None:
        expanded_oof = schema.expand_predictions(oof_predictions)
        result["oof_predictions"] = oof_predictions
        result["oof_rmse"] = float(competition_rmse(y_train_full, expanded_oof))

    log_progress(f"{spec.name}: CV evaluation finished - mean RMSE {result['mean_rmse']:.6f}")
    return result


def evaluate_model_holdout(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    spec: ModelSpec,
    params: dict,
    *,
    ratio_eps: float,
    holdout_fraction: float,
    random_state: int,
) -> dict:
    log_progress(f"{spec.name}: starting Optuna holdout evaluation")
    X_fit, X_valid, y_fit_model, y_valid_model, y_fit_full, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    preprocessor = fit_feature_preprocessor(
        X_fit,
        ratio_eps=ratio_eps,
    )
    X_fit_model = preprocessor.transform(X_fit, ratio_eps=ratio_eps)
    X_valid_model = preprocessor.transform(X_valid, ratio_eps=ratio_eps)

    log_progress(
        f"{spec.name}: holdout training on {len(X_fit_model)} rows with the full generated pool "
        f"({X_fit_model.shape[1]} features after dropping constants)"
    )
    model = make_et_model(params)
    model.fit(X_fit_model, y_fit_model)

    valid_pred_model = pd.DataFrame(
        model.predict(X_valid_model),
        columns=y_valid_model.columns,
        index=X_valid.index,
    ).astype(np.float32)
    valid_pred_full = schema.expand_predictions(valid_pred_model)
    holdout_score = float(competition_rmse(y_valid_full, valid_pred_full))
    log_progress(f"{spec.name}: holdout done - RMSE {holdout_score:.6f}")

    return {
        "holdout_rmse": holdout_score,
        "fit_rows": int(len(X_fit)),
        "valid_rows": int(len(X_valid)),
        "all_feature_count": int(preprocessor.total_feature_count),
        "selected_feature_count": int(len(preprocessor.selected_columns)),
        "dropped_constant_feature_count": int(len(preprocessor.dropped_constant_columns)),
    }


def optimize_model_spec(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    spec: ModelSpec,
    *,
    ratio_eps: float,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
    random_state: int,
    n_jobs: int,
) -> dict:
    log_progress(
        f"{spec.name}: starting Optuna with {n_trials} trial(s), holdout_fraction={holdout_fraction:.2f}, n_jobs={n_jobs}"
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_et_params(trial, spec, random_state=random_state, n_jobs=n_jobs)
        evaluation = evaluate_model_holdout(
            X_train_raw,
            y_train_model,
            y_train_full,
            schema,
            spec,
            params,
            ratio_eps=ratio_eps,
            holdout_fraction=holdout_fraction,
            random_state=random_state,
        )
        trial.report(float(evaluation["holdout_rmse"]), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(evaluation["holdout_rmse"])

    def on_trial_complete(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = "pruned" if trial.value is None else f"{trial.value:.6f}"
        best = study.best_value if study.best_trial is not None else float("nan")
        log_progress(f"{spec.name}: Optuna trial {trial.number + 1} finished - value {value}, best {best:.6f}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=max(3, min(5, n_trials // 2)), n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True, callbacks=[on_trial_complete])

    best_params = dict(study.best_params)
    if "bootstrap" not in best_params:
        best_params["bootstrap"] = bool(spec.bootstrap_choices[0])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_samples_split"] = int(best_params["min_samples_split"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    best_params["random_state"] = random_state
    best_params["n_jobs"] = n_jobs
    if best_params.get("bootstrap", False):
        if "max_samples" not in best_params:
            best_params["max_samples"] = 0.8
    else:
        best_params["max_samples"] = None

    result = {
        "best_params": best_params,
        "best_score": float(study.best_value),
        "n_trials": int(len(study.trials)),
        "strategy": "single_holdout",
        "holdout_fraction": float(holdout_fraction),
    }
    log_progress(f"{spec.name}: Optuna finished - best holdout RMSE {result['best_score']:.6f}")
    return result


def fit_full_model_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    spec: ModelSpec,
    params: dict,
    *,
    ratio_eps: float,
) -> tuple[pd.DataFrame, dict]:
    log_progress(f"{spec.name}: fitting final model on full train")
    preprocessor = fit_feature_preprocessor(
        X_train_raw,
        ratio_eps=ratio_eps,
    )
    X_train_model = preprocessor.transform(X_train_raw, ratio_eps=ratio_eps)
    X_test_model = preprocessor.transform(X_test_raw, ratio_eps=ratio_eps)

    log_progress(
        f"{spec.name}: final fit uses the full generated pool "
        f"({X_train_model.shape[1]} features after dropping constants)"
    )
    model = make_et_model(params)
    model.fit(X_train_model, y_train_model)
    predictions = pd.DataFrame(
        model.predict(X_test_model),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    ).astype(np.float32)

    meta = {
        "all_feature_count": int(preprocessor.total_feature_count),
        "all_feature_columns": preprocessor.all_feature_columns,
        "selected_feature_count": int(len(preprocessor.selected_columns)),
        "selected_columns": preprocessor.selected_columns,
        "dropped_constant_feature_count": int(len(preprocessor.dropped_constant_columns)),
    }
    log_progress(f"{spec.name}: final fit done")
    return predictions, meta


def build_xgb_raw_features(features: pd.DataFrame) -> pd.DataFrame:
    return raw_features(features)[FIXED_XGB_RAW_COLUMNS].copy().astype(np.float32)


def make_xgb_model(*, random_state: int, n_jobs: int) -> MultiOutputRegressor:
    if XGBRegressor is None:
        raise ModuleNotFoundError(
            "xgboost is required to run this script. Install it in the active environment, then relaunch."
        ) from _XGBOOST_IMPORT_ERROR
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        **FIXED_XGB_PARAMS,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def evaluate_xgb_raw_cv(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    n_splits: int,
    random_state: int,
    n_jobs: int,
    collect_predictions: bool = False,
) -> dict:
    log_progress(f"xgb_raw_fixed: starting CV evaluation with {n_splits} folds")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []
    fold_reports: list[dict] = []

    oof_predictions = None
    if collect_predictions:
        oof_predictions = pd.DataFrame(
            np.zeros((len(X_train_raw), y_train_model.shape[1]), dtype=np.float32),
            columns=y_train_model.columns,
            index=X_train_raw.index,
        )

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        X_fit_xgb = build_xgb_raw_features(X_fit)
        X_valid_xgb = build_xgb_raw_features(X_valid)

        model = make_xgb_model(random_state=random_state, n_jobs=n_jobs)
        model.fit(X_fit_xgb, y_fit_model)

        valid_pred_model = pd.DataFrame(
            model.predict(X_valid_xgb),
            columns=y_train_model.columns,
            index=X_valid.index,
        ).astype(np.float32)
        valid_pred_full = schema.expand_predictions(valid_pred_model)
        fold_score = float(competition_rmse(y_valid_full, valid_pred_full))
        fold_scores.append(fold_score)
        log_progress(f"xgb_raw_fixed: CV fold {fold_idx}/{n_splits} done - RMSE {fold_score:.6f}")

        if collect_predictions and oof_predictions is not None:
            oof_predictions.iloc[valid_idx] = valid_pred_model.to_numpy(dtype=np.float32)

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse": fold_score,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "raw_feature_count": int(len(FIXED_XGB_RAW_COLUMNS)),
            }
        )

    result = {
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "min_rmse": float(np.min(fold_scores)),
        "max_rmse": float(np.max(fold_scores)),
        "fold_reports": fold_reports,
    }

    if collect_predictions and oof_predictions is not None:
        expanded_oof = schema.expand_predictions(oof_predictions)
        result["oof_predictions"] = oof_predictions
        result["oof_rmse"] = float(competition_rmse(y_train_full, expanded_oof))

    log_progress(f"xgb_raw_fixed: CV evaluation finished - mean RMSE {result['mean_rmse']:.6f}")
    return result


def fit_xgb_raw_full_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    random_state: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    log_progress("xgb_raw_fixed: fitting final model on full train")
    X_train_xgb = build_xgb_raw_features(X_train_raw)
    X_test_xgb = build_xgb_raw_features(X_test_raw)
    model = make_xgb_model(random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train_xgb, y_train_model)

    predictions = pd.DataFrame(
        model.predict(X_test_xgb),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    ).astype(np.float32)
    meta = {
        "raw_feature_count": int(len(FIXED_XGB_RAW_COLUMNS)),
        "raw_columns": list(FIXED_XGB_RAW_COLUMNS),
    }
    log_progress("xgb_raw_fixed: final fit done")
    return predictions, meta


def weighted_rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    weights = np.where(y_true >= 0.5, 1.2, 1.0).astype(np.float32)
    return float(np.sqrt(np.mean(weights * np.square(y_pred - y_true))))


def build_fixed_weight_blend(
    model_predictions: dict[str, pd.DataFrame],
    y_columns: list[str],
    *,
    et_model_name: str,
    xgb_model_name: str,
    et_weight: float,
    xgb_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_progress(
        f"Blend: applying fixed weights et={et_weight:.3f}, xgb={xgb_weight:.3f}"
    )
    if et_model_name not in model_predictions:
        raise KeyError(f"Missing ET predictions for '{et_model_name}'.")
    if xgb_model_name not in model_predictions:
        raise KeyError(f"Missing XGB predictions for '{xgb_model_name}'.")

    target_weights = pd.DataFrame(
        {
            et_model_name: np.full(len(y_columns), et_weight, dtype=np.float32),
            xgb_model_name: np.full(len(y_columns), xgb_weight, dtype=np.float32),
        },
        index=y_columns,
    )
    blended = pd.DataFrame(index=next(iter(model_predictions.values())).index, columns=y_columns, dtype=np.float32)
    for target in y_columns:
        et_pred = model_predictions[et_model_name][target].to_numpy(dtype=np.float32)
        xgb_pred = model_predictions[xgb_model_name][target].to_numpy(dtype=np.float32)
        blended[target] = np.clip((et_weight * et_pred) + (xgb_weight * xgb_pred), 0.0, 1.0)
    return target_weights, blended


def fit_target_calibration(
    blended_oof: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    optuna_trials: int,
    random_state: int,
) -> tuple[dict[str, CalibrationParams], pd.DataFrame]:
    log_progress(f"Calibration: fitting per-target linear calibration with Optuna shrinkage ({optuna_trials} trials)")
    calibration: dict[str, CalibrationParams] = {}
    calibrated = pd.DataFrame(index=blended_oof.index, columns=blended_oof.columns, dtype=np.float32)

    for target_idx, target in enumerate(blended_oof.columns):
        x = np.clip(blended_oof[target].to_numpy(dtype=np.float32), 0.0, 1.0)
        y = y_train_model[target].to_numpy(dtype=np.float32)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_centered = x - x_mean
        variance = float(np.mean(np.square(x_centered)))
        if variance <= 1e-12:
            slope = 1.0
            intercept = 0.0
        else:
            covariance = float(np.mean(x_centered * (y - y_mean)))
            slope = covariance / variance
            intercept = y_mean - (slope * x_mean)

        linear = np.clip((slope * x) + intercept, 0.0, 1.0)
        def objective(trial: optuna.Trial) -> float:
            shrinkage = trial.suggest_float("shrinkage", 0.0, 0.2)
            candidate = np.clip(((1.0 - shrinkage) * linear) + (shrinkage * y_mean), 0.0, 1.0)
            return weighted_rmse_1d(y, candidate)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=random_state + 5000 + target_idx),
            pruner=MedianPruner(n_startup_trials=min(3, optuna_trials), n_warmup_steps=0),
        )
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

        best_shrinkage = float(study.best_params["shrinkage"])
        best_prediction = np.clip(((1.0 - best_shrinkage) * linear) + (best_shrinkage * y_mean), 0.0, 1.0)
        log_verbose(
            f"Calibration target={target} slope={float(slope):.6f} intercept={float(intercept):.6f} shrinkage={best_shrinkage:.6f}"
        )

        calibration[target] = CalibrationParams(
            slope=float(slope),
            intercept=float(intercept),
            shrinkage=float(best_shrinkage),
            target_mean=float(y_mean),
        )
        calibrated[target] = best_prediction.astype(np.float32)

    log_progress("Calibration: done")
    return calibration, calibrated


def apply_target_calibration(predictions: pd.DataFrame, calibration: dict[str, CalibrationParams]) -> pd.DataFrame:
    calibrated = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=np.float32)
    for target in predictions.columns:
        params = calibration[target]
        raw = predictions[target].to_numpy(dtype=np.float32)
        linear = (params.slope * raw) + params.intercept
        shrunk = ((1.0 - params.shrinkage) * linear) + (params.shrinkage * params.target_mean)
        calibrated[target] = np.clip(shrunk, 0.0, 1.0).astype(np.float32)
    return calibrated


def build_blended_test_predictions(
    model_test_predictions: dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
) -> pd.DataFrame:
    target_names = list(target_weights.index)
    blended = pd.DataFrame(index=next(iter(model_test_predictions.values())).index, columns=target_names, dtype=np.float32)

    for target in target_names:
        total = np.zeros(len(blended), dtype=np.float32)
        for model_name, model_predictions in model_test_predictions.items():
            total += target_weights.loc[target, model_name] * model_predictions[target].to_numpy(dtype=np.float32)
        blended[target] = np.clip(total, 0.0, 1.0)
    return blended


def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = bool(args.verbose)
    log_progress("Pipeline: loading data and preparing run")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_modeling_data(data_dir)
    data = bundle.data
    schema = bundle.schema

    X_train_raw = bundle.x_train_raw
    X_test_raw = bundle.x_test_raw
    y_train_full = bundle.y_train_full
    y_train_model = bundle.y_train_model

    if args.max_train_rows is not None:
        X_train_raw = X_train_raw.iloc[: args.max_train_rows].copy()
        y_train_full = y_train_full.iloc[: args.max_train_rows].copy()
        y_train_model = y_train_model.iloc[: args.max_train_rows].copy()

    if args.max_test_rows is not None:
        X_test_raw = X_test_raw.iloc[: args.max_test_rows].copy()
        x_test_ids = data.x_test["ID"].iloc[: args.max_test_rows].copy()
    else:
        x_test_ids = data.x_test["ID"].copy()

    model_specs = [ModelSpec(**payload) for payload in DEFAULT_MODEL_SPECS]
    model_reports: list[dict] = []
    model_oofs: dict[str, pd.DataFrame] = {}
    model_test_predictions: dict[str, pd.DataFrame] = {}

    for spec in model_specs:
        fixed_params = dict(FIXED_ET_PARAMS[spec.name])
        fixed_params["n_jobs"] = args.n_jobs
        seed = int(fixed_params["random_state"])
        log_progress(f"Pipeline: starting fixed model {spec.name} with training_seed={seed}")

        cv_result = evaluate_model_cv(
            X_train_raw,
            y_train_model,
            y_train_full,
            schema,
            spec,
            fixed_params,
            ratio_eps=args.ratio_eps,
            n_splits=args.eval_cv_folds,
            random_state=seed + 1,
            collect_predictions=True,
        )

        test_predictions, full_fit_meta = fit_full_model_predict(
            X_train_raw,
            X_test_raw,
            y_train_model,
            spec,
            fixed_params,
            ratio_eps=args.ratio_eps,
        )

        model_oofs[spec.name] = cv_result["oof_predictions"]
        model_test_predictions[spec.name] = test_predictions
        model_reports.append(
            {
                "name": spec.name,
                "role": spec.role,
                "training_seed": int(seed),
                "feature_strategy": "all_generated_features_shared_by_every_model",
                "feature_config": asdict(spec),
                "optuna": {
                    "strategy": "fixed_params_reused",
                    "n_trials": 0,
                    "best_params": fixed_params,
                },
                "cv3": {
                    "mean_rmse": float(cv_result["mean_rmse"]),
                    "std_rmse": float(cv_result["std_rmse"]),
                    "min_rmse": float(cv_result["min_rmse"]),
                    "max_rmse": float(cv_result["max_rmse"]),
                    "oof_rmse": float(cv_result["oof_rmse"]),
                    "fold_reports": cv_result["fold_reports"],
                },
                "full_fit": full_fit_meta,
            }
        )
        log_progress(f"Pipeline: model {spec.name} complete")

    xgb_seed = args.random_state + 2048
    xgb_cv_result = evaluate_xgb_raw_cv(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        n_splits=args.eval_cv_folds,
        random_state=xgb_seed,
        n_jobs=args.xgb_n_jobs,
        collect_predictions=True,
    )
    xgb_test_predictions, xgb_full_fit_meta = fit_xgb_raw_full_predict(
        X_train_raw,
        X_test_raw,
        y_train_model,
        random_state=xgb_seed,
        n_jobs=args.xgb_n_jobs,
    )
    model_oofs["xgb_raw_fixed"] = xgb_cv_result["oof_predictions"]
    model_test_predictions["xgb_raw_fixed"] = xgb_test_predictions
    model_reports.append(
        {
            "name": "xgb_raw_fixed",
            "training_seed": int(xgb_seed),
            "feature_strategy": "fixed_raw_13_features",
            "feature_config": {
                "raw_columns": list(FIXED_XGB_RAW_COLUMNS),
            },
            "optuna": {
                "strategy": "fixed_params_reused",
                "n_trials": 0,
                "best_params": dict(FIXED_XGB_PARAMS),
            },
            "cv3": {
                "mean_rmse": float(xgb_cv_result["mean_rmse"]),
                "std_rmse": float(xgb_cv_result["std_rmse"]),
                "min_rmse": float(xgb_cv_result["min_rmse"]),
                "max_rmse": float(xgb_cv_result["max_rmse"]),
                "oof_rmse": float(xgb_cv_result["oof_rmse"]),
                "fold_reports": xgb_cv_result["fold_reports"],
            },
            "full_fit": xgb_full_fit_meta,
        }
    )
    log_progress("Pipeline: model xgb_raw_fixed complete")

    target_weights, blended_oof = build_fixed_weight_blend(
        model_oofs,
        list(y_train_model.columns),
        et_model_name="et_allpool_3",
        xgb_model_name="xgb_raw_fixed",
        et_weight=args.et_weight,
        xgb_weight=args.xgb_weight,
    )
    blended_oof_full = schema.expand_predictions(blended_oof)
    blend_oof_rmse = float(competition_rmse(y_train_full, blended_oof_full))
    log_progress(f"Blend: OOF RMSE before calibration = {blend_oof_rmse:.6f}")

    calibration, calibrated_oof = fit_target_calibration(
        blended_oof,
        y_train_model,
        optuna_trials=args.calibration_optuna_trials,
        random_state=args.random_state,
    )
    calibrated_oof_full = schema.expand_predictions(calibrated_oof)
    calibrated_oof_rmse = float(competition_rmse(y_train_full, calibrated_oof_full))
    log_progress(f"Blend: OOF RMSE after calibration = {calibrated_oof_rmse:.6f}")

    blended_test = build_blended_test_predictions(model_test_predictions, target_weights)
    calibrated_test = apply_target_calibration(blended_test, calibration)
    final_predictions_full = schema.expand_predictions(calibrated_test)
    submission = build_submission_frame(x_test_ids, final_predictions_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    weights_path = output_dir / f"{args.submission_prefix}_{timestamp}_target_weights.csv"
    target_weights.to_csv(weights_path)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Fixed et_allpool_3 + fixed raw XGB + fixed 0.7/0.3 blend + calibration",
        "data_dir": str(data_dir),
        "notes": {
            "humidity_handling": "Humidity was not present in the current raw schema, so it stays excluded.",
            "constant_target": "d15 is inferred as constant and excluded from training via schema expansion.",
            "loss": "competition weighted RMSE",
            "base_model_reuse": "et_allpool_3 and raw XGB reuse fixed best-known hyperparameters; only CV predictions, blend, and calibration are retrained.",
            "ensemble_diversity": "The pruned ensemble keeps one strong ExtraTrees model on the full generated feature pool plus the fixed raw-only XGB model.",
            "blend_rule": "Predictions are blended with a fixed linear rule: et_allpool_3 * et_weight + xgb_raw_fixed * xgb_weight.",
        },
        "preprocessing": {
            "ratio_eps": float(args.ratio_eps),
            "row_stats": ["mean", "std", "p10", "p25", "p50", "p75", "p90", "iqr", "mad", "l1", "l2"],
            "global_feature_pool": "raw + row stats + signed logs + all pairwise sensor ratios + sensor_over_row_mean + block features",
            "feature_selection": "no explicit feature selection; all generated features are used for et_allpool_3 and et_allpool_4, with constant columns dropped fold-by-fold",
            "model_specs": [asdict(spec) for spec in model_specs],
        },
        "training": {
            "base_model_strategy": "fixed_best_params_reused",
            "eval_cv_folds": int(args.eval_cv_folds),
            "fixed_blend_weights": {
                "et_allpool_3": float(args.et_weight),
                "xgb_raw_fixed": float(args.xgb_weight),
            },
            "calibration_optuna_trials": int(args.calibration_optuna_trials),
            "n_jobs": int(args.n_jobs),
            "xgb_n_jobs": int(args.xgb_n_jobs),
            "max_train_rows": None if args.max_train_rows is None else int(args.max_train_rows),
            "max_test_rows": None if args.max_test_rows is None else int(args.max_test_rows),
        },
        "best_params_3_models": {
            **{
                report["name"]: report["optuna"]["best_params"]
                for report in model_reports
                if "optuna" in report and isinstance(report["optuna"], dict) and "best_params" in report["optuna"]
            }
        },
        "models": model_reports,
        "blend": {
            "oof_rmse_before_calibration": float(blend_oof_rmse),
            "oof_rmse_after_calibration": float(calibrated_oof_rmse),
            "target_weight_path": str(weights_path.relative_to(ROOT)),
            "target_weights": {
                target: {model_name: float(target_weights.loc[target, model_name]) for model_name in target_weights.columns}
                for target in target_weights.index
            },
        },
        "calibration": {target: asdict(params) for target, params in calibration.items()},
        "target_handling": {
            "modeled_targets": schema.model_targets,
            "duplicate_groups": [group for group in schema.duplicate_groups if len(group) > 1],
            "constant_targets": schema.constant_targets,
        },
        "submission_path": str(submission_path.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
    }

    summary_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log_progress(f"Pipeline: finished. Submission written to {submission_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
