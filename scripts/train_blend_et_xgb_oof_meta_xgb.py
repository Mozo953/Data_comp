from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    feature_target_signal,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


@dataclass(frozen=True)
class ETPreprocessor:
    selected_columns: list[str]
    ratio_columns: list[str]
    dropped_correlated_columns: list[str]

    def transform(self, features: pd.DataFrame, ratio_eps: float) -> pd.DataFrame:
        expanded = build_ratio_features(features, eps=ratio_eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class XGBFeatureSelector:
    selected_columns: list[str]

    def transform(self, features: pd.DataFrame, ratio_eps: float) -> pd.DataFrame:
        expanded = build_ratio_features(features, eps=ratio_eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class TrainConfig:
    et_params: dict
    et_tuned_params: dict
    xgb_params: dict
    et_corr_threshold: float
    ratio_eps: float
    et_signal_quantile: float
    et_max_selected_features: int
    xgb_signal_quantile: float
    xgb_max_selected_features: int
    xgb_n_jobs: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust OOF stacking: ET fixed + XGB(enriched ratios) + XGB meta-learner."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--et-params-json",
        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/06_experiments_blender__eight_n_traget",
    )
    parser.add_argument("--submission-prefix", default="blend_oof_meta_xgb_cv3")

    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--ratio-eps", type=float, default=1e-3)

    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.20)
    parser.add_argument("--et-max-selected-features", type=int, default=45)
    parser.add_argument("--et-optuna-trials", type=int, default=5)
    parser.add_argument("--et-optuna-timeout-sec", type=int, default=300)
    parser.add_argument("--et-optuna-holdout", type=float, default=0.2)

    parser.add_argument(
        "--xgb-feature-space",
        choices=["raw", "raw_plus5"],
        default="raw_plus5",
        help="Feature space for base XGB. raw_plus5 adds 5 stable engineered features.",
    )
    parser.add_argument("--xgb-signal-quantile", type=float, default=0.15)
    parser.add_argument("--xgb-max-selected-features", type=int, default=15)
    parser.add_argument("--xgb-n-jobs", type=int, default=6)

    parser.add_argument("--xgb-optuna-trials", type=int, default=5)
    parser.add_argument("--xgb-optuna-timeout-sec", type=int, default=900)
    parser.add_argument("--xgb-optuna-holdout", type=float, default=0.2)
    parser.add_argument(
        "--xgb-params-json",
        default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw_best(0.1434)/xgb_trial11_best_params.json",
        help="Fixed XGB params JSON. If set, XGB Optuna is skipped.",
    )

    parser.add_argument("--meta-include-mean-diff", action="store_true")
    parser.add_argument("--meta-optuna-trials", type=int, default=5)
    parser.add_argument("--meta-optuna-timeout-sec", type=int, default=600)
    parser.add_argument("--meta-optuna-holdout", type=float, default=0.2)

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if not 0.0 < args.ratio_eps <= 0.1:
        raise ValueError("--ratio-eps must be in (0, 0.1]")

    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be in (0, 1)")
    if not 0.0 < args.et_signal_quantile <= 1.0:
        raise ValueError("--et-signal-quantile must be in (0, 1]")
    if args.et_max_selected_features < 1:
        raise ValueError("--et-max-selected-features must be >= 1")
    if args.et_optuna_trials < 1:
        raise ValueError("--et-optuna-trials must be >= 1")
    if args.et_optuna_timeout_sec < 1:
        raise ValueError("--et-optuna-timeout-sec must be >= 1")
    if not 0.0 < args.et_optuna_holdout < 1.0:
        raise ValueError("--et-optuna-holdout must be in (0, 1)")

    if not 0.0 < args.xgb_signal_quantile <= 1.0:
        raise ValueError("--xgb-signal-quantile must be in (0, 1]")
    if args.xgb_max_selected_features < 1:
        raise ValueError("--xgb-max-selected-features must be >= 1")
    if args.xgb_n_jobs < 1:
        raise ValueError("--xgb-n-jobs must be >= 1")

    if args.xgb_optuna_trials < 1:
        raise ValueError("--xgb-optuna-trials must be >= 1")
    if args.xgb_optuna_timeout_sec < 1:
        raise ValueError("--xgb-optuna-timeout-sec must be >= 1")
    if not 0.0 < args.xgb_optuna_holdout < 1.0:
        raise ValueError("--xgb-optuna-holdout must be in (0, 1)")

    if args.meta_optuna_trials < 1:
        raise ValueError("--meta-optuna-trials must be >= 1")
    if args.meta_optuna_timeout_sec < 1:
        raise ValueError("--meta-optuna-timeout-sec must be >= 1")
    if not 0.0 < args.meta_optuna_holdout < 1.0:
        raise ValueError("--meta-optuna-holdout must be in (0, 1)")

    return args


def load_best_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        return payload["optuna"]["best_params"]
    if "best_params" in payload:
        return payload["best_params"]
    raise KeyError(f"No best_params found in {json_path}")


def prune_xgb_params(params: dict) -> dict:
    cleaned = dict(params)
    for noisy_key in ["et_weight", "xgb_weight", "weight", "blend_weight"]:
        cleaned.pop(noisy_key, None)
    if "n_estimators" in cleaned:
        cleaned["n_estimators"] = min(int(cleaned["n_estimators"]), 850)
    if "max_depth" in cleaned:
        cleaned["max_depth"] = min(int(cleaned["max_depth"]), 7)
    return cleaned


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    adjusted[np.abs(raw) < eps] = np.where(raw[np.abs(raw) < eps] >= 0.0, eps, -eps)
    return adjusted


def build_ratio_features(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    expanded = base.copy()
    cols = list(base.columns)

    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            denom = _safe_denominator(base[right], eps)
            ratio = base[left].to_numpy(dtype=float) / denom
            expanded[f"{left}_over_{right}"] = np.sign(ratio) * np.log1p(np.abs(ratio))

    return expanded


def build_xgb_feature_bank(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    engineered = pd.DataFrame(index=base.index)

    engineered["mean_y"] = base[["Y1", "Y2", "Y3"]].mean(axis=1)
    engineered["mean_x_core"] = base[["X4", "X5", "X6", "X7"]].mean(axis=1)
    engineered["mean_x_tail"] = base[["X12", "X13", "X14", "X15"]].mean(axis=1)
    engineered["logabs_y1"] = np.sign(base["Y1"].to_numpy(dtype=float)) * np.log1p(np.abs(base["Y1"].to_numpy(dtype=float)))
    engineered["logabs_z"] = np.sign(base["Z"].to_numpy(dtype=float)) * np.log1p(np.abs(base["Z"].to_numpy(dtype=float)))

    # Keep exactly 5 stable extra features.
    return pd.concat([base, engineered], axis=1)


def _prune_correlated_by_order(
    features: pd.DataFrame,
    ordered_columns: list[str],
    threshold: float,
) -> tuple[list[str], list[str]]:
    corr = features[ordered_columns].corr().abs().fillna(0.0)
    kept: list[str] = []
    dropped: list[str] = []

    for col in ordered_columns:
        if any(corr.loc[col, other] >= threshold for other in kept):
            dropped.append(col)
        else:
            kept.append(col)

    return kept, dropped


def fit_et_preprocessor(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
) -> ETPreprocessor:
    expanded = build_ratio_features(X_train_raw, eps=ratio_eps)
    signal = feature_target_signal(expanded, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected, dropped = _prune_correlated_by_order(expanded, ordered, corr_threshold)
    if len(selected) > max_selected_features:
        selected = selected[:max_selected_features]

    ratio_cols = [c for c in expanded.columns if "_over_" in c]
    return ETPreprocessor(
        selected_columns=selected,
        ratio_columns=ratio_cols,
        dropped_correlated_columns=dropped,
    )


def tune_et_params_on_holdout(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    base_et_params: dict,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
    random_state: int,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, _, _, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    et_pre = fit_et_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=corr_threshold,
        ratio_eps=ratio_eps,
        signal_quantile=signal_quantile,
        max_selected_features=max_selected_features,
    )
    X_fit_et = et_pre.transform(X_fit, ratio_eps=ratio_eps)
    X_valid_et = et_pre.transform(X_valid, ratio_eps=ratio_eps)

    def objective(trial: optuna.Trial) -> float:
        params = dict(base_et_params)
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", max(300, int(base_et_params.get("n_estimators", 440)) - 120), int(base_et_params.get("n_estimators", 440)) + 160, step=20),
                "max_depth": trial.suggest_int("max_depth", max(8, int(base_et_params.get("max_depth", 20)) - 6), int(base_et_params.get("max_depth", 20)) + 2),
                "min_samples_split": trial.suggest_int("min_samples_split", max(4, int(base_et_params.get("min_samples_split", 15)) - 6), int(base_et_params.get("min_samples_split", 15)) + 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", max(1, int(base_et_params.get("min_samples_leaf", 2)) - 1), int(base_et_params.get("min_samples_leaf", 2)) + 2),
                "max_features": trial.suggest_float("max_features", max(0.35, float(base_et_params.get("max_features", 0.52)) - 0.12), min(0.95, float(base_et_params.get("max_features", 0.52)) + 0.12)),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "max_samples": trial.suggest_float("max_samples", 0.45, 0.85),
            }
        )
        model = make_et_model(params)
        model.fit(X_fit_et, y_fit_model)
        pred_valid_model = pd.DataFrame(model.predict(X_valid_et), columns=y_fit_model.columns, index=X_valid_et.index)
        pred_valid_full = schema.expand_predictions(pred_valid_model)
        score = float(competition_rmse(y_valid_full, pred_valid_full))
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state + 11),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best_params = dict(base_et_params)
    best_params.update(study.best_params)
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_samples_split"] = int(best_params["min_samples_split"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    best_params["bootstrap"] = bool(best_params["bootstrap"])
    best_params["random_state"] = random_state
    best_params["n_jobs"] = -1
    return best_params, float(study.best_value)


def fit_xgb_selector_enriched(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
) -> XGBFeatureSelector:
    expanded = build_ratio_features(X_train_raw, eps=ratio_eps)
    signal = feature_target_signal(expanded, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected = ordered[:max_selected_features] if len(ordered) > max_selected_features else ordered
    return XGBFeatureSelector(selected_columns=selected)


def fit_xgb_selector_raw_plus5(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
) -> XGBFeatureSelector:
    expanded = build_xgb_feature_bank(X_train_raw, eps=ratio_eps)
    engineered_cols = ["mean_y", "mean_x_core", "mean_x_tail", "logabs_y1", "logabs_z"]
    signal = feature_target_signal(expanded[engineered_cols], y_train_model)
    ordered_engineered = list(signal.sort_values(ascending=False).index)
    selected_engineered = ordered_engineered[: min(5, len(ordered_engineered))]
    return XGBFeatureSelector(selected_columns=list(X_train_raw.columns) + selected_engineered)


def fit_xgb_selector_raw(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    signal_quantile: float,
    max_selected_features: int,
) -> XGBFeatureSelector:
    signal = feature_target_signal(X_train_raw, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected = ordered[:max_selected_features] if len(ordered) > max_selected_features else ordered
    return XGBFeatureSelector(selected_columns=selected)


def make_et_model(params: dict) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(**params)


def make_xgb_model(params: dict, n_jobs: int, random_state: int) -> MultiOutputRegressor:
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        **params,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def optimize_xgb_params_on_holdout(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    ratio_eps: float,
    xgb_signal_quantile: float,
    xgb_max_selected_features: int,
    n_jobs: int,
    random_state: int,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float, dict]:
    X_fit_raw, X_valid_raw, y_fit_model, _, _, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    selector = fit_xgb_selector_raw_plus5(
        X_fit_raw,
        y_fit_model,
        ratio_eps=ratio_eps,
        signal_quantile=xgb_signal_quantile,
        max_selected_features=xgb_max_selected_features,
    )
    fit_features = build_xgb_feature_bank(X_fit_raw, eps=ratio_eps)[selector.selected_columns].copy()
    valid_features = build_xgb_feature_bank(X_valid_raw, eps=ratio_eps)[selector.selected_columns].copy()

    def objective(trial: optuna.Trial) -> float:
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 350, 850, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.9),
            "min_child_weight": trial.suggest_float("min_child_weight", 8.0, 25.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.15),
        }
        model = make_xgb_model(xgb_params, n_jobs=n_jobs, random_state=random_state)
        model.fit(fit_features, y_fit_model)
        pred_valid_model = pd.DataFrame(model.predict(valid_features), columns=y_fit_model.columns, index=valid_features.index)
        pred_valid_full = schema.expand_predictions(pred_valid_model)
        score = float(competition_rmse(y_valid_full, pred_valid_full))
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state + 23),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_info = {
        "selected_feature_count": int(len(selector.selected_columns)),
        "selected_features": selector.selected_columns,
    }
    return best_params, float(study.best_value), best_info


def score_xgb_feature_space(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    y_valid_full: pd.DataFrame,
    schema,
    xgb_params: dict,
    *,
    ratio_eps: float,
    xgb_signal_quantile: float,
    xgb_max_selected_features: int,
    xgb_n_jobs: int,
    random_state: int,
    feature_space: str,
) -> tuple[float, list[str]]:
    if feature_space == "raw_plus5":
        selector = fit_xgb_selector_raw_plus5(
            X_fit_raw,
            y_fit_model,
            ratio_eps=ratio_eps,
            signal_quantile=xgb_signal_quantile,
            max_selected_features=xgb_max_selected_features,
        )
        fit_features = build_xgb_feature_bank(X_fit_raw, eps=ratio_eps)[selector.selected_columns].copy()
        valid_features = build_xgb_feature_bank(X_valid_raw, eps=ratio_eps)[selector.selected_columns].copy()
    elif feature_space == "enriched":
        selector = fit_xgb_selector_enriched(
            X_fit_raw,
            y_fit_model,
            ratio_eps=ratio_eps,
            signal_quantile=xgb_signal_quantile,
            max_selected_features=xgb_max_selected_features,
        )
        fit_features = build_ratio_features(X_fit_raw, eps=ratio_eps)[selector.selected_columns].copy()
        valid_features = build_ratio_features(X_valid_raw, eps=ratio_eps)[selector.selected_columns].copy()
    elif feature_space == "raw":
        selector = fit_xgb_selector_raw(
            X_fit_raw,
            y_fit_model,
            signal_quantile=xgb_signal_quantile,
            max_selected_features=xgb_max_selected_features,
        )
        fit_features = X_fit_raw[selector.selected_columns].copy()
        valid_features = X_valid_raw[selector.selected_columns].copy()
    else:
        raise ValueError(f"Unsupported feature space: {feature_space}")

    xgb_model = make_xgb_model(xgb_params, n_jobs=xgb_n_jobs, random_state=random_state)
    xgb_model.fit(fit_features, y_fit_model)
    valid_pred_model = pd.DataFrame(
        xgb_model.predict(valid_features),
        columns=y_fit_model.columns,
        index=valid_features.index,
    )
    valid_pred_full = schema.expand_predictions(valid_pred_model)
    score = float(competition_rmse(y_valid_full, valid_pred_full))
    return score, selector.selected_columns


def optimize_xgb_params_on_holdout(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    feature_space: str,
    ratio_eps: float,
    xgb_signal_quantile: float,
    xgb_max_selected_features: int,
    n_jobs: int,
    random_state: int,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float, dict]:
    X_fit_raw, X_valid_raw, y_fit_model, _, _, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        score, _ = score_xgb_feature_space(
            X_fit_raw,
            X_valid_raw,
            y_fit_model,
            y_valid_full,
            schema,
            xgb_params,
            ratio_eps=ratio_eps,
            xgb_signal_quantile=xgb_signal_quantile,
            xgb_max_selected_features=xgb_max_selected_features,
            xgb_n_jobs=n_jobs,
            random_state=random_state,
            feature_space=feature_space,
        )
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])

    best_score, best_cols = score_xgb_feature_space(
        X_fit_raw,
        X_valid_raw,
        y_fit_model,
        y_valid_full,
        schema,
        best_params,
        ratio_eps=ratio_eps,
        xgb_signal_quantile=xgb_signal_quantile,
        xgb_max_selected_features=xgb_max_selected_features,
        xgb_n_jobs=n_jobs,
        random_state=random_state,
        feature_space=feature_space,
    )

    info = {
        "feature_space": feature_space,
        "selected_feature_count": int(len(best_cols)),
        "selected_features": best_cols,
    }
    return best_params, float(best_score), info


def pick_xgb_feature_space_auto(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    default_xgb_params: dict,
    *,
    ratio_eps: float,
    xgb_signal_quantile: float,
    xgb_max_selected_features: int,
    xgb_n_jobs: int,
    random_state: int,
    holdout_fraction: float,
) -> tuple[str, dict]:
    X_fit_raw, X_valid_raw, y_fit_model, _, _, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    raw_score, raw_cols = score_xgb_feature_space(
        X_fit_raw,
        X_valid_raw,
        y_fit_model,
        y_valid_full,
        schema,
        default_xgb_params,
        ratio_eps=ratio_eps,
        xgb_signal_quantile=xgb_signal_quantile,
        xgb_max_selected_features=xgb_max_selected_features,
        xgb_n_jobs=xgb_n_jobs,
        random_state=random_state,
        feature_space="raw",
    )
    enriched_score, enriched_cols = score_xgb_feature_space(
        X_fit_raw,
        X_valid_raw,
        y_fit_model,
        y_valid_full,
        schema,
        default_xgb_params,
        ratio_eps=ratio_eps,
        xgb_signal_quantile=xgb_signal_quantile,
        xgb_max_selected_features=xgb_max_selected_features,
        xgb_n_jobs=xgb_n_jobs,
        random_state=random_state,
        feature_space="enriched",
    )

    if enriched_score <= raw_score:
        chosen = "enriched"
        chosen_score = enriched_score
        chosen_cols = enriched_cols
    else:
        chosen = "raw"
        chosen_score = raw_score
        chosen_cols = raw_cols

    report = {
        "raw_score": float(raw_score),
        "raw_feature_count": int(len(raw_cols)),
        "enriched_score": float(enriched_score),
        "enriched_feature_count": int(len(enriched_cols)),
        "chosen_feature_space": chosen,
        "chosen_score": float(chosen_score),
        "chosen_feature_count": int(len(chosen_cols)),
    }
    return chosen, report


def transform_xgb_features(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    *,
    feature_space: str,
    ratio_eps: float,
    xgb_signal_quantile: float,
    xgb_max_selected_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if feature_space == "raw_plus5":
        selector = fit_xgb_selector_raw_plus5(
            X_fit_raw,
            y_fit_model,
            ratio_eps=ratio_eps,
            signal_quantile=xgb_signal_quantile,
            max_selected_features=xgb_max_selected_features,
        )
        fit_all = build_xgb_feature_bank(X_fit_raw, eps=ratio_eps)
        valid_all = build_xgb_feature_bank(X_valid_raw, eps=ratio_eps)
        return (
            fit_all[selector.selected_columns].copy(),
            valid_all[selector.selected_columns].copy(),
            selector.selected_columns,
        )
    if feature_space == "enriched":
        selector = fit_xgb_selector_enriched(
            X_fit_raw,
            y_fit_model,
            ratio_eps=ratio_eps,
            signal_quantile=xgb_signal_quantile,
            max_selected_features=xgb_max_selected_features,
        )
        fit_all = build_ratio_features(X_fit_raw, eps=ratio_eps)
        valid_all = build_ratio_features(X_valid_raw, eps=ratio_eps)
        return (
            fit_all[selector.selected_columns].copy(),
            valid_all[selector.selected_columns].copy(),
            selector.selected_columns,
        )

    selector = fit_xgb_selector_raw(
        X_fit_raw,
        y_fit_model,
        signal_quantile=xgb_signal_quantile,
        max_selected_features=xgb_max_selected_features,
    )
    return (
        X_fit_raw[selector.selected_columns].copy(),
        X_valid_raw[selector.selected_columns].copy(),
        selector.selected_columns,
    )


def run_oof_base_models(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: TrainConfig,
    *,
    xgb_feature_space: str,
    cv_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)
    oof_et = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    oof_xgb = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    reports: list[dict] = []

    for fold, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit_raw = X_train_raw.iloc[fit_idx]
        X_valid_raw = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]

        et_pre = fit_et_preprocessor(
            X_fit_raw,
            y_fit_model,
            corr_threshold=cfg.et_corr_threshold,
            ratio_eps=cfg.ratio_eps,
            signal_quantile=cfg.et_signal_quantile,
            max_selected_features=cfg.et_max_selected_features,
        )
        X_fit_et = et_pre.transform(X_fit_raw, ratio_eps=cfg.ratio_eps)
        X_valid_et = et_pre.transform(X_valid_raw, ratio_eps=cfg.ratio_eps)

        et_model = make_et_model(cfg.et_tuned_params)
        et_model.fit(X_fit_et, y_fit_model)
        pred_et = pd.DataFrame(
            et_model.predict(X_valid_et),
            columns=y_fit_model.columns,
            index=X_valid_raw.index,
        )

        X_fit_xgb, X_valid_xgb, xgb_cols = transform_xgb_features(
            X_fit_raw,
            X_valid_raw,
            y_fit_model,
            feature_space=xgb_feature_space,
            ratio_eps=cfg.ratio_eps,
            xgb_signal_quantile=cfg.xgb_signal_quantile,
            xgb_max_selected_features=cfg.xgb_max_selected_features,
        )
        xgb_model = make_xgb_model(cfg.xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        xgb_model.fit(X_fit_xgb, y_fit_model)
        pred_xgb = pd.DataFrame(
            xgb_model.predict(X_valid_xgb),
            columns=y_fit_model.columns,
            index=X_valid_raw.index,
        )

        oof_et.loc[X_valid_raw.index, :] = pred_et
        oof_xgb.loc[X_valid_raw.index, :] = pred_xgb

        reports.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "et_features": int(X_fit_et.shape[1]),
                "xgb_features": int(len(xgb_cols)),
            }
        )

    return oof_et, oof_xgb, reports


def build_meta_features(
    oof_et: pd.DataFrame,
    oof_xgb: pd.DataFrame,
    include_mean_diff: bool,
) -> pd.DataFrame:
    et = oof_et.copy()
    et.columns = [f"{c}__et" for c in et.columns]

    xgb = oof_xgb.copy()
    xgb.columns = [f"{c}__xgb" for c in xgb.columns]

    blocks = [et, xgb]

    if include_mean_diff:
        mean_block = ((oof_et + oof_xgb) * 0.5).copy()
        mean_block.columns = [f"{c}__mean" for c in mean_block.columns]

        diff_block = (oof_et - oof_xgb).copy()
        diff_block.columns = [f"{c}__diff" for c in diff_block.columns]

        blocks.extend([mean_block, diff_block])

    return pd.concat(blocks, axis=1)


def optimize_meta_xgb_on_oof(
    X_meta_oof: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    n_jobs: int,
    random_state: int,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, _, _, y_valid_full = train_test_split(
        X_meta_oof,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 500, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }
        model = make_xgb_model(params, n_jobs=n_jobs, random_state=random_state)
        model.fit(X_fit, y_fit_model)

        pred_valid_model = pd.DataFrame(
            model.predict(X_valid),
            columns=y_fit_model.columns,
            index=X_valid.index,
        )
        pred_valid_full = schema.expand_predictions(pred_valid_model)
        score = float(competition_rmse(y_valid_full, pred_valid_full))

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state + 97),
        pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best = dict(study.best_params)
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    return best, float(study.best_value)


def run_meta_oof_cv(
    X_meta_oof: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    meta_params: dict,
    n_jobs: int,
    random_state: int,
    cv_folds: int,
) -> tuple[pd.DataFrame, list[dict], dict]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_meta_pred = pd.DataFrame(index=y_train_model.index, columns=y_train_model.columns, dtype=float)
    fold_reports: list[dict] = []

    for fold, (fit_idx, valid_idx) in enumerate(kfold.split(X_meta_oof), start=1):
        X_fit = X_meta_oof.iloc[fit_idx]
        X_valid = X_meta_oof.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        model = make_xgb_model(meta_params, n_jobs=n_jobs, random_state=random_state)
        model.fit(X_fit, y_fit_model)

        pred_valid_model = pd.DataFrame(
            model.predict(X_valid),
            columns=y_fit_model.columns,
            index=X_valid.index,
        )
        oof_meta_pred.loc[X_valid.index, :] = pred_valid_model

        pred_valid_full = schema.expand_predictions(pred_valid_model)
        fold_rmse = float(competition_rmse(y_valid_full, pred_valid_full))

        fold_reports.append(
            {
                "fold": fold,
                "rmse": fold_rmse,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    scores = [row["rmse"] for row in fold_reports]
    summary = {
        "mean_rmse": float(np.mean(scores)),
        "std_rmse": float(np.std(scores)),
        "min_rmse": float(np.min(scores)),
        "max_rmse": float(np.max(scores)),
    }
    return oof_meta_pred, fold_reports, summary


def fit_full_base_and_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: TrainConfig,
    *,
    xgb_feature_space: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    et_pre = fit_et_preprocessor(
        X_train_raw,
        y_train_model,
        corr_threshold=cfg.et_corr_threshold,
        ratio_eps=cfg.ratio_eps,
        signal_quantile=cfg.et_signal_quantile,
        max_selected_features=cfg.et_max_selected_features,
    )
    X_train_et = et_pre.transform(X_train_raw, ratio_eps=cfg.ratio_eps)
    X_test_et = et_pre.transform(X_test_raw, ratio_eps=cfg.ratio_eps)

    et_model = make_et_model(cfg.et_tuned_params)
    et_model.fit(X_train_et, y_train_model)
    pred_et_test = pd.DataFrame(
        et_model.predict(X_test_et),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )

    X_train_xgb, X_test_xgb, xgb_cols = transform_xgb_features(
        X_train_raw,
        X_test_raw,
        y_train_model,
        feature_space=xgb_feature_space,
        ratio_eps=cfg.ratio_eps,
        xgb_signal_quantile=cfg.xgb_signal_quantile,
        xgb_max_selected_features=cfg.xgb_max_selected_features,
    )
    xgb_model = make_xgb_model(cfg.xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
    xgb_model.fit(X_train_xgb, y_train_model)
    pred_xgb_test = pd.DataFrame(
        xgb_model.predict(X_test_xgb),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )

    report = {
        "et_selected_feature_count": int(X_train_et.shape[1]),
        "xgb_feature_space": xgb_feature_space,
        "xgb_selected_feature_count": int(len(xgb_cols)),
        "xgb_selected_features": xgb_cols,
    }
    return pred_et_test, pred_xgb_test, report


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    et_params_json = Path(args.et_params_json)
    if not et_params_json.is_absolute():
        et_params_json = (ROOT / et_params_json).resolve()

    et_params = load_best_params_from_json(et_params_json)
    et_params["random_state"] = args.random_state
    et_params["n_jobs"] = -1

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train_raw = raw_features(data.x_train)
    X_test_raw = raw_features(data.x_test)
    y_train_full = data.y_train.drop(columns=["ID"]).copy() if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    tuned_et_params, et_optuna_best_rmse = tune_et_params_on_holdout(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        et_params,
        corr_threshold=args.et_corr_threshold,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.et_signal_quantile,
        max_selected_features=args.et_max_selected_features,
        n_trials=args.et_optuna_trials,
        timeout_sec=args.et_optuna_timeout_sec,
        holdout_fraction=args.et_optuna_holdout,
        random_state=args.random_state,
    )

    chosen_feature_space = args.xgb_feature_space
    xgb_params, xgb_optuna_best_rmse, xgb_optuna_info = optimize_xgb_params_on_holdout(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        feature_space=chosen_feature_space,
        ratio_eps=args.ratio_eps,
        xgb_signal_quantile=args.xgb_signal_quantile,
        xgb_max_selected_features=args.xgb_max_selected_features,
        n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
        n_trials=args.xgb_optuna_trials,
        timeout_sec=args.xgb_optuna_timeout_sec,
        holdout_fraction=args.xgb_optuna_holdout,
    )
    xgb_params = prune_xgb_params(xgb_params)

    cfg = TrainConfig(
        et_params=et_params,
        et_tuned_params=tuned_et_params,
        xgb_params=xgb_params,
        et_corr_threshold=args.et_corr_threshold,
        ratio_eps=args.ratio_eps,
        et_signal_quantile=args.et_signal_quantile,
        et_max_selected_features=args.et_max_selected_features,
        xgb_signal_quantile=args.xgb_signal_quantile,
        xgb_max_selected_features=args.xgb_max_selected_features,
        xgb_n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
    )

    oof_et, oof_xgb, base_fold_reports = run_oof_base_models(
        X_train_raw,
        y_train_model,
        cfg,
        xgb_feature_space=chosen_feature_space,
        cv_folds=args.cv_folds,
    )

    X_meta_oof = build_meta_features(
        oof_et,
        oof_xgb,
        include_mean_diff=args.meta_include_mean_diff,
    )

    meta_params, meta_optuna_best_rmse = optimize_meta_xgb_on_oof(
        X_meta_oof,
        y_train_model,
        y_train_full,
        schema,
        n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
        n_trials=args.meta_optuna_trials,
        timeout_sec=args.meta_optuna_timeout_sec,
        holdout_fraction=args.meta_optuna_holdout,
    )

    oof_meta_pred, meta_fold_reports, meta_cv_summary = run_meta_oof_cv(
        X_meta_oof,
        y_train_model,
        y_train_full,
        schema,
        meta_params=meta_params,
        n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )
    oof_meta_full = schema.expand_predictions(oof_meta_pred)
    oof_meta_rmse = float(competition_rmse(y_train_full, oof_meta_full))

    pred_et_test, pred_xgb_test, full_base_report = fit_full_base_and_predict(
        X_train_raw,
        X_test_raw,
        y_train_model,
        cfg,
        xgb_feature_space=chosen_feature_space,
    )

    X_meta_test = build_meta_features(
        pred_et_test,
        pred_xgb_test,
        include_mean_diff=args.meta_include_mean_diff,
    )

    meta_full_model = make_xgb_model(meta_params, n_jobs=args.xgb_n_jobs, random_state=args.random_state)
    meta_full_model.fit(X_meta_oof, y_train_model)
    pred_meta_test_model = pd.DataFrame(
        meta_full_model.predict(X_meta_test),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )
    pred_meta_test_full = schema.expand_predictions(pred_meta_test_model)

    submission = build_submission_frame(data.x_test["ID"], pred_meta_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    oof_et_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_et_model.csv"
    oof_xgb_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_xgb_model.csv"
    oof_meta_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_meta_model.csv"

    submission.to_csv(submission_file, index=False)
    oof_et.to_csv(oof_et_file, index=True)
    oof_xgb.to_csv(oof_xgb_file, index=True)
    oof_meta_pred.to_csv(oof_meta_file, index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Robust OOF stacking (ET fixed + XGB base + XGB meta)",
        "data_dir": str(data_dir),
        "et_params_json": str(et_params_json),
        "et_optuna": {
            "trials": int(args.et_optuna_trials),
            "timeout_sec": int(args.et_optuna_timeout_sec),
            "holdout_fraction": float(args.et_optuna_holdout),
            "best_holdout_rmse": float(et_optuna_best_rmse),
            "best_params": tuned_et_params,
        },
        "et_feature_pipeline": {
            "corr_threshold": float(args.et_corr_threshold),
            "signal_quantile": float(args.et_signal_quantile),
            "max_selected_features": int(args.et_max_selected_features),
            "ratio_eps": float(args.ratio_eps),
        },
        "xgb_base": {
            "feature_space": chosen_feature_space,
            "signal_quantile": float(args.xgb_signal_quantile),
            "max_selected_features": int(args.xgb_max_selected_features),
            "n_jobs": int(args.xgb_n_jobs),
            "best_params": xgb_params,
            "optuna_trials": int(args.xgb_optuna_trials),
            "optuna_timeout_sec": int(args.xgb_optuna_timeout_sec),
            "optuna_holdout": float(args.xgb_optuna_holdout),
            "optuna_best_rmse": xgb_optuna_best_rmse,
            "optuna_info": xgb_optuna_info,
            "xgb_params_json": args.xgb_params_json,
        },
        "meta_features": {
            "sources": ["oof_et", "oof_xgb"],
            "include_mean_diff": bool(args.meta_include_mean_diff),
            "meta_feature_count": int(X_meta_oof.shape[1]),
        },
        "meta_learner": {
            "name": "MultiOutputRegressor(XGBRegressor)",
            "best_params": meta_params,
            "optuna_best_rmse": float(meta_optuna_best_rmse),
            "training": "trained_strictly_on_oof_predictions",
        },
        "cv": {
            "folds": int(args.cv_folds),
            "base_fold_reports": base_fold_reports,
            "meta_fold_reports": meta_fold_reports,
            "meta_cv_summary": meta_cv_summary,
            "meta_oof_rmse": float(oof_meta_rmse),
        },
        "full_train": full_base_report,
        "oof_paths": {
            "oof_et_model_path": str(oof_et_file.relative_to(ROOT)),
            "oof_xgb_model_path": str(oof_xgb_file.relative_to(ROOT)),
            "oof_meta_model_path": str(oof_meta_file.relative_to(ROOT)),
        },
        "target_handling": {
            "d15_strategy": "constant_target_removed_from_training_via_schema",
            "modeled_targets": schema.model_targets,
            "duplicate_groups": [group for group in schema.duplicate_groups if len(group) > 1],
            "constant_targets": schema.constant_targets,
        },
        "submission_path": str(submission_file.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
    }

    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
