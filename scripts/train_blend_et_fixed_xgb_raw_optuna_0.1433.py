from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    feature_target_signal,
    load_modeling_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


FEATURE_MODE = "aggregated"


@dataclass(frozen=True)
class FeaturePreprocessor:
    selected_columns: list[str]
    ratio_columns: list[str]
    dropped_correlated_columns: list[str]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        expanded = build_ratio_features(features)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class XGBFeaturePreprocessor:
    selected_columns: list[str]
    dropped_correlated_columns: list[str]

    def transform(self, features: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
        expanded = build_xgb_features(features, eps=ratio_eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class BlendConfig:
    et_params: dict
    et_corr_threshold: float
    et_ratio_eps: float
    et_signal_quantile: float
    et_max_selected_features: int
    xgb_corr_threshold: float
    xgb_ratio_eps: float
    xgb_signal_quantile: float
    xgb_max_selected_features: int
    xgb_n_jobs: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend fixed ExtraTrees + conservative XGBoost with Optuna.")
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--et-params-json",
        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json",
        help="JSON file containing ExtraTrees best_params from prior run.",
    )
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw")
    parser.add_argument("--submission-prefix", default="blend_et_fixed_xgb_raw")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        choices=["aggregated", "legacy"],
        default="aggregated",
        help="Feature engineering mode: aggregated row stats or legacy ratios/log features.",
    )

    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.20)
    parser.add_argument("--et-max-selected-features", type=int, default=45)
    parser.add_argument("--et-ratio-eps", type=float, default=1e-3)

    parser.add_argument(
        "--xgb-corr-threshold",
        type=float,
        default=0.98,
        help="Correlation threshold used to prune redundant XGB features.",
    )
    parser.add_argument("--xgb-ratio-eps", type=float, default=1e-3)
    parser.add_argument("--xgb-signal-quantile", type=float, default=0.2)
    parser.add_argument(
        "--xgb-max-selected-features",
        type=int,
        default=20,
        help="Maximum number of XGB input features after pruning.",
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=6)

    parser.add_argument("--xgb-optuna-trials", type=int, default=20)
    parser.add_argument("--xgb-optuna-timeout-sec", type=int, default=900)
    parser.add_argument("--xgb-optuna-holdout", type=float, default=0.2)
    parser.add_argument("--blend-optuna-trials", type=int, default=20)
    parser.add_argument("--blend-optuna-timeout-sec", type=int, default=600)
    parser.add_argument("--blend-optuna-holdout", type=float, default=0.2)
    parser.add_argument(
        "--fixed-et-weight",
        type=float,
        default=None,
        help="Fixed ET blend weight in [0,1]. When set, blend Optuna is skipped.",
    )
    parser.add_argument(
        "--xgb-params-json",
        default=None,
        help="Optional JSON file with fixed XGB parameters. When set, XGB Optuna is skipped.",
    )

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be between 0 and 1.")
    if not 0.0 < args.et_signal_quantile <= 1.0:
        raise ValueError("--et-signal-quantile must be between 0 and 1.")
    if args.et_max_selected_features < 1:
        raise ValueError("--et-max-selected-features must be >= 1.")
    if not 0.0 < args.xgb_corr_threshold < 1.0:
        raise ValueError("--xgb-corr-threshold must be between 0 and 1.")
    if args.xgb_max_selected_features < 1:
        raise ValueError("--xgb-max-selected-features must be >= 1.")
    if not 0.0 < args.xgb_optuna_holdout < 1.0:
        raise ValueError("--xgb-optuna-holdout must be between 0 and 1.")
    if not 0.0 < args.blend_optuna_holdout < 1.0:
        raise ValueError("--blend-optuna-holdout must be between 0 and 1.")
    if args.xgb_optuna_trials < 1:
        raise ValueError("--xgb-optuna-trials must be >= 1.")
    if args.blend_optuna_trials < 1:
        raise ValueError("--blend-optuna-trials must be >= 1.")
    if args.fixed_et_weight is not None and not 0.0 <= args.fixed_et_weight <= 1.0:
        raise ValueError("--fixed-et-weight must be between 0 and 1.")

    return args


def load_best_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        return payload["optuna"]["best_params"]
    if "best_params" in payload:
        return payload["best_params"]
    raise KeyError(f"No best_params found in {json_path}")


def load_xgb_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        params = payload["optuna"]["best_params"]
    elif "best_params" in payload:
        params = payload["best_params"]
    else:
        raise KeyError(f"No xgb params found in {json_path}")
    params = dict(params)
    params.pop("et_weight", None)
    return params


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    adjusted[np.abs(raw) < eps] = np.where(raw[np.abs(raw) < eps] >= 0.0, eps, -eps)
    return adjusted


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_row_aggregated_features(features: pd.DataFrame) -> pd.DataFrame:
    base = raw_features(features)
    values = base.to_numpy(dtype=float)

    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    p10 = np.percentile(values, 10, axis=1)
    p25 = np.percentile(values, 25, axis=1)
    p50 = np.percentile(values, 50, axis=1)
    p75 = np.percentile(values, 75, axis=1)
    p90 = np.percentile(values, 90, axis=1)
    iqr = p75 - p25
    mad = np.median(np.abs(values - p50[:, None]), axis=1)
    l1 = np.linalg.norm(values, ord=1, axis=1)
    l2 = np.linalg.norm(values, ord=2, axis=1)

    aggregated = pd.DataFrame(
        {
            "row_mean": mean,
            "row_std": std,
            "row_p10": p10,
            "row_p25": p25,
            "row_p50": p50,
            "row_p75": p75,
            "row_p90": p90,
            "row_iqr": iqr,
            "row_mad": mad,
            "row_l1": l1,
            "row_l2": l2,
        },
        index=base.index,
    )
    return pd.concat([base, aggregated], axis=1)


def build_ratio_features(features: pd.DataFrame, *, eps: float = 1e-3) -> pd.DataFrame:
    if FEATURE_MODE == "legacy":
        base = raw_features(features)
        expanded = base.copy()
        columns = list(base.columns)

        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                denom = _safe_denominator(base[right], eps)
                ratio = base[left].to_numpy(dtype=float) / denom
                expanded[f"{left}_over_{right}"] = np.sign(ratio) * np.log1p(np.abs(ratio))
        return expanded

    _ = eps
    return build_row_aggregated_features(features)


def build_xgb_features(features: pd.DataFrame, *, eps: float = 1e-3) -> pd.DataFrame:
    if FEATURE_MODE == "legacy":
        base = raw_features(features)
        engineered = base.copy()

        for column in base.columns:
            if column == "Env":
                continue
            engineered[f"{column}_logabs"] = _signed_log1p(base[column].to_numpy(dtype=float))

        if {"Y1", "Y2", "Y3"}.issubset(base.columns):
            y_group_mean = base[["Y1", "Y2", "Y3"]].abs().mean(axis=1).to_numpy(dtype=float)
            engineered["Y1_over_mean_Y"] = base["Y1"].to_numpy(dtype=float) / (y_group_mean + eps)
            engineered["Z_over_Y1"] = base["Z"].to_numpy(dtype=float) / (_safe_denominator(base["Y1"], eps))
            engineered["Y1_over_Y2"] = base["Y1"].to_numpy(dtype=float) / (_safe_denominator(base["Y2"], eps))

        if {"X4", "X5", "X6", "X7"}.issubset(base.columns):
            x_group_mean = base[["X4", "X5", "X6", "X7"]].abs().mean(axis=1).to_numpy(dtype=float)
            engineered["X4_over_mean_X"] = base["X4"].to_numpy(dtype=float) / (x_group_mean + eps)
            engineered["X4_x_X5"] = base["X4"] * base["X5"]
            engineered["X6_x_X7"] = base["X6"] * base["X7"]

        if {"Y1", "Z"}.issubset(base.columns):
            engineered["Y1_x_Z"] = base["Y1"] * base["Z"]
        return engineered

    _ = eps
    return build_row_aggregated_features(features)


def _prune_correlated_by_order(
    features: pd.DataFrame,
    ordered_columns: list[str],
    threshold: float,
) -> tuple[list[str], list[str]]:
    corr = features[ordered_columns].corr().abs().fillna(0.0)
    kept: list[str] = []
    dropped: list[str] = []

    for column in ordered_columns:
        if any(corr.loc[column, other] >= threshold for other in kept):
            dropped.append(column)
        else:
            kept.append(column)

    return kept, dropped


def fit_feature_preprocessor(
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
) -> FeaturePreprocessor:
    expanded = build_ratio_features(X_train, eps=ratio_eps)
    signal = feature_target_signal(expanded, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected, dropped = _prune_correlated_by_order(expanded, ordered, corr_threshold)
    if len(selected) > max_selected_features:
        selected = selected[:max_selected_features]
    ratio_columns = [column for column in expanded.columns if "_over_" in column]
    return FeaturePreprocessor(
        selected_columns=selected,
        ratio_columns=ratio_columns,
        dropped_correlated_columns=dropped,
    )


def fit_xgb_preprocessor(
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
) -> XGBFeaturePreprocessor:
    expanded = build_xgb_features(X_train, eps=ratio_eps)
    signal = feature_target_signal(expanded, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected, dropped = _prune_correlated_by_order(expanded, ordered, corr_threshold)
    if len(selected) > max_selected_features:
        selected = selected[:max_selected_features]
    return XGBFeaturePreprocessor(selected_columns=selected, dropped_correlated_columns=dropped)


def make_et_model(params: dict) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(**params)


def select_xgb_raw_columns(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    max_features: int,
) -> list[str]:
    max_features = min(max_features, X_train_raw.shape[1])
    if max_features >= X_train_raw.shape[1]:
        return list(X_train_raw.columns)
    signal = feature_target_signal(X_train_raw, y_train_model)
    ordered = list(signal.sort_values(ascending=False).index)
    return ordered[:max_features]


def make_xgb_model(params: dict, n_jobs: int, random_state: int) -> MultiOutputRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        **params,
    )
    return MultiOutputRegressor(model, n_jobs=1)


def optimize_xgb_params(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: BlendConfig,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, y_valid_model, y_fit_full, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=cfg.random_state,
    )

    xgb_pre = fit_xgb_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=cfg.xgb_corr_threshold,
        ratio_eps=cfg.xgb_ratio_eps,
        signal_quantile=cfg.xgb_signal_quantile,
        max_selected_features=cfg.xgb_max_selected_features,
    )
    X_fit_xgb = xgb_pre.transform(X_fit, ratio_eps=cfg.xgb_ratio_eps)
    X_valid_xgb = xgb_pre.transform(X_valid, ratio_eps=cfg.xgb_ratio_eps)

    def objective(trial: optuna.Trial) -> float:
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 450, 650, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.45, 0.7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.78),
            "min_child_weight": trial.suggest_float("min_child_weight", 15.0, 35.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.2, 4.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.2),
        }

        xgb_model = make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        xgb_model.fit(X_fit_xgb, y_fit_model)
        xgb_valid_pred_model = pd.DataFrame(
            xgb_model.predict(X_valid_xgb),
            columns=y_fit_model.columns,
            index=X_valid_xgb.index,
        )

        xgb_valid_full = schema.expand_predictions(xgb_valid_pred_model)
        score = competition_rmse(y_valid_full, xgb_valid_full)

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(score)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg.random_state),
        pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best = study.best_params.copy()
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])

    return best, float(study.best_value)


def optimize_blend_weight(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: BlendConfig,
    xgb_params: dict,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[float, float]:
    X_fit, X_valid, y_fit_model, _, _, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=cfg.random_state + 1,
    )

    et_pre = fit_feature_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=cfg.et_corr_threshold,
        ratio_eps=cfg.et_ratio_eps,
        signal_quantile=cfg.et_signal_quantile,
        max_selected_features=cfg.et_max_selected_features,
    )
    xgb_pre = fit_xgb_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=cfg.xgb_corr_threshold,
        ratio_eps=cfg.xgb_ratio_eps,
        signal_quantile=cfg.xgb_signal_quantile,
        max_selected_features=cfg.xgb_max_selected_features,
    )

    X_fit_et = et_pre.transform(X_fit)
    X_valid_et = et_pre.transform(X_valid)
    X_fit_xgb = xgb_pre.transform(X_fit, ratio_eps=cfg.xgb_ratio_eps)
    X_valid_xgb = xgb_pre.transform(X_valid, ratio_eps=cfg.xgb_ratio_eps)

    et_model = make_et_model(cfg.et_params)
    et_model.fit(X_fit_et, y_fit_model)
    pred_valid_et = pd.DataFrame(et_model.predict(X_valid_et), columns=y_fit_model.columns, index=X_valid.index)

    xgb_model = make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
    xgb_model.fit(X_fit_xgb, y_fit_model)
    pred_valid_xgb = pd.DataFrame(xgb_model.predict(X_valid_xgb), columns=y_fit_model.columns, index=X_valid.index)

    def objective(trial: optuna.Trial) -> float:
        et_weight = trial.suggest_float("et_weight", 0.65, 0.78)
        blend_valid_model = (et_weight * pred_valid_et) + ((1.0 - et_weight) * pred_valid_xgb)
        blend_valid_full = schema.expand_predictions(blend_valid_model)
        score = competition_rmse(y_valid_full, blend_valid_full)
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(score)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg.random_state + 1),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    return float(study.best_params["et_weight"]), float(study.best_value)


def run_cv_blend(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: BlendConfig,
    xgb_params: dict,
    et_weight: float,
    cv_folds: int,
) -> tuple[list[dict], dict]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)
    fold_reports: list[dict] = []
    weight_candidates = [0.70, 0.72, et_weight]

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        et_pre = fit_feature_preprocessor(
            X_fit,
            y_fit_model,
            corr_threshold=cfg.et_corr_threshold,
            ratio_eps=cfg.et_ratio_eps,
            signal_quantile=cfg.et_signal_quantile,
            max_selected_features=cfg.et_max_selected_features,
        )
        X_fit_et = et_pre.transform(X_fit)
        X_valid_et = et_pre.transform(X_valid)

        et_model = make_et_model(cfg.et_params)
        et_model.fit(X_fit_et, y_fit_model)
        et_valid_pred_model = pd.DataFrame(
            et_model.predict(X_valid_et),
            columns=y_fit_model.columns,
            index=X_valid_et.index,
        )

        xgb_pre = fit_xgb_preprocessor(
            X_fit,
            y_fit_model,
            corr_threshold=cfg.xgb_corr_threshold,
            ratio_eps=cfg.xgb_ratio_eps,
            signal_quantile=cfg.xgb_signal_quantile,
            max_selected_features=cfg.xgb_max_selected_features,
        )
        X_fit_xgb = xgb_pre.transform(X_fit, ratio_eps=cfg.xgb_ratio_eps)
        X_valid_xgb = xgb_pre.transform(X_valid, ratio_eps=cfg.xgb_ratio_eps)

        xgb_model = make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        xgb_model.fit(X_fit_xgb, y_fit_model)
        xgb_valid_pred_model = pd.DataFrame(
            xgb_model.predict(X_valid_xgb),
            columns=y_fit_model.columns,
            index=X_valid_xgb.index,
        )

        metrics: dict[str, float] = {}
        for current_weight in weight_candidates:
            blend_valid_model = (current_weight * et_valid_pred_model) + ((1.0 - current_weight) * xgb_valid_pred_model)
            blend_valid_full = schema.expand_predictions(blend_valid_model)
            metrics[f"rmse_et_weight_{current_weight:.2f}"] = float(competition_rmse(y_valid_full, blend_valid_full))

        fold_reports.append(
            {
                "fold": fold_idx,
                **metrics,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "et_feature_count_after_pruning": int(X_fit_et.shape[1]),
                "xgb_feature_count_after_pruning": int(X_fit_xgb.shape[1]),
                "best_et_weight": float(et_weight),
                "best_xgb_weight": float(1.0 - et_weight),
            }
        )

    cv_summary: dict[str, float] = {}
    for current_weight in weight_candidates:
        scores = [fold[f"rmse_et_weight_{current_weight:.2f}"] for fold in fold_reports]
        label = f"et_weight_{current_weight:.2f}"
        cv_summary[f"mean_rmse_{label}"] = float(np.mean(scores))
        cv_summary[f"std_rmse_{label}"] = float(np.std(scores))
        cv_summary[f"min_rmse_{label}"] = float(np.min(scores))
        cv_summary[f"max_rmse_{label}"] = float(np.max(scores))
    return fold_reports, cv_summary


def fit_full_and_predict_blend(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: BlendConfig,
    xgb_params: dict,
    et_weight: float,
) -> tuple[pd.DataFrame, dict]:
    et_pre = fit_feature_preprocessor(
        X_train_raw,
        y_train_model,
        corr_threshold=cfg.et_corr_threshold,
        ratio_eps=cfg.et_ratio_eps,
        signal_quantile=cfg.et_signal_quantile,
        max_selected_features=cfg.et_max_selected_features,
    )
    X_train_et = et_pre.transform(X_train_raw)
    X_test_et = et_pre.transform(X_test_raw)

    et_model = make_et_model(cfg.et_params)
    et_model.fit(X_train_et, y_train_model)
    et_test_pred_model = pd.DataFrame(
        et_model.predict(X_test_et),
        columns=y_train_model.columns,
        index=X_test_et.index,
    )

    xgb_pre = fit_xgb_preprocessor(
        X_train_raw,
        y_train_model,
        corr_threshold=cfg.xgb_corr_threshold,
        ratio_eps=cfg.xgb_ratio_eps,
        signal_quantile=cfg.xgb_signal_quantile,
        max_selected_features=cfg.xgb_max_selected_features,
    )
    X_train_xgb = xgb_pre.transform(X_train_raw, ratio_eps=cfg.xgb_ratio_eps)
    X_test_xgb = xgb_pre.transform(X_test_raw, ratio_eps=cfg.xgb_ratio_eps)

    xgb_model = make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
    xgb_model.fit(X_train_xgb, y_train_model)
    xgb_test_pred_model = pd.DataFrame(
        xgb_model.predict(X_test_xgb),
        columns=y_train_model.columns,
        index=X_test_xgb.index,
    )

    blend_test_model = (et_weight * et_test_pred_model) + ((1.0 - et_weight) * xgb_test_pred_model)
    meta = {
        "et_selected_feature_count": int(len(et_pre.selected_columns)),
        "xgb_selected_feature_count": int(len(xgb_pre.selected_columns)),
        "xgb_selected_columns": xgb_pre.selected_columns,
        "et_weight": float(et_weight),
        "xgb_weight": float(1.0 - et_weight),
    }
    return blend_test_model, meta


def main() -> None:
    args = parse_args()
    global FEATURE_MODE
    FEATURE_MODE = args.feature_mode

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    et_params_json = Path(args.et_params_json)
    if not et_params_json.is_absolute():
        et_params_json = (ROOT / et_params_json).resolve()

    xgb_params_json = None
    if args.xgb_params_json is not None:
        xgb_params_json = Path(args.xgb_params_json)
        if not xgb_params_json.is_absolute():
            xgb_params_json = (ROOT / xgb_params_json).resolve()

    et_params = load_best_params_from_json(et_params_json)
    et_params["random_state"] = args.random_state
    et_params["n_jobs"] = -1

    cfg = BlendConfig(
        et_params=et_params,
        et_corr_threshold=args.et_corr_threshold,
        et_ratio_eps=args.et_ratio_eps,
        et_signal_quantile=args.et_signal_quantile,
        et_max_selected_features=args.et_max_selected_features,
        xgb_corr_threshold=args.xgb_corr_threshold,
        xgb_ratio_eps=args.xgb_ratio_eps,
        xgb_signal_quantile=args.xgb_signal_quantile,
        xgb_max_selected_features=args.xgb_max_selected_features,
        xgb_n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
    )

    bundle = load_modeling_data(data_dir)
    data = bundle.data
    schema = bundle.schema

    X_train_raw = bundle.x_train_raw
    X_test_raw = bundle.x_test_raw
    y_train_full = bundle.y_train_full
    y_train_model = bundle.y_train_model

    if xgb_params_json is not None:
        best_xgb_params = load_xgb_params_from_json(xgb_params_json)
        best_xgb_holdout_rmse = None
    else:
        best_xgb_params, best_xgb_holdout_rmse = optimize_xgb_params(
            X_train_raw,
            y_train_model,
            y_train_full,
            schema,
            cfg,
            n_trials=args.xgb_optuna_trials,
            timeout_sec=args.xgb_optuna_timeout_sec,
            holdout_fraction=args.xgb_optuna_holdout,
        )

    if args.fixed_et_weight is not None:
        best_et_weight = float(args.fixed_et_weight)
        best_blend_holdout_rmse = None
    else:
        best_et_weight, best_blend_holdout_rmse = optimize_blend_weight(
            X_train_raw,
            y_train_model,
            y_train_full,
            schema,
            cfg,
            best_xgb_params,
            n_trials=args.blend_optuna_trials,
            timeout_sec=args.blend_optuna_timeout_sec,
            holdout_fraction=args.blend_optuna_holdout,
        )

    fold_reports, cv_summary = run_cv_blend(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        cfg,
        xgb_params=best_xgb_params,
        et_weight=best_et_weight,
        cv_folds=args.cv_folds,
    )

    blend_test_model, blend_meta = fit_full_and_predict_blend(
        X_train_raw,
        X_test_raw,
        y_train_model,
        cfg,
        xgb_params=best_xgb_params,
        et_weight=best_et_weight,
    )

    pred_test_full = schema.expand_predictions(blend_test_model)
    submission = build_submission_frame(data.x_test["ID"], pred_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_file, index=False)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Blend(ExtraTrees fixed + conservative XGBoost)",
        "data_dir": str(data_dir),
        "et_params_json": str(et_params_json),
        "et_feature_pipeline": {
            "corr_threshold": float(args.et_corr_threshold),
            "signal_quantile": float(args.et_signal_quantile),
            "max_selected_features": int(args.et_max_selected_features),
            "ratio_eps": float(args.et_ratio_eps),
        },
        "xgb_feature_pipeline": {
            "type": (
                "raw + logabs + robust ratios + interactions"
                if FEATURE_MODE == "legacy"
                else "raw features + row aggregated stats (mean/std/percentiles/IQR/MAD/L1/L2)"
            ),
            "corr_threshold": float(args.xgb_corr_threshold),
            "signal_quantile": float(args.xgb_signal_quantile),
            "max_selected_features": int(args.xgb_max_selected_features),
            "ratio_eps": float(args.xgb_ratio_eps),
            "n_jobs": int(args.xgb_n_jobs),
        },
        "xgb_optuna": {
            "enabled": bool(xgb_params_json is None),
            "trials": int(args.xgb_optuna_trials),
            "timeout_sec": int(args.xgb_optuna_timeout_sec),
            "holdout_fraction": float(args.xgb_optuna_holdout),
            "best_holdout_rmse": None if best_xgb_holdout_rmse is None else float(best_xgb_holdout_rmse),
            "best_xgb_params": best_xgb_params,
        },
        "blend_optuna": {
            "enabled": bool(args.fixed_et_weight is None),
            "trials": int(args.blend_optuna_trials),
            "timeout_sec": int(args.blend_optuna_timeout_sec),
            "holdout_fraction": float(args.blend_optuna_holdout),
            "best_holdout_rmse": None if best_blend_holdout_rmse is None else float(best_blend_holdout_rmse),
            "best_et_weight": float(best_et_weight),
            "best_xgb_weight": float(1.0 - best_et_weight),
            "tested_fixed_weights": [0.2, 0.3],
        },
        "cv": {
            "folds": int(args.cv_folds),
            "fold_reports": fold_reports,
            "summary": cv_summary,
        },
        "final_blend": blend_meta,
        "target_handling": {
            "d15_strategy": "constant_target_removed_from_training_via_schema",
            "modeled_targets": schema.model_targets,
            "duplicate_groups": [group for group in schema.duplicate_groups if len(group) > 1],
            "constant_targets": schema.constant_targets,
        },
        "submission_path": str(submission_file.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
