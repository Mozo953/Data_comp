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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
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


MODEL10_NAME = "Model10 ET-main + XGB(raw+miniFE) + Ridge-OOF"


@dataclass(frozen=True)
class ETFeaturePreprocessor:
    selected_columns: list[str]
    dropped_correlated_columns: list[str]

    def transform(self, features: pd.DataFrame, ratio_eps: float) -> pd.DataFrame:
        expanded = build_ratio_features(features, eps=ratio_eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class MiniFeatureSelection:
    selected_engineered_columns: list[str]
    best_k: int
    grid_results: list[dict]


@dataclass(frozen=True)
class Model10Config:
    et_base_params: dict
    et_corr_threshold: float
    et_signal_quantile: float
    et_max_selected_features: int
    ratio_eps: float
    xgb_max_engineered_features: int
    xgb_engineered_grid: list[int]
    xgb_n_jobs: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model 10: ET main + XGB(raw+mini engineered) + Ridge OOF stacker."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--et-params-json",
        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json",
    )
    parser.add_argument(
        "--xgb-init-params-json",
        default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw_best(0.1434)/xgb_trial11_best_params.json",
        help="Base XGB params used for mini-feature grid/correlation screening before Optuna.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/10_model10_et_main_xgb_mini_fe_ridge_oof",
    )
    parser.add_argument("--submission-prefix", default="model10_et_main_xgb_mini_fe_ridge")

    parser.add_argument("--cv-folds", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.20)
    parser.add_argument("--et-max-selected-features", type=int, default=45)

    parser.add_argument(
        "--xgb-max-engineered-features",
        type=int,
        default=10,
        help="Hard cap for engineered features injected into XGB on top of raw features.",
    )
    parser.add_argument(
        "--xgb-engineered-grid",
        default="0,2,4,6,8,10",
        help="Comma-separated candidate counts tested by grid search after correlation ranking.",
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=6)

    parser.add_argument("--xgb-optuna-trials", type=int, default=50)
    parser.add_argument("--xgb-optuna-timeout-sec", type=int, default=1200)
    parser.add_argument("--xgb-optuna-holdout", type=float, default=0.2)

    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--meta-optuna-trials", type=int, default=5)
    parser.add_argument("--meta-optuna-timeout-sec", type=int, default=300)
    parser.add_argument("--meta-optuna-holdout", type=float, default=0.2)
    parser.add_argument(
        "--save-aux-outputs",
        action="store_true",
        help="Save OOF CSVs and the JSON summary in addition to the final submission CSV.",
    )

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
    if args.xgb_max_engineered_features < 0:
        raise ValueError("--xgb-max-engineered-features must be >= 0")
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
    if args.ridge_alpha <= 0.0:
        raise ValueError("--ridge-alpha must be > 0")

    grid_values: list[int] = []
    for token in args.xgb_engineered_grid.split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed < 0:
            raise ValueError("--xgb-engineered-grid values must be >= 0")
        if parsed <= args.xgb_max_engineered_features:
            grid_values.append(parsed)

    if not grid_values:
        raise ValueError("--xgb-engineered-grid has no valid value <= --xgb-max-engineered-features")

    args.xgb_engineered_grid = sorted(set(grid_values))
    return args


def load_best_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        return payload["optuna"]["best_params"]
    if "best_params" in payload:
        return payload["best_params"]
    raise KeyError(f"No best_params found in {json_path}")


def sanitize_xgb_params(params: dict) -> dict:
    cleaned = dict(params)
    for noisy_key in ["et_weight", "xgb_weight", "weight", "blend_weight"]:
        cleaned.pop(noisy_key, None)
    return cleaned


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    tiny = np.abs(raw) < eps
    adjusted[tiny] = np.where(raw[tiny] >= 0.0, eps, -eps)
    return adjusted


def build_ratio_features(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    expanded = base.copy()
    columns = list(base.columns)

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            denom = _safe_denominator(base[right], eps)
            ratio = base[left].to_numpy(dtype=float) / denom
            expanded[f"ratio_{left}_over_{right}"] = np.sign(ratio) * np.log1p(np.abs(ratio))

    return expanded


def build_engineered_feature_bank(features: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
    base = raw_features(features)
    engineered = pd.DataFrame(index=base.index)

    for col in base.columns:
        values = base[col].to_numpy(dtype=float)
        engineered[f"logabs_{col}"] = np.sign(values) * np.log1p(np.abs(values))

    for col in base.columns:
        engineered[f"sq_{col}"] = np.square(base[col].to_numpy(dtype=float))

    block_a = ["X12", "X13", "X14", "X15"]
    block_b = ["X4", "X5", "X6", "X7"]
    support = ["Y1", "Y2", "Y3", "Z"]

    engineered["mean_block_a"] = base[block_a].mean(axis=1)
    engineered["mean_block_b"] = base[block_b].mean(axis=1)
    engineered["mean_support"] = base[support].mean(axis=1)
    engineered["mean_all_raw"] = base.mean(axis=1)

    ratio_pairs = [
        ("X12", "X4"),
        ("X13", "X5"),
        ("X14", "X6"),
        ("X15", "X7"),
        ("X12", "X13"),
        ("X13", "X14"),
        ("X14", "X15"),
        ("X4", "X5"),
        ("X5", "X6"),
        ("X6", "X7"),
        ("Y1", "Y2"),
        ("Y2", "Y3"),
        ("Z", "Y1"),
        ("Z", "Y3"),
    ]
    for left, right in ratio_pairs:
        denom = _safe_denominator(base[right], ratio_eps)
        ratio = base[left].to_numpy(dtype=float) / denom
        engineered[f"ratio_{left}_over_{right}"] = np.sign(ratio) * np.log1p(np.abs(ratio))

    return engineered


def fit_et_preprocessor(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
) -> ETFeaturePreprocessor:
    expanded = build_ratio_features(X_train_raw, eps=ratio_eps)
    signal = feature_target_signal(expanded, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)

    corr = expanded[ordered].corr().abs().fillna(0.0)
    selected: list[str] = []
    dropped: list[str] = []
    for col in ordered:
        if any(corr.loc[col, keep] >= corr_threshold for keep in selected):
            dropped.append(col)
        else:
            selected.append(col)

    if len(selected) > max_selected_features:
        selected = selected[:max_selected_features]

    return ETFeaturePreprocessor(
        selected_columns=selected,
        dropped_correlated_columns=dropped,
    )


def moderate_tune_et_params(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: Model10Config,
) -> tuple[dict, dict]:
    X_fit, X_valid, y_fit_model, _y_valid_model, _y_fit_full, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=0.2,
        random_state=cfg.random_state,
    )

    et_pre = fit_et_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=cfg.et_corr_threshold,
        ratio_eps=cfg.ratio_eps,
        signal_quantile=cfg.et_signal_quantile,
        max_selected_features=cfg.et_max_selected_features,
    )

    X_fit_et = et_pre.transform(X_fit, cfg.ratio_eps)
    X_valid_et = et_pre.transform(X_valid, cfg.ratio_eps)

    # Keep ET tuning intentionally small and slightly conservative to reduce hidden-data overfit.
    candidate_grid = [
        {"max_depth": 8, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 0.8},
        {"max_depth": 8, "min_samples_leaf": 2, "min_samples_split": 4, "max_features": 0.8},
        {"max_depth": 10, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 0.8},
        {"max_depth": 10, "min_samples_leaf": 2, "min_samples_split": 4, "max_features": 0.8},
        {"max_depth": 12, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 0.75},
        {"max_depth": 12, "min_samples_leaf": 2, "min_samples_split": 4, "max_features": 0.75},
    ]

    reports: list[dict] = []
    best_score = float("inf")
    best_params = dict(cfg.et_base_params)

    for candidate in candidate_grid:
        trial_params = dict(cfg.et_base_params)
        trial_params.update(candidate)

        model = ExtraTreesRegressor(**trial_params)
        model.fit(X_fit_et, y_fit_model)

        pred_model = pd.DataFrame(
            model.predict(X_valid_et),
            columns=y_fit_model.columns,
            index=X_valid_et.index,
        )
        pred_full = schema.expand_predictions(pred_model)
        rmse = float(competition_rmse(y_valid_full, pred_full))

        report = {
            "max_depth": candidate["max_depth"],
            "min_samples_leaf": int(candidate["min_samples_leaf"]),
            "min_samples_split": int(candidate["min_samples_split"]),
            "max_features": candidate["max_features"],
            "rmse": rmse,
        }
        reports.append(report)

        if rmse < best_score:
            best_score = rmse
            best_params = trial_params

    tuning_meta = {
        "best_holdout_rmse": float(best_score),
        "tested_candidates": len(reports),
        "top5": sorted(reports, key=lambda row: row["rmse"])[:5],
    }
    return best_params, tuning_meta


def make_xgb_model(params: dict, n_jobs: int, random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        **params,
    )


def build_xgb_matrix(
    X_raw: pd.DataFrame,
    selected_engineered: list[str],
    ratio_eps: float,
) -> pd.DataFrame:
    base = raw_features(X_raw)
    if not selected_engineered:
        return base.copy()

    engineered = build_engineered_feature_bank(X_raw, ratio_eps=ratio_eps)
    selected = [col for col in selected_engineered if col in engineered.columns]
    if not selected:
        return base.copy()

    return pd.concat([base, engineered[selected]], axis=1)


def _select_balanced_engineered_columns(ordered_columns: list[str], k: int) -> list[str]:
    if k <= 0:
        return []

    families = {
        "ratio": [c for c in ordered_columns if c.startswith("ratio_")],
        "log": [c for c in ordered_columns if c.startswith("logabs_")],
        "poly": [c for c in ordered_columns if c.startswith("sq_")],
    }

    selected: list[str] = []
    if k >= 3:
        for fam in ["ratio", "log", "poly"]:
            if families[fam]:
                selected.append(families[fam][0])

    for col in ordered_columns:
        if len(selected) >= k:
            break
        if col not in selected:
            selected.append(col)

    return selected[:k]


def select_mini_engineered_features(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: Model10Config,
    xgb_init_params: dict,
) -> MiniFeatureSelection:
    engineered = build_engineered_feature_bank(X_train_raw, ratio_eps=cfg.ratio_eps)
    signal = feature_target_signal(engineered, y_train_model)
    ordered = list(signal.sort_values(ascending=False).index)

    X_fit, X_valid, y_fit_model, _y_valid_model, _y_fit_full, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=0.2,
        random_state=cfg.random_state,
    )

    grid_results: list[dict] = []
    best_k = 0
    best_rmse = float("inf")
    best_cols: list[str] = []

    for k in cfg.xgb_engineered_grid:
        cols = _select_balanced_engineered_columns(ordered, k)
        X_fit_xgb = build_xgb_matrix(X_fit, cols, ratio_eps=cfg.ratio_eps)
        X_valid_xgb = build_xgb_matrix(X_valid, cols, ratio_eps=cfg.ratio_eps)

        preds = pd.DataFrame(index=X_valid_xgb.index, columns=y_fit_model.columns, dtype=float)
        for target in y_fit_model.columns:
            model = make_xgb_model(xgb_init_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
            model.fit(X_fit_xgb, y_fit_model[target])
            preds[target] = model.predict(X_valid_xgb)

        pred_full = schema.expand_predictions(preds)
        rmse = float(competition_rmse(y_valid_full, pred_full))
        grid_results.append({"k_engineered": int(k), "holdout_rmse": rmse})

        if rmse < best_rmse:
            best_rmse = rmse
            best_k = int(k)
            best_cols = cols

    return MiniFeatureSelection(
        selected_engineered_columns=best_cols,
        best_k=best_k,
        grid_results=sorted(grid_results, key=lambda row: row["holdout_rmse"]),
    )


def tune_xgb_optuna(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: Model10Config,
    selected_engineered: list[str],
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, _y_valid_model, _y_fit_full, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=cfg.random_state,
    )

    X_fit_xgb = build_xgb_matrix(X_fit, selected_engineered, ratio_eps=cfg.ratio_eps)
    X_valid_xgb = build_xgb_matrix(X_valid, selected_engineered, ratio_eps=cfg.ratio_eps)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.8),
            "min_child_weight": trial.suggest_float("min_child_weight", 8.0, 40.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 80.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
        }

        preds = pd.DataFrame(index=X_valid_xgb.index, columns=y_fit_model.columns, dtype=float)
        for target in y_fit_model.columns:
            model = make_xgb_model(params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
            model.fit(X_fit_xgb, y_fit_model[target])
            preds[target] = model.predict(X_valid_xgb)

        pred_full = schema.expand_predictions(preds)
        score = float(competition_rmse(y_valid_full, pred_full))
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg.random_state),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    return best_params, float(study.best_value)


def fit_predict_et(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    cfg: Model10Config,
    et_params: dict,
) -> tuple[pd.DataFrame, dict]:
    et_pre = fit_et_preprocessor(
        X_fit_raw,
        y_fit_model,
        corr_threshold=cfg.et_corr_threshold,
        ratio_eps=cfg.ratio_eps,
        signal_quantile=cfg.et_signal_quantile,
        max_selected_features=cfg.et_max_selected_features,
    )
    X_fit_et = et_pre.transform(X_fit_raw, cfg.ratio_eps)
    X_valid_et = et_pre.transform(X_valid_raw, cfg.ratio_eps)

    model = ExtraTreesRegressor(**et_params)
    model.fit(X_fit_et, y_fit_model)
    pred = pd.DataFrame(
        model.predict(X_valid_et),
        columns=y_fit_model.columns,
        index=X_valid_et.index,
    )

    info = {
        "et_feature_count_after_pruning": int(X_fit_et.shape[1]),
        "et_dropped_correlated_count": int(len(et_pre.dropped_correlated_columns)),
    }
    return pred, info


def fit_predict_xgb(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    cfg: Model10Config,
    xgb_params: dict,
    selected_engineered: list[str],
) -> pd.DataFrame:
    X_fit_xgb = build_xgb_matrix(X_fit_raw, selected_engineered, ratio_eps=cfg.ratio_eps)
    X_valid_xgb = build_xgb_matrix(X_valid_raw, selected_engineered, ratio_eps=cfg.ratio_eps)

    pred = pd.DataFrame(index=X_valid_xgb.index, columns=y_fit_model.columns, dtype=float)
    for target in y_fit_model.columns:
        model = make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        model.fit(X_fit_xgb, y_fit_model[target])
        pred[target] = model.predict(X_valid_xgb)

    return pred


def build_meta_features(
    pred_et: pd.DataFrame,
    pred_xgb: pd.DataFrame,
) -> pd.DataFrame:
    et = pred_et.copy()
    xgb = pred_xgb.copy()

    et.columns = [f"et_{c}" for c in et.columns]
    xgb.columns = [f"xgb_{c}" for c in xgb.columns]

    joined = pd.concat([et, xgb], axis=1)

    for c in pred_et.columns:
        joined[f"diff_{c}"] = pred_et[c] - pred_xgb[c]
        joined[f"mean_{c}"] = 0.5 * (pred_et[c] + pred_xgb[c])

    return joined


def run_oof_base_models(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: Model10Config,
    et_params: dict,
    xgb_params: dict,
    selected_engineered: list[str],
    cv_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)

    oof_et = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    oof_xgb = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    fold_reports: list[dict] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit = y_train_model.iloc[fit_idx]

        pred_et, et_info = fit_predict_et(X_fit, X_valid, y_fit, cfg, et_params)
        pred_xgb = fit_predict_xgb(X_fit, X_valid, y_fit, cfg, xgb_params, selected_engineered)

        oof_et.loc[X_valid.index, :] = pred_et
        oof_xgb.loc[X_valid.index, :] = pred_xgb

        fold_reports.append(
            {
                "fold": int(fold_idx),
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "xgb_raw_feature_count": int(raw_features(X_fit).shape[1]),
                "xgb_engineered_feature_count": int(len(selected_engineered)),
                **et_info,
            }
        )

    return oof_et, oof_xgb, fold_reports


def run_oof_meta_ridge(
    meta_features: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cv_folds: int,
    random_state: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, list[dict], dict]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state + 1000)
    oof_meta = pd.DataFrame(index=meta_features.index, columns=y_train_model.columns, dtype=float)

    fold_reports: list[dict] = []
    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(meta_features), start=1):
        X_fit = meta_features.iloc[fit_idx]
        X_valid = meta_features.iloc[valid_idx]
        y_fit = y_train_model.iloc[fit_idx]

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_fit, y_fit)

        pred = pd.DataFrame(
            ridge.predict(X_valid),
            columns=y_train_model.columns,
            index=X_valid.index,
        )
        oof_meta.loc[X_valid.index, :] = pred

        pred_full = schema.expand_predictions(pred)
        true_full = y_train_full.loc[X_valid.index]
        fold_reports.append(
            {
                "fold": int(fold_idx),
                "rmse": float(competition_rmse(true_full, pred_full)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    all_scores = [r["rmse"] for r in fold_reports]
    summary = {
        "mean_rmse": float(np.mean(all_scores)),
        "std_rmse": float(np.std(all_scores)),
        "min_rmse": float(np.min(all_scores)),
        "max_rmse": float(np.max(all_scores)),
    }
    return oof_meta, fold_reports, summary


def tune_meta_ridge_optuna(
    meta_features: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    random_state: int,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, _y_valid_model, y_fit_full, y_valid_full = train_test_split(
        meta_features,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 50.0, log=True)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        model.fit(X_fit, y_fit_model)

        pred_model = pd.DataFrame(
            model.predict(X_valid),
            columns=y_fit_model.columns,
            index=X_valid.index,
        )
        pred_full = schema.expand_predictions(pred_model)
        score = float(competition_rmse(y_valid_full, pred_full))
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state + 77),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best = dict(study.best_params)
    best["alpha"] = float(best["alpha"])
    best["fit_intercept"] = bool(best["fit_intercept"])
    return best, float(study.best_value)


def fit_full_and_predict_test(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: Model10Config,
    et_params: dict,
    xgb_params: dict,
    selected_engineered: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred_et_test, et_info = fit_predict_et(X_train_raw, X_test_raw, y_train_model, cfg, et_params)
    pred_xgb_test = fit_predict_xgb(
        X_train_raw,
        X_test_raw,
        y_train_model,
        cfg,
        xgb_params,
        selected_engineered,
    )

    info = {
        "et_selected_feature_count": int(et_info["et_feature_count_after_pruning"]),
        "xgb_raw_feature_count": int(raw_features(X_train_raw).shape[1]),
        "xgb_engineered_feature_count": int(len(selected_engineered)),
        "xgb_engineered_features": selected_engineered,
    }
    return pred_et_test, pred_xgb_test, info


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    et_params_json = Path(args.et_params_json)
    if not et_params_json.is_absolute():
        et_params_json = (ROOT / et_params_json).resolve()

    xgb_init_params_json = Path(args.xgb_init_params_json)
    if not xgb_init_params_json.is_absolute():
        xgb_init_params_json = (ROOT / xgb_init_params_json).resolve()

    et_base_params = load_best_params_from_json(et_params_json)
    et_base_params["random_state"] = args.random_state
    et_base_params["n_jobs"] = -1
    if "n_estimators" in et_base_params:
        et_base_params["n_estimators"] = min(int(et_base_params["n_estimators"]), 700)
    else:
        et_base_params["n_estimators"] = 700

    xgb_init_params = sanitize_xgb_params(load_best_params_from_json(xgb_init_params_json))

    cfg = Model10Config(
        et_base_params=et_base_params,
        et_corr_threshold=args.et_corr_threshold,
        et_signal_quantile=args.et_signal_quantile,
        et_max_selected_features=args.et_max_selected_features,
        ratio_eps=args.ratio_eps,
        xgb_max_engineered_features=args.xgb_max_engineered_features,
        xgb_engineered_grid=args.xgb_engineered_grid,
        xgb_n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
    )

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train_raw = raw_features(data.x_train)
    X_test_raw = raw_features(data.x_test)

    y_train_full = data.y_train.drop(columns=["ID"]).copy() if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    tuned_et_params, et_tuning_meta = moderate_tune_et_params(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        cfg,
    )

    mini_selection = select_mini_engineered_features(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        cfg,
        xgb_init_params,
    )

    tuned_xgb_params, xgb_holdout_rmse = tune_xgb_optuna(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        cfg,
        selected_engineered=mini_selection.selected_engineered_columns,
        n_trials=args.xgb_optuna_trials,
        timeout_sec=args.xgb_optuna_timeout_sec,
        holdout_fraction=args.xgb_optuna_holdout,
    )

    oof_et, oof_xgb, base_fold_reports = run_oof_base_models(
        X_train_raw,
        y_train_model,
        cfg,
        tuned_et_params,
        tuned_xgb_params,
        mini_selection.selected_engineered_columns,
        cv_folds=args.cv_folds,
    )

    oof_meta_features = build_meta_features(oof_et, oof_xgb)
    best_meta_params, meta_holdout_rmse = tune_meta_ridge_optuna(
        oof_meta_features,
        y_train_model,
        y_train_full,
        schema,
        random_state=args.random_state,
        n_trials=args.meta_optuna_trials,
        timeout_sec=args.meta_optuna_timeout_sec,
        holdout_fraction=args.meta_optuna_holdout,
    )

    oof_ridge, ridge_fold_reports, ridge_summary = run_oof_meta_ridge(
        oof_meta_features,
        y_train_model,
        y_train_full,
        schema,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        ridge_alpha=float(best_meta_params["alpha"]),
    )

    oof_ridge_full = schema.expand_predictions(oof_ridge)
    oof_ridge_rmse = float(competition_rmse(y_train_full, oof_ridge_full))

    pred_et_test, pred_xgb_test, final_base_info = fit_full_and_predict_test(
        X_train_raw,
        X_test_raw,
        y_train_model,
        cfg,
        tuned_et_params,
        tuned_xgb_params,
        mini_selection.selected_engineered_columns,
    )

    meta_test_features = build_meta_features(pred_et_test, pred_xgb_test)
    ridge_final = Ridge(
        alpha=float(best_meta_params["alpha"]),
        fit_intercept=bool(best_meta_params["fit_intercept"]),
    )
    ridge_final.fit(oof_meta_features, y_train_model)

    pred_model_test = pd.DataFrame(
        ridge_final.predict(meta_test_features),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )
    pred_full_test = schema.expand_predictions(pred_model_test)
    submission = build_submission_frame(data.x_test["ID"], pred_full_test)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    oof_et_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_et_model.csv"
    oof_xgb_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_xgb_model.csv"
    oof_ridge_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_ridge_model.csv"

    submission.to_csv(csv_file, index=False)
    if args.save_aux_outputs:
        summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
        oof_et.to_csv(oof_et_file, index=True)
        oof_xgb.to_csv(oof_xgb_file, index=True)
        oof_ridge.to_csv(oof_ridge_file, index=True)

    engineered_family_counts = {
        "ratio": int(sum(1 for c in mini_selection.selected_engineered_columns if c.startswith("ratio_"))),
        "log": int(sum(1 for c in mini_selection.selected_engineered_columns if c.startswith("logabs_"))),
        "poly_sq": int(sum(1 for c in mini_selection.selected_engineered_columns if c.startswith("sq_"))),
        "mean": int(sum(1 for c in mini_selection.selected_engineered_columns if c.startswith("mean_"))),
    }

    summary = {
        "generated_at_utc": timestamp,
        "model_name": MODEL10_NAME,
        "model": "ET main + XGB(raw + mini engineered <=10) + Ridge meta-learner",
        "data_dir": str(data_dir),
        "et_params_json": str(et_params_json),
        "xgb_init_params_json": str(xgb_init_params_json),
        "feature_design": {
            "xgb_raw_features": "all raw features",
            "xgb_engineered_limit": int(args.xgb_max_engineered_features),
            "xgb_engineered_grid": args.xgb_engineered_grid,
            "xgb_selected_engineered_count": int(len(mini_selection.selected_engineered_columns)),
            "xgb_selected_engineered_columns": mini_selection.selected_engineered_columns,
            "xgb_selected_engineered_family_counts": engineered_family_counts,
            "selection_method": "correlation ranking + holdout grid search over K engineered",
            "selection_scope": "global pre-oof",
        },
        "et_tuning": {
            "method": "moderate grid tuning",
            "best_params": tuned_et_params,
            **et_tuning_meta,
        },
        "xgb_tuning": {
            "method": "optuna",
            "trials": int(args.xgb_optuna_trials),
            "timeout_sec": int(args.xgb_optuna_timeout_sec),
            "holdout_fraction": float(args.xgb_optuna_holdout),
            "best_holdout_rmse": float(xgb_holdout_rmse),
            "best_params": tuned_xgb_params,
            "mini_feature_grid_results": mini_selection.grid_results,
        },
        "meta_tuning": {
            "method": "optuna",
            "trials": int(args.meta_optuna_trials),
            "timeout_sec": int(args.meta_optuna_timeout_sec),
            "holdout_fraction": float(args.meta_optuna_holdout),
            "best_holdout_rmse": float(meta_holdout_rmse),
            "best_params": best_meta_params,
        },
        "cv": {
            "folds": int(args.cv_folds),
            "base_fold_reports": base_fold_reports,
            "ridge_fold_reports": ridge_fold_reports,
            "ridge_summary": {
                **ridge_summary,
                "oof_rmse": float(oof_ridge_rmse),
            },
        },
        "oof_paths": {
            "oof_et_model_path": str(oof_et_file.relative_to(ROOT)) if args.save_aux_outputs else None,
            "oof_xgb_model_path": str(oof_xgb_file.relative_to(ROOT)) if args.save_aux_outputs else None,
            "oof_ridge_model_path": str(oof_ridge_file.relative_to(ROOT)) if args.save_aux_outputs else None,
        },
        "final_blend": {
            **final_base_info,
            "meta_learner": "Ridge",
            "meta_ridge_alpha": float(best_meta_params["alpha"]),
            "meta_fit_intercept": bool(best_meta_params["fit_intercept"]),
        },
        "target_handling": {
            "d15_strategy": "constant_target_removed_from_training_via_schema",
            "modeled_targets": schema.model_targets,
            "duplicate_groups": [group for group in schema.duplicate_groups if len(group) > 1],
            "constant_targets": schema.constant_targets,
        },
        "submission_path": str(csv_file.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
    }

    if args.save_aux_outputs:
        summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
