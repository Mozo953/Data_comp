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
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    infer_target_schema,
    load_competition_data,
    raw_features,
    feature_target_signal,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


ANCHOR_FEATURES = ["Y1", "Y2", "Y3", "Z", "X4", "X5", "X6", "X7", "X12", "X13", "X14", "X15"]
LOG_RATIO_PAIRS = [("Y1", "Y2"), ("Z", "Y1"), ("Y1", "Y3"), ("Z", "Y2"), ("Z", "Y3")]
DIFF_PAIRS = [("Y1", "Y2"), ("Y1", "Y3")]


@dataclass(frozen=True)
class ETFeaturePreprocessor:
    selected_columns: list[str]

    def transform(self, features: pd.DataFrame, ratio_eps: float) -> pd.DataFrame:
        expanded = build_ratio_features(features, eps=ratio_eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class XGBFeaturePreprocessor:
    selected_columns: list[str]

    def transform(self, features: pd.DataFrame, ratio_eps: float) -> pd.DataFrame:
        expanded = build_log_ratio_features(features, eps=ratio_eps)
        return expanded[self.selected_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend best ExtraTrees + best XGB with simple CV averaging, no Ridge meta-learner."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--et-params-json",
        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json",
    )
    parser.add_argument(
        "--xgb-params-json",
        default="artifacts_extratrees_corr_optuna/11_experiment_xgb_alone_ratio_log/xgb_alone_ratio_log_cv3_t10_20260409T152703Z.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/12_experiments_blend_et_xgb_holdout_ridge",
    )
    parser.add_argument("--report-prefix", default="blend_holdout_ridge_cv3")

    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--ratio-eps", type=float, default=1e-3)

    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.2)
    parser.add_argument("--et-max-selected-features", type=int, default=45)

    parser.add_argument("--xgb-corr-threshold", type=float, default=0.99)
    parser.add_argument("--xgb-signal-quantile", type=float, default=0.25)
    parser.add_argument("--xgb-max-selected-features", type=int, default=20)
    parser.add_argument("--xgb-min-feature-std", type=float, default=1e-6)
    parser.add_argument("--xgb-max-tail-ratio", type=float, default=250.0)

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be in (0, 1)")
    if not 0.0 < args.xgb_corr_threshold < 1.0:
        raise ValueError("--xgb-corr-threshold must be in (0, 1)")

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
    for noisy_key in ["et_weight", "xgb_weight", "weight", "blend_weight", "verbose"]:
        cleaned.pop(noisy_key, None)

    if "verbosity" not in cleaned:
        cleaned["verbosity"] = 0
    cleaned["random_state"] = int(cleaned.get("random_state", 42))
    cleaned["n_jobs"] = int(cleaned.get("n_jobs", -1))
    if "n_estimators" in cleaned:
        cleaned["n_estimators"] = int(cleaned["n_estimators"])
    if "max_depth" in cleaned:
        cleaned["max_depth"] = int(cleaned["max_depth"])
    return cleaned


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    tiny = np.abs(raw) < eps
    adjusted[tiny] = np.where(raw[tiny] >= 0.0, eps, -eps)
    return adjusted


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_ratio_features(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    expanded = base.copy()
    cols = list(base.columns)

    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            denom = _safe_denominator(base[right], eps)
            ratio = base[left].to_numpy(dtype=float) / denom
            expanded[f"{left}_over_{right}"] = _signed_log1p(ratio)

    return expanded


def build_log_ratio_features(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    engineered = base.copy()

    for column in ANCHOR_FEATURES:
        values = base[column].to_numpy(dtype=float)
        engineered[f"{column}_logabs"] = _signed_log1p(values)
        engineered[f"{column}_sq"] = np.square(values)
        engineered[f"{column}_inv"] = np.sign(values) / (np.abs(values) + eps)

    for left, right in LOG_RATIO_PAIRS:
        denom = _safe_denominator(base[right], eps)
        ratio = base[left].to_numpy(dtype=float) / denom
        engineered[f"log_ratio_{left}_over_{right}"] = _signed_log1p(ratio)

    for left, right in DIFF_PAIRS:
        engineered[f"{left}_minus_{right}"] = base[left] - base[right]

    return engineered


def _prune_correlated_by_order(
    features: pd.DataFrame,
    ordered_columns: list[str],
    threshold: float,
) -> list[str]:
    corr = features[ordered_columns].corr().abs().fillna(0.0)
    kept: list[str] = []

    for col in ordered_columns:
        if not any(corr.loc[col, other] >= threshold for other in kept):
            kept.append(col)

    return kept


def fit_et_preprocessor(
    X_fit: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    *,
    ratio_eps: float,
    signal_quantile: float,
    corr_threshold: float,
    max_selected_features: int,
) -> ETFeaturePreprocessor:
    expanded = build_ratio_features(X_fit, eps=ratio_eps)
    signal = feature_target_signal(expanded, y_fit_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    kept = _prune_correlated_by_order(expanded, ordered, corr_threshold)
    selected = kept[:max_selected_features]
    return ETFeaturePreprocessor(selected_columns=selected)


def fit_xgb_preprocessor(
    X_fit: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    *,
    ratio_eps: float,
    signal_quantile: float,
    corr_threshold: float,
    max_selected_features: int,
    min_feature_std: float,
    max_tail_ratio: float,
) -> XGBFeaturePreprocessor:
    expanded = build_log_ratio_features(X_fit, eps=ratio_eps)

    stable_columns: list[str] = []
    for col in expanded.columns:
        values = expanded[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            continue

        std = float(np.std(values))
        if std < min_feature_std:
            continue

        abs_values = np.abs(values)
        p50 = float(np.quantile(abs_values, 0.5))
        p99 = float(np.quantile(abs_values, 0.99))
        denom = max(p50, ratio_eps)
        tail_ratio = p99 / denom
        if tail_ratio > max_tail_ratio:
            continue

        stable_columns.append(col)

    filtered = expanded[stable_columns].copy()
    signal = feature_target_signal(filtered, y_fit_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    kept = _prune_correlated_by_order(filtered, ordered, corr_threshold)
    selected = kept[:max_selected_features]

    return XGBFeaturePreprocessor(selected_columns=selected)


def evaluate_cv(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    et_params: dict,
    xgb_params: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    fold_reports: list[dict] = []

    for fold_idx, (dev_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_dev = X_train_raw.iloc[dev_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_dev_model = y_train_model.iloc[dev_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        et_pre = fit_et_preprocessor(
            X_dev,
            y_dev_model,
            ratio_eps=args.ratio_eps,
            signal_quantile=args.et_signal_quantile,
            corr_threshold=args.et_corr_threshold,
            max_selected_features=args.et_max_selected_features,
        )
        X_dev_et = et_pre.transform(X_dev, args.ratio_eps)
        X_valid_et = et_pre.transform(X_valid, args.ratio_eps)

        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_dev_et, y_dev_model)

        pred_valid_et = pd.DataFrame(et_model.predict(X_valid_et), columns=y_train_model.columns, index=X_valid.index)

        xgb_pre = fit_xgb_preprocessor(
            X_dev,
            y_dev_model,
            ratio_eps=args.ratio_eps,
            signal_quantile=args.xgb_signal_quantile,
            corr_threshold=args.xgb_corr_threshold,
            max_selected_features=args.xgb_max_selected_features,
            min_feature_std=args.xgb_min_feature_std,
            max_tail_ratio=args.xgb_max_tail_ratio,
        )
        X_dev_xgb = xgb_pre.transform(X_dev, args.ratio_eps)
        X_valid_xgb = xgb_pre.transform(X_valid, args.ratio_eps)

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_dev_xgb, y_dev_model)

        pred_valid_xgb = pd.DataFrame(xgb_model.predict(X_valid_xgb), columns=y_train_model.columns, index=X_valid.index)

        pred_valid_blend_model = pd.DataFrame(
            0.5 * pred_valid_et.to_numpy(dtype=float) + 0.5 * pred_valid_xgb.to_numpy(dtype=float),
            columns=y_train_model.columns,
            index=X_valid.index,
        )

        pred_valid_blend_full = schema.expand_predictions(pred_valid_blend_model)
        pred_valid_et_full = schema.expand_predictions(pred_valid_et)
        pred_valid_xgb_full = schema.expand_predictions(pred_valid_xgb)

        rmse_blend = float(competition_rmse(y_valid_full, pred_valid_blend_full))
        rmse_et = float(competition_rmse(y_valid_full, pred_valid_et_full))
        rmse_xgb = float(competition_rmse(y_valid_full, pred_valid_xgb_full))

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse_blend_mean": rmse_blend,
                "rmse_et": rmse_et,
                "rmse_xgb": rmse_xgb,
                "valid_rows": int(len(X_valid)),
                "et_feature_count": int(X_dev_et.shape[1]),
                "xgb_feature_count": int(X_dev_xgb.shape[1]),
            }
        )

    meta_scores = np.array([f["rmse_blend_mean"] for f in fold_reports], dtype=float)
    et_scores = np.array([f["rmse_et"] for f in fold_reports], dtype=float)
    xgb_scores = np.array([f["rmse_xgb"] for f in fold_reports], dtype=float)

    summary = {
        "blend_mean_rmse": float(np.mean(meta_scores)),
        "blend_std_rmse": float(np.std(meta_scores)),
        "blend_min_rmse": float(np.min(meta_scores)),
        "blend_max_rmse": float(np.max(meta_scores)),
        "et_mean_rmse": float(np.mean(et_scores)),
        "xgb_mean_rmse": float(np.mean(xgb_scores)),
    }
    return fold_reports, summary


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    et_params_json = Path(args.et_params_json)
    if not et_params_json.is_absolute():
        et_params_json = (ROOT / et_params_json).resolve()

    xgb_params_json = Path(args.xgb_params_json)
    if not xgb_params_json.is_absolute():
        xgb_params_json = (ROOT / xgb_params_json).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train_raw = raw_features(data.x_train)
    y_train_full = data.y_train.drop(columns=["ID"]) if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    et_params = load_best_params_from_json(et_params_json)
    et_params["random_state"] = args.random_state
    et_params["n_jobs"] = -1

    xgb_params = sanitize_xgb_params(load_best_params_from_json(xgb_params_json))
    xgb_params["random_state"] = args.random_state

    fold_reports, summary = evaluate_cv(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        et_params,
        xgb_params,
        args,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "generated_at_utc": timestamp,
        "experiment": "blend_et_xgb_simple_mean_cv",
        "cv_folds": int(args.cv_folds),
        "et_params_source": str(et_params_json.relative_to(ROOT)),
        "xgb_params_source": str(xgb_params_json.relative_to(ROOT)),
        "et_params": et_params,
        "xgb_params": xgb_params,
        "fold_reports": fold_reports,
        "summary": summary,
    }

    report_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
    report_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
