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

try:
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OPTUNA_AVAILABLE = False

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
        description="Blend best ExtraTrees + best XGB with a global weighted average and Optuna-tuned weight."
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
        default="artifacts_extratrees_corr_optuna/13_experiments_blend_et_xgb_weighted_global",
    )
    parser.add_argument("--report-prefix", default="blend_weighted_global_cv3")
    parser.add_argument("--submission-prefix", default="blend_weighted_global_cv3")

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

    parser.add_argument("--optuna-trials", type=int, default=15)
    parser.add_argument("--optuna-holdout", type=float, default=0.2)
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=None,
        help="Fixed ET weight in [0,1]. If omitted, Optuna tunes it on a holdout split.",
    )

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be in (0, 1)")
    if not 0.0 < args.xgb_corr_threshold < 1.0:
        raise ValueError("--xgb-corr-threshold must be in (0, 1)")
    if not 0.0 < args.optuna_holdout < 1.0:
        raise ValueError("--optuna-holdout must be in (0, 1)")
    if args.optuna_trials < 1:
        raise ValueError("--optuna-trials must be >= 1")
    if args.blend_weight is not None and not 0.0 <= args.blend_weight <= 1.0:
        raise ValueError("--blend-weight must be in [0, 1]")

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
    cleaned.setdefault("verbosity", 0)
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


def _prune_correlated_by_order(features: pd.DataFrame, ordered_columns: list[str], threshold: float) -> list[str]:
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
    selected = _prune_correlated_by_order(expanded, ordered, corr_threshold)[:max_selected_features]
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
        if p99 / denom > max_tail_ratio:
            continue
        stable_columns.append(col)

    filtered = expanded[stable_columns].copy()
    signal = feature_target_signal(filtered, y_fit_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected = _prune_correlated_by_order(filtered, ordered, corr_threshold)[:max_selected_features]
    return XGBFeaturePreprocessor(selected_columns=selected)


def blend_predictions(pred_et: pd.DataFrame, pred_xgb: pd.DataFrame, et_weight: float) -> pd.DataFrame:
    xgb_weight = 1.0 - et_weight
    blended = et_weight * pred_et.to_numpy(dtype=float) + xgb_weight * pred_xgb.to_numpy(dtype=float)
    return pd.DataFrame(blended, columns=pred_et.columns, index=pred_et.index)


def optimize_blend_weight(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    et_params: dict,
    xgb_params: dict,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    if args.blend_weight is not None:
        return float(args.blend_weight), {"source": "fixed", "blend_weight": float(args.blend_weight)}

    if not OPTUNA_AVAILABLE:
        return 0.5, {"source": "default", "blend_weight": 0.5}

    X_fit, X_valid, y_fit_model, y_valid_model, y_fit_full, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=args.optuna_holdout,
        random_state=args.random_state,
    )

    et_pre = fit_et_preprocessor(
        X_fit,
        y_fit_model,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.et_signal_quantile,
        corr_threshold=args.et_corr_threshold,
        max_selected_features=args.et_max_selected_features,
    )
    xgb_pre = fit_xgb_preprocessor(
        X_fit,
        y_fit_model,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.xgb_signal_quantile,
        corr_threshold=args.xgb_corr_threshold,
        max_selected_features=args.xgb_max_selected_features,
        min_feature_std=args.xgb_min_feature_std,
        max_tail_ratio=args.xgb_max_tail_ratio,
    )

    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(et_pre.transform(X_fit, args.ratio_eps), y_fit_model)
    pred_valid_et = pd.DataFrame(
        et_model.predict(et_pre.transform(X_valid, args.ratio_eps)),
        columns=y_fit_model.columns,
        index=X_valid.index,
    )

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(xgb_pre.transform(X_fit, args.ratio_eps), y_fit_model)
    pred_valid_xgb = pd.DataFrame(
        xgb_model.predict(xgb_pre.transform(X_valid, args.ratio_eps)),
        columns=y_fit_model.columns,
        index=X_valid.index,
    )

    def objective(trial: optuna.Trial) -> float:
        et_weight = trial.suggest_float("et_weight", 0.0, 1.0)
        pred_valid_blend = blend_predictions(pred_valid_et, pred_valid_xgb, et_weight)
        pred_valid_full = schema.expand_predictions(pred_valid_blend)
        score = float(competition_rmse(y_valid_full, pred_valid_full))
        trial.report(score, step=0)
        if trial.should_prune():
            raise TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=True)

    best_weight = float(study.best_params["et_weight"])
    meta = {
        "source": "optuna",
        "trials": int(args.optuna_trials),
        "holdout": float(args.optuna_holdout),
        "best_et_weight": best_weight,
        "best_xgb_weight": float(1.0 - best_weight),
        "holdout_rmse": float(study.best_value),
    }
    return best_weight, meta


def evaluate_cv(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    et_params: dict,
    xgb_params: dict,
    et_weight: float,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    fold_reports: list[dict] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        et_pre = fit_et_preprocessor(
            X_fit,
            y_fit_model,
            ratio_eps=args.ratio_eps,
            signal_quantile=args.et_signal_quantile,
            corr_threshold=args.et_corr_threshold,
            max_selected_features=args.et_max_selected_features,
        )
        X_fit_et = et_pre.transform(X_fit, args.ratio_eps)
        X_valid_et = et_pre.transform(X_valid, args.ratio_eps)

        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_fit_et, y_fit_model)
        pred_valid_et = pd.DataFrame(et_model.predict(X_valid_et), columns=y_train_model.columns, index=X_valid.index)

        xgb_pre = fit_xgb_preprocessor(
            X_fit,
            y_fit_model,
            ratio_eps=args.ratio_eps,
            signal_quantile=args.xgb_signal_quantile,
            corr_threshold=args.xgb_corr_threshold,
            max_selected_features=args.xgb_max_selected_features,
            min_feature_std=args.xgb_min_feature_std,
            max_tail_ratio=args.xgb_max_tail_ratio,
        )
        X_fit_xgb = xgb_pre.transform(X_fit, args.ratio_eps)
        X_valid_xgb = xgb_pre.transform(X_valid, args.ratio_eps)

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_fit_xgb, y_fit_model)
        pred_valid_xgb = pd.DataFrame(xgb_model.predict(X_valid_xgb), columns=y_train_model.columns, index=X_valid.index)

        pred_valid_blend = blend_predictions(pred_valid_et, pred_valid_xgb, et_weight)
        pred_valid_blend_full = schema.expand_predictions(pred_valid_blend)
        pred_valid_et_full = schema.expand_predictions(pred_valid_et)
        pred_valid_xgb_full = schema.expand_predictions(pred_valid_xgb)

        rmse_blend = float(competition_rmse(y_valid_full, pred_valid_blend_full))
        rmse_et = float(competition_rmse(y_valid_full, pred_valid_et_full))
        rmse_xgb = float(competition_rmse(y_valid_full, pred_valid_xgb_full))

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse_blend_weighted": rmse_blend,
                "rmse_et": rmse_et,
                "rmse_xgb": rmse_xgb,
                "valid_rows": int(len(X_valid)),
                "et_feature_count": int(X_fit_et.shape[1]),
                "xgb_feature_count": int(X_fit_xgb.shape[1]),
            }
        )

    blend_scores = np.array([f["rmse_blend_weighted"] for f in fold_reports], dtype=float)
    et_scores = np.array([f["rmse_et"] for f in fold_reports], dtype=float)
    xgb_scores = np.array([f["rmse_xgb"] for f in fold_reports], dtype=float)

    summary = {
        "blend_weighted_mean_rmse": float(np.mean(blend_scores)),
        "blend_weighted_std_rmse": float(np.std(blend_scores)),
        "blend_weighted_min_rmse": float(np.min(blend_scores)),
        "blend_weighted_max_rmse": float(np.max(blend_scores)),
        "et_mean_rmse": float(np.mean(et_scores)),
        "xgb_mean_rmse": float(np.mean(xgb_scores)),
    }
    return fold_reports, summary


def fit_full_and_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    et_params: dict,
    xgb_params: dict,
    et_weight: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, ETFeaturePreprocessor, XGBFeaturePreprocessor, ExtraTreesRegressor, XGBRegressor]:
    et_pre = fit_et_preprocessor(
        X_train_raw,
        y_train_model,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.et_signal_quantile,
        corr_threshold=args.et_corr_threshold,
        max_selected_features=args.et_max_selected_features,
    )
    xgb_pre = fit_xgb_preprocessor(
        X_train_raw,
        y_train_model,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.xgb_signal_quantile,
        corr_threshold=args.xgb_corr_threshold,
        max_selected_features=args.xgb_max_selected_features,
        min_feature_std=args.xgb_min_feature_std,
        max_tail_ratio=args.xgb_max_tail_ratio,
    )

    X_train_et = et_pre.transform(X_train_raw, args.ratio_eps)
    X_test_et = et_pre.transform(X_test_raw, args.ratio_eps)
    X_train_xgb = xgb_pre.transform(X_train_raw, args.ratio_eps)
    X_test_xgb = xgb_pre.transform(X_test_raw, args.ratio_eps)

    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(X_train_et, y_train_model)

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_xgb, y_train_model)

    pred_test_et = pd.DataFrame(et_model.predict(X_test_et), columns=y_train_model.columns, index=X_test_raw.index)
    pred_test_xgb = pd.DataFrame(xgb_model.predict(X_test_xgb), columns=y_train_model.columns, index=X_test_raw.index)
    pred_test_blend = blend_predictions(pred_test_et, pred_test_xgb, et_weight)
    return pred_test_blend, et_pre, xgb_pre, et_model, xgb_model


def summarize_importance(model, columns: list[str]) -> dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {}
    values = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    return {name: float(score) for name, score in values.items()}


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
    X_test_raw = raw_features(data.x_test)
    y_train_full = data.y_train.drop(columns=["ID"]) if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    et_params = load_best_params_from_json(et_params_json)
    et_params["random_state"] = args.random_state
    et_params["n_jobs"] = -1

    xgb_params = sanitize_xgb_params(load_best_params_from_json(xgb_params_json))
    xgb_params["random_state"] = args.random_state

    et_weight, weight_meta = optimize_blend_weight(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        et_params,
        xgb_params,
        args,
    )

    fold_reports, summary = evaluate_cv(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        et_params,
        xgb_params,
        et_weight,
        args,
    )

    pred_test_blend, et_pre, xgb_pre, et_model, xgb_model = fit_full_and_predict(
        X_train_raw,
        X_test_raw,
        y_train_model,
        et_params,
        xgb_params,
        et_weight,
        args,
    )
    pred_test_full = schema.expand_predictions(pred_test_blend)
    submission = build_submission_frame(data.x_test["ID"], pred_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_file, index=False)

    result = {
        "generated_at_utc": timestamp,
        "experiment": "blend_et_xgb_weighted_global_cv",
        "cv_folds": int(args.cv_folds),
        "et_weight": float(et_weight),
        "xgb_weight": float(1.0 - et_weight),
        "weight_tuning": weight_meta,
        "et_params_source": str(et_params_json.relative_to(ROOT)),
        "xgb_params_source": str(xgb_params_json.relative_to(ROOT)),
        "et_params": et_params,
        "xgb_params": xgb_params,
        "cv": {
            "fold_reports": fold_reports,
            "summary": summary,
        },
        "feature_pipeline": {
            "et_selected_features": int(len(et_pre.selected_columns)),
            "xgb_selected_features": int(len(xgb_pre.selected_columns)),
        },
        "submission_path": str(submission_file.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
        "feature_importance": {
            "et": summarize_importance(et_model, et_pre.selected_columns),
            "xgb": summarize_importance(xgb_model, xgb_pre.selected_columns),
        },
    }

    report_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
    report_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
