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
from sklearn.model_selection import KFold
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
class FeaturePreprocessor:
    selected_columns: list[str]
    ratio_columns: list[str]
    dropped_correlated_columns: list[str]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        expanded = build_ratio_features(features)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class BlendConfig:
    et_params: dict
    et_corr_threshold: float
    et_ratio_eps: float
    et_signal_quantile: float
    et_max_selected_features: int
    xgb_max_raw_features: int
    xgb_n_jobs: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF blend for fixed ExtraTrees + XGBoost(raw) with fixed params.")
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--et-params-json",
        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json",
        help="JSON file containing ExtraTrees best_params from prior run.",
    )
    parser.add_argument(
        "--xgb-params-json",
        default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw/xgb_trial11_best_params.json",
        help="JSON file with XGB best_params and blend weight.",
    )
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw_oof")
    parser.add_argument("--submission-prefix", default="blend_trial11_oof")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.20)
    parser.add_argument("--et-max-selected-features", type=int, default=45)
    parser.add_argument("--et-ratio-eps", type=float, default=1e-3)

    parser.add_argument(
        "--xgb-max-raw-features",
        type=int,
        default=13,
        help="Top-k raw features by train-fold target-signal. 13 means all raw features.",
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=6)

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be between 0 and 1.")
    if not 0.0 < args.et_signal_quantile <= 1.0:
        raise ValueError("--et-signal-quantile must be between 0 and 1.")
    if args.et_max_selected_features < 1:
        raise ValueError("--et-max-selected-features must be >= 1.")
    if args.xgb_max_raw_features < 1:
        raise ValueError("--xgb-max-raw-features must be >= 1.")

    return args


def load_best_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        return payload["optuna"]["best_params"]
    if "best_params" in payload:
        return payload["best_params"]
    raise KeyError(f"No best_params found in {json_path}")


def load_blend_params_from_json(json_path: Path) -> tuple[dict, float]:
    payload = json.loads(json_path.read_text())
    if "best_params" not in payload:
        raise KeyError(f"No best_params found in {json_path}")
    params = dict(payload["best_params"])
    if "et_weight" not in params:
        raise KeyError(f"No et_weight found in {json_path}")
    et_weight = float(params.pop("et_weight"))
    return params, et_weight


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    adjusted[np.abs(raw) < eps] = np.where(raw[np.abs(raw) < eps] >= 0.0, eps, -eps)
    return adjusted


def build_ratio_features(features: pd.DataFrame, *, eps: float = 1e-3) -> pd.DataFrame:
    base = raw_features(features)
    expanded = base.copy()
    columns = list(base.columns)

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            denom = _safe_denominator(base[right], eps)
            ratio = base[left].to_numpy(dtype=float) / denom
            expanded[f"{left}_over_{right}"] = np.sign(ratio) * np.log1p(np.abs(ratio))

    return expanded


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


def run_oof_blend(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    cfg: BlendConfig,
    xgb_params: dict,
    et_weight: float,
    cv_folds: int,
) -> tuple[pd.DataFrame, list[dict], dict, dict]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)
    oof_model = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    fold_reports: list[dict] = []

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

        xgb_cols = select_xgb_raw_columns(X_fit, y_fit_model, cfg.xgb_max_raw_features)
        X_fit_xgb = X_fit[xgb_cols].copy()
        X_valid_xgb = X_valid[xgb_cols].copy()

        xgb_model = make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        xgb_model.fit(X_fit_xgb, y_fit_model)
        xgb_valid_pred_model = pd.DataFrame(
            xgb_model.predict(X_valid_xgb),
            columns=y_fit_model.columns,
            index=X_valid_xgb.index,
        )

        blend_valid_model = (et_weight * et_valid_pred_model) + ((1.0 - et_weight) * xgb_valid_pred_model)
        oof_model.loc[X_valid.index, :] = blend_valid_model
        blend_valid_full = schema.expand_predictions(blend_valid_model)
        rmse = competition_rmse(y_valid_full, blend_valid_full)

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse": float(rmse),
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "et_feature_count_after_pruning": int(X_fit_et.shape[1]),
                "xgb_raw_feature_count": int(len(xgb_cols)),
                "et_weight": float(et_weight),
                "xgb_weight": float(1.0 - et_weight),
            }
        )

    oof_full = schema.expand_predictions(oof_model)
    overall_rmse = competition_rmse(y_train_full, oof_full)
    scores = [fold["rmse"] for fold in fold_reports]
    cv_summary = {
        "mean_rmse": float(np.mean(scores)),
        "std_rmse": float(np.std(scores)),
        "min_rmse": float(np.min(scores)),
        "max_rmse": float(np.max(scores)),
        "oof_rmse": float(overall_rmse),
    }
    return oof_model, fold_reports, cv_summary, {
        "oof_full": oof_full,
    }


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

    xgb_cols = select_xgb_raw_columns(X_train_raw, y_train_model, cfg.xgb_max_raw_features)
    X_train_xgb = X_train_raw[xgb_cols].copy()
    X_test_xgb = X_test_raw[xgb_cols].copy()

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
        "xgb_raw_feature_count": int(len(xgb_cols)),
        "xgb_raw_columns": xgb_cols,
        "et_weight": float(et_weight),
        "xgb_weight": float(1.0 - et_weight),
    }
    return blend_test_model, meta


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

    et_params = load_best_params_from_json(et_params_json)
    et_params["random_state"] = args.random_state
    et_params["n_jobs"] = -1

    xgb_params, et_weight = load_blend_params_from_json(xgb_params_json)

    cfg = BlendConfig(
        et_params=et_params,
        et_corr_threshold=args.et_corr_threshold,
        et_ratio_eps=args.et_ratio_eps,
        et_signal_quantile=args.et_signal_quantile,
        et_max_selected_features=args.et_max_selected_features,
        xgb_max_raw_features=args.xgb_max_raw_features,
        xgb_n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
    )

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train_raw = raw_features(data.x_train)
    X_test_raw = raw_features(data.x_test)
    y_train_full = data.y_train.drop(columns=["ID"]).copy() if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    oof_model, fold_reports, cv_summary, oof_payload = run_oof_blend(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        cfg,
        xgb_params=xgb_params,
        et_weight=et_weight,
        cv_folds=args.cv_folds,
    )

    blend_test_model, blend_meta = fit_full_and_predict_blend(
        X_train_raw,
        X_test_raw,
        y_train_model,
        cfg,
        xgb_params=xgb_params,
        et_weight=et_weight,
    )

    pred_test_full = schema.expand_predictions(blend_test_model)
    submission = build_submission_frame(data.x_test["ID"], pred_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_file, index=False)

    oof_model_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_model.csv"
    oof_model.to_csv(oof_model_file, index=True)
    oof_full_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_full.csv"
    oof_payload["oof_full"].to_csv(oof_full_file, index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "OOF Blend(ExtraTrees fixed + XGBoost raw)",
        "data_dir": str(data_dir),
        "et_params_json": str(et_params_json),
        "xgb_params_json": str(xgb_params_json),
        "et_feature_pipeline": {
            "corr_threshold": float(args.et_corr_threshold),
            "signal_quantile": float(args.et_signal_quantile),
            "max_selected_features": int(args.et_max_selected_features),
            "ratio_eps": float(args.et_ratio_eps),
        },
        "xgb_input": {
            "type": "raw_features_only",
            "max_raw_features": int(args.xgb_max_raw_features),
            "n_jobs": int(args.xgb_n_jobs),
        },
        "blend": {
            "et_weight": float(et_weight),
            "xgb_weight": float(1.0 - et_weight),
            "best_xgb_params": xgb_params,
        },
        "cv": {
            "folds": int(args.cv_folds),
            "fold_reports": fold_reports,
            "summary": cv_summary,
        },
        "oof_paths": {
            "oof_model_path": str(oof_model_file.relative_to(ROOT)),
            "oof_full_path": str(oof_full_file.relative_to(ROOT)),
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
