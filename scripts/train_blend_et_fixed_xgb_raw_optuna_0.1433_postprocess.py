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

from odor_competition.data import build_submission_frame, load_modeling_data, raw_features  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


FIXED_ET_FEATURES = [
    "Y1_over_Y2",
    "Z_over_Y1",
    "Y1_over_Y3",
    "Y1",
    "Z_over_Y2",
    "Z_over_Y3",
    "X4_over_Y3",
    "X4_over_Y1",
    "X6_over_Y3",
    "X7_over_Y3",
    "X6_over_Y2",
    "X5_over_Y3",
    "X14_over_Y3",
    "X15_over_Y3",
    "X4_over_Y2",
    "X4_over_Z",
    "X5_over_Y2",
    "X13_over_Y3",
    "X7_over_Y1",
    "X5_over_Y1",
    "X7",
    "X12_over_Y3",
    "X15_over_Y2",
    "X12_over_Y2",
    "X6_over_Y1",
    "X13_over_Y1",
    "X14",
    "X14_over_Y2",
    "X12_over_Y1",
    "X6",
    "X4",
    "X13",
    "X5",
    "X5_over_Z",
    "X14_over_Y1",
    "X13_over_Y2",
    "X12_over_Z",
    "X15_over_Z",
    "X12",
    "X6_over_Z",
    "X14_over_Z",
    "X7_over_Z",
    "X12_over_X7",
    "X13_over_Z",
    "X13_over_X7",
]

FIXED_ET_PARAMS = {
    "n_estimators": 440,
    "max_depth": 20,
    "min_samples_split": 15,
    "min_samples_leaf": 2,
    "max_features": 0.5179947623685809,
    "bootstrap": True,
    "max_samples": 0.6258787450274191,
    "random_state": 42,
    "n_jobs": -1,
}

FIXED_XGB_RAW_COLUMNS = [
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

FIXED_ET_WEIGHT = 0.6340876216153135
FIXED_XGB_WEIGHT = 0.3659123783846865
ET_N_JOBS = 1


@dataclass(frozen=True)
class LinearCalibration:
    slopes: dict[str, float]
    intercepts: dict[str, float]
    train_means: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed best ET + fixed raw-only XGB + fixed blend + OOF linear calibration."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw_best_postprocess")
    parser.add_argument("--submission-prefix", default="blend_et_fixed_xgb_raw_best_postprocess")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--et-n-jobs", type=int, default=1)
    parser.add_argument("--xgb-n-jobs", type=int, default=6)
    args = parser.parse_args()

    if args.cv_folds != 3:
        raise ValueError("Ce script est figé pour une CV=3.")
    if not 0.0 < args.ratio_eps:
        raise ValueError("--ratio-eps must be > 0.")
    return args


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    tiny = np.abs(raw) < eps
    adjusted[tiny] = np.where(raw[tiny] >= 0.0, eps, -eps)
    return adjusted


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_fixed_et_features(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    engineered = base.copy()

    ratio_pairs = sorted(
        {
            tuple(name.split("_over_"))
            for name in FIXED_ET_FEATURES
            if "_over_" in name
        }
    )

    for left, right in ratio_pairs:
        denom = _safe_denominator(base[right], eps)
        ratio = base[left].to_numpy(dtype=float) / denom
        engineered[f"{left}_over_{right}"] = _signed_log1p(ratio)

    return engineered[FIXED_ET_FEATURES].copy()


def build_fixed_xgb_features(features: pd.DataFrame) -> pd.DataFrame:
    base = raw_features(features)
    return base[FIXED_XGB_RAW_COLUMNS].copy()


def make_et_model(random_state: int) -> ExtraTreesRegressor:
    params = dict(FIXED_ET_PARAMS)
    params["random_state"] = random_state
    params["n_jobs"] = ET_N_JOBS
    return ExtraTreesRegressor(**params)


def make_xgb_model(random_state: int, n_jobs: int) -> MultiOutputRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        **FIXED_XGB_PARAMS,
    )
    return MultiOutputRegressor(model, n_jobs=1)


def fit_split_models(
    X_fit: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    *,
    ratio_eps: float,
    random_state: int,
    xgb_n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_fit_et = build_fixed_et_features(X_fit, eps=ratio_eps)
    X_eval_et = build_fixed_et_features(X_eval, eps=ratio_eps)

    et_model = make_et_model(random_state=random_state)
    et_model.fit(X_fit_et, y_fit_model)
    et_pred = pd.DataFrame(
        et_model.predict(X_eval_et),
        columns=y_fit_model.columns,
        index=X_eval.index,
    )

    X_fit_xgb = build_fixed_xgb_features(X_fit)
    X_eval_xgb = build_fixed_xgb_features(X_eval)

    xgb_model = make_xgb_model(random_state=random_state, n_jobs=xgb_n_jobs)
    xgb_model.fit(X_fit_xgb, y_fit_model)
    xgb_pred = pd.DataFrame(
        xgb_model.predict(X_eval_xgb),
        columns=y_fit_model.columns,
        index=X_eval.index,
    )

    return et_pred, xgb_pred


def blend_predictions(et_pred: pd.DataFrame, xgb_pred: pd.DataFrame) -> pd.DataFrame:
    return (FIXED_ET_WEIGHT * et_pred) + (FIXED_XGB_WEIGHT * xgb_pred)


def fit_linear_calibration(
    oof_pred_model: pd.DataFrame,
    y_true_model: pd.DataFrame,
) -> LinearCalibration:
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    train_means: dict[str, float] = {}

    for column in y_true_model.columns:
        pred = oof_pred_model[column].to_numpy(dtype=float)
        truth = y_true_model[column].to_numpy(dtype=float)
        pred_mean = float(np.mean(pred))
        truth_mean = float(np.mean(truth))

        if float(np.std(pred)) < 1e-12:
            slope = 1.0
            intercept = truth_mean - pred_mean
        else:
            design = np.column_stack([pred, np.ones_like(pred)])
            slope, intercept = np.linalg.lstsq(design, truth, rcond=None)[0]

        slopes[column] = float(slope)
        intercepts[column] = float(intercept)
        train_means[column] = truth_mean

    return LinearCalibration(
        slopes=slopes,
        intercepts=intercepts,
        train_means=train_means,
    )


def apply_linear_calibration(pred_model: pd.DataFrame, calibration: LinearCalibration) -> pd.DataFrame:
    calibrated = pred_model.copy()
    for column in pred_model.columns:
        raw_pred = pred_model[column].to_numpy(dtype=float)
        linear_pred = calibration.slopes[column] * raw_pred + calibration.intercepts[column]
        calibrated[column] = linear_pred
    return calibrated


def run_cv_blend(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    cv_folds: int,
    ratio_eps: float,
    random_state: int,
    xgb_n_jobs: int,
) -> tuple[list[dict], dict, pd.DataFrame]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_reports: list[dict] = []
    oof_blend_model = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        et_pred, xgb_pred = fit_split_models(
            X_fit,
            X_valid,
            y_fit_model,
            ratio_eps=ratio_eps,
            random_state=random_state,
            xgb_n_jobs=xgb_n_jobs,
        )
        blend_pred = blend_predictions(et_pred, xgb_pred)
        oof_blend_model.iloc[valid_idx] = blend_pred.to_numpy(dtype=float)

        blend_full = schema.expand_predictions(blend_pred)
        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse_raw_blend": float(competition_rmse(y_valid_full, blend_full)),
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "et_feature_count": int(len(FIXED_ET_FEATURES)),
                "xgb_raw_feature_count": int(len(FIXED_XGB_RAW_COLUMNS)),
                "et_weight": float(FIXED_ET_WEIGHT),
                "xgb_weight": float(FIXED_XGB_WEIGHT),
            }
        )

    scores = np.array([fold["rmse_raw_blend"] for fold in fold_reports], dtype=float)
    summary = {
        "mean_rmse_raw_blend": float(np.mean(scores)),
        "std_rmse_raw_blend": float(np.std(scores)),
        "min_rmse_raw_blend": float(np.min(scores)),
        "max_rmse_raw_blend": float(np.max(scores)),
    }
    return fold_reports, summary, oof_blend_model


def fit_full_and_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    ratio_eps: float,
    random_state: int,
    xgb_n_jobs: int,
) -> pd.DataFrame:
    et_pred, xgb_pred = fit_split_models(
        X_train_raw,
        X_test_raw,
        y_train_model,
        ratio_eps=ratio_eps,
        random_state=random_state,
        xgb_n_jobs=xgb_n_jobs,
    )
    return blend_predictions(et_pred, xgb_pred)


def calibration_to_json(calibration: LinearCalibration, ordered_targets: list[str]) -> dict[str, dict[str, float]]:
    return {
        target: {
            "slope": calibration.slopes[target],
            "intercept": calibration.intercepts[target],
            "train_mean": calibration.train_means[target],
        }
        for target in ordered_targets
    }


def main() -> None:
    global ET_N_JOBS
    args = parse_args()
    ET_N_JOBS = int(args.et_n_jobs)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    bundle = load_modeling_data(data_dir)
    data = bundle.data
    schema = bundle.schema

    X_train_raw = bundle.x_train_raw
    X_test_raw = bundle.x_test_raw
    y_train_full = bundle.y_train_full
    y_train_model = bundle.y_train_model

    fold_reports, cv_summary, oof_blend_model = run_cv_blend(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        cv_folds=args.cv_folds,
        ratio_eps=args.ratio_eps,
        random_state=args.random_state,
        xgb_n_jobs=args.xgb_n_jobs,
    )

    oof_raw_full = schema.expand_predictions(oof_blend_model)
    oof_rmse_before = float(competition_rmse(y_train_full, oof_raw_full))

    calibration = fit_linear_calibration(
        oof_blend_model,
        y_train_model,
    )
    oof_postprocessed_model = apply_linear_calibration(oof_blend_model, calibration)
    oof_postprocessed_full = schema.expand_predictions(oof_postprocessed_model)
    oof_rmse_after = float(competition_rmse(y_train_full, oof_postprocessed_full))

    test_blend_model = fit_full_and_predict(
        X_train_raw,
        X_test_raw,
        y_train_model,
        ratio_eps=args.ratio_eps,
        random_state=args.random_state,
        xgb_n_jobs=args.xgb_n_jobs,
    )
    test_postprocessed_model = apply_linear_calibration(test_blend_model, calibration)
    pred_test_full = schema.expand_predictions(test_postprocessed_model)
    submission = build_submission_frame(data.x_test["ID"], pred_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_file, index=False)

    oof_raw_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_raw_model.csv"
    oof_blend_model.to_csv(oof_raw_file, index=True)

    oof_postprocessed_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_postprocessed_model.csv"
    oof_postprocessed_model.to_csv(oof_postprocessed_file, index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Fixed ET best + fixed XGB raw best + fixed blend + OOF linear calibration",
        "data_dir": str(data_dir),
        "cv": {
            "folds": int(args.cv_folds),
            "fold_reports": fold_reports,
            "summary": {
                **cv_summary,
                "oof_rmse_before_postprocess": oof_rmse_before,
                "oof_rmse_after_postprocess": oof_rmse_after,
            },
        },
        "et": {
            "params": FIXED_ET_PARAMS,
            "feature_count": int(len(FIXED_ET_FEATURES)),
            "fixed_features": FIXED_ET_FEATURES,
            "ratio_eps": float(args.ratio_eps),
        },
        "xgb": {
            "params": FIXED_XGB_PARAMS,
            "raw_feature_count": int(len(FIXED_XGB_RAW_COLUMNS)),
            "raw_columns": FIXED_XGB_RAW_COLUMNS,
            "n_jobs": int(args.xgb_n_jobs),
        },
        "blend": {
            "et_weight": float(FIXED_ET_WEIGHT),
            "xgb_weight": float(FIXED_XGB_WEIGHT),
        },
        "postprocess": {
            "type": "linear_calibration_per_target",
            "targets": calibration_to_json(calibration, list(y_train_model.columns)),
        },
        "target_handling": {
            "d15_strategy": "constant_target_removed_from_training_via_schema",
            "modeled_targets": schema.model_targets,
            "duplicate_groups": [group for group in schema.duplicate_groups if len(group) > 1],
            "constant_targets": schema.constant_targets,
        },
        "submission_path": str(submission_file.relative_to(ROOT)),
        "oof_raw_model_path": str(oof_raw_file.relative_to(ROOT)),
        "oof_postprocessed_model_path": str(oof_postprocessed_file.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
