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
    xgb_params: dict
    et_corr_threshold: float
    et_ratio_eps: float
    et_signal_quantile: float
    et_max_selected_features: int
    xgb_n_jobs: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOF stacking: fixed ExtraTrees + XGBoost(same ET feature pipeline) + trained meta-model."
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
    parser.add_argument("--submission-prefix", default="blend_oof_stacking_et_xgb")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.20)
    parser.add_argument("--et-max-selected-features", type=int, default=45)
    parser.add_argument("--et-ratio-eps", type=float, default=1e-3)

    parser.add_argument("--xgb-n-jobs", type=int, default=6)
    parser.add_argument("--xgb-optuna-trials", type=int, default=24)
    parser.add_argument("--xgb-optuna-timeout-sec", type=int, default=900)
    parser.add_argument("--xgb-optuna-holdout", type=float, default=0.2)
    parser.add_argument(
        "--xgb-params-json",
        default=None,
        help="Optional fixed XGB params JSON. If set, Optuna is skipped for XGB.",
    )

    parser.add_argument("--meta-ridge-alpha", type=float, default=1.0)

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be between 0 and 1.")
    if not 0.0 < args.et_signal_quantile <= 1.0:
        raise ValueError("--et-signal-quantile must be between 0 and 1.")
    if args.et_max_selected_features < 1:
        raise ValueError("--et-max-selected-features must be >= 1.")
    if args.xgb_n_jobs < 1:
        raise ValueError("--xgb-n-jobs must be >= 1.")
    if args.xgb_optuna_trials < 1:
        raise ValueError("--xgb-optuna-trials must be >= 1.")
    if args.xgb_optuna_timeout_sec < 1:
        raise ValueError("--xgb-optuna-timeout-sec must be >= 1.")
    if not 0.0 < args.xgb_optuna_holdout < 1.0:
        raise ValueError("--xgb-optuna-holdout must be in (0, 1).")
    if args.meta_ridge_alpha <= 0.0:
        raise ValueError("--meta-ridge-alpha must be > 0.")

    return args


def load_best_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        return payload["optuna"]["best_params"]
    if "best_params" in payload:
        return payload["best_params"]
    raise KeyError(f"No best_params found in {json_path}")


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
    et_params: dict,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
    random_state: int,
    xgb_n_jobs: int,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, _, _, y_valid_full = train_test_split(
        X_train_raw,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    pre = fit_feature_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=corr_threshold,
        ratio_eps=ratio_eps,
        signal_quantile=signal_quantile,
        max_selected_features=max_selected_features,
    )
    X_fit_trans = pre.transform(X_fit)
    X_valid_trans = pre.transform(X_valid)

    et_model = make_et_model(et_params)
    et_model.fit(X_fit_trans, y_fit_model)
    et_valid_pred_model = pd.DataFrame(
        et_model.predict(X_valid_trans),
        columns=y_fit_model.columns,
        index=X_valid_trans.index,
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

        xgb_model = make_xgb_model(xgb_params, n_jobs=xgb_n_jobs, random_state=random_state)
        xgb_model.fit(X_fit_trans, y_fit_model)
        xgb_valid_pred_model = pd.DataFrame(
            xgb_model.predict(X_valid_trans),
            columns=y_fit_model.columns,
            index=X_valid_trans.index,
        )

        # Quick but robust objective: keep ET fixed, assess blend potential on validation.
        blend_valid = 0.5 * et_valid_pred_model + 0.5 * xgb_valid_pred_model
        blend_valid_full = schema.expand_predictions(blend_valid)
        score = float(competition_rmse(y_valid_full, blend_valid_full))

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

    best = study.best_params.copy()
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    return best, float(study.best_value)


def run_oof_base_models(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: BlendConfig,
    cv_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)

    oof_et = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    oof_xgb = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    fold_reports: list[dict] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]

        pre = fit_feature_preprocessor(
            X_fit,
            y_fit_model,
            corr_threshold=cfg.et_corr_threshold,
            ratio_eps=cfg.et_ratio_eps,
            signal_quantile=cfg.et_signal_quantile,
            max_selected_features=cfg.et_max_selected_features,
        )
        X_fit_trans = pre.transform(X_fit)
        X_valid_trans = pre.transform(X_valid)

        et_model = make_et_model(cfg.et_params)
        et_model.fit(X_fit_trans, y_fit_model)
        pred_et = pd.DataFrame(
            et_model.predict(X_valid_trans),
            columns=y_fit_model.columns,
            index=X_valid.index,
        )

        xgb_model = make_xgb_model(cfg.xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        xgb_model.fit(X_fit_trans, y_fit_model)
        pred_xgb = pd.DataFrame(
            xgb_model.predict(X_valid_trans),
            columns=y_fit_model.columns,
            index=X_valid.index,
        )

        oof_et.loc[X_valid.index, :] = pred_et
        oof_xgb.loc[X_valid.index, :] = pred_xgb

        fold_reports.append(
            {
                "fold": fold_idx,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "feature_count_after_pruning": int(X_fit_trans.shape[1]),
            }
        )

    return oof_et, oof_xgb, fold_reports


def build_meta_features(pred_et_model: pd.DataFrame, pred_xgb_model: pd.DataFrame) -> pd.DataFrame:
    left = pred_et_model.copy()
    left.columns = [f"{c}__et" for c in left.columns]
    right = pred_xgb_model.copy()
    right.columns = [f"{c}__xgb" for c in right.columns]
    return pd.concat([left, right], axis=1)


def run_meta_oof_cv(
    meta_features: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    cv_folds: int,
    random_state: int,
    alpha: float,
) -> tuple[pd.DataFrame, list[dict], dict]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    meta_oof = pd.DataFrame(index=y_train_model.index, columns=y_train_model.columns, dtype=float)
    fold_reports: list[dict] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(meta_features), start=1):
        X_fit = meta_features.iloc[fit_idx]
        X_valid = meta_features.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        meta_model = MultiOutputRegressor(Ridge(alpha=alpha, random_state=random_state), n_jobs=1)
        meta_model.fit(X_fit, y_fit_model)

        pred_valid_model = pd.DataFrame(
            meta_model.predict(X_valid),
            columns=y_fit_model.columns,
            index=X_valid.index,
        )
        meta_oof.loc[X_valid.index, :] = pred_valid_model

        pred_valid_full = schema.expand_predictions(pred_valid_model)
        fold_rmse = float(competition_rmse(y_valid_full, pred_valid_full))
        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse": fold_rmse,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    scores = [f["rmse"] for f in fold_reports]
    summary = {
        "mean_rmse": float(np.mean(scores)),
        "std_rmse": float(np.std(scores)),
        "min_rmse": float(np.min(scores)),
        "max_rmse": float(np.max(scores)),
    }
    return meta_oof, fold_reports, summary


def fit_full_models_and_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    cfg: BlendConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pre = fit_feature_preprocessor(
        X_train_raw,
        y_train_model,
        corr_threshold=cfg.et_corr_threshold,
        ratio_eps=cfg.et_ratio_eps,
        signal_quantile=cfg.et_signal_quantile,
        max_selected_features=cfg.et_max_selected_features,
    )
    X_train_trans = pre.transform(X_train_raw)
    X_test_trans = pre.transform(X_test_raw)

    et_model = make_et_model(cfg.et_params)
    et_model.fit(X_train_trans, y_train_model)
    pred_et_test = pd.DataFrame(
        et_model.predict(X_test_trans),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )

    xgb_model = make_xgb_model(cfg.xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
    xgb_model.fit(X_train_trans, y_train_model)
    pred_xgb_test = pd.DataFrame(
        xgb_model.predict(X_test_trans),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )

    meta = {
        "selected_feature_count": int(len(pre.selected_columns)),
        "selected_features": pre.selected_columns,
    }
    return pred_et_test, pred_xgb_test, meta


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

    if args.xgb_params_json:
        xgb_params_json = Path(args.xgb_params_json)
        if not xgb_params_json.is_absolute():
            xgb_params_json = (ROOT / xgb_params_json).resolve()
        xgb_params = load_best_params_from_json(xgb_params_json)
        xgb_optuna_best_rmse = None
    else:
        xgb_params, xgb_optuna_best_rmse = optimize_xgb_params(
            X_train_raw,
            y_train_model,
            y_train_full,
            schema,
            et_params,
            corr_threshold=args.et_corr_threshold,
            ratio_eps=args.et_ratio_eps,
            signal_quantile=args.et_signal_quantile,
            max_selected_features=args.et_max_selected_features,
            random_state=args.random_state,
            xgb_n_jobs=args.xgb_n_jobs,
            n_trials=args.xgb_optuna_trials,
            timeout_sec=args.xgb_optuna_timeout_sec,
            holdout_fraction=args.xgb_optuna_holdout,
        )

    cfg = BlendConfig(
        et_params=et_params,
        xgb_params=xgb_params,
        et_corr_threshold=args.et_corr_threshold,
        et_ratio_eps=args.et_ratio_eps,
        et_signal_quantile=args.et_signal_quantile,
        et_max_selected_features=args.et_max_selected_features,
        xgb_n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
    )

    oof_et, oof_xgb, base_fold_reports = run_oof_base_models(
        X_train_raw,
        y_train_model,
        cfg,
        cv_folds=args.cv_folds,
    )

    meta_train_X = build_meta_features(oof_et, oof_xgb)
    oof_meta, meta_fold_reports, meta_cv_summary = run_meta_oof_cv(
        meta_train_X,
        y_train_model,
        y_train_full,
        schema,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        alpha=args.meta_ridge_alpha,
    )

    oof_meta_full = schema.expand_predictions(oof_meta)
    oof_rmse = float(competition_rmse(y_train_full, oof_meta_full))

    pred_et_test, pred_xgb_test, final_meta = fit_full_models_and_predict(
        X_train_raw,
        X_test_raw,
        y_train_model,
        cfg,
    )

    meta_test_X = build_meta_features(pred_et_test, pred_xgb_test)
    meta_model_full = MultiOutputRegressor(
        Ridge(alpha=args.meta_ridge_alpha, random_state=args.random_state),
        n_jobs=1,
    )
    meta_model_full.fit(meta_train_X, y_train_model)
    pred_stack_test_model = pd.DataFrame(
        meta_model_full.predict(meta_test_X),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )

    pred_stack_test_full = schema.expand_predictions(pred_stack_test_model)
    submission = build_submission_frame(data.x_test["ID"], pred_stack_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    csv_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    oof_et_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_et_model.csv"
    oof_xgb_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_xgb_model.csv"
    oof_stack_file = output_dir / f"{args.submission_prefix}_{timestamp}_oof_stack_model.csv"

    submission.to_csv(csv_file, index=False)
    oof_et.to_csv(oof_et_file, index=True)
    oof_xgb.to_csv(oof_xgb_file, index=True)
    oof_meta.to_csv(oof_stack_file, index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "OOF Stacking (ET fixed + XGB same ET-pipeline + Ridge meta)",
        "data_dir": str(data_dir),
        "et_params_json": str(et_params_json),
        "et_feature_pipeline": {
            "corr_threshold": float(args.et_corr_threshold),
            "signal_quantile": float(args.et_signal_quantile),
            "max_selected_features": int(args.et_max_selected_features),
            "ratio_eps": float(args.et_ratio_eps),
        },
        "xgb": {
            "input_type": "same_as_extratrees_feature_pipeline",
            "n_jobs": int(args.xgb_n_jobs),
            "best_params": xgb_params,
            "optuna_trials": int(args.xgb_optuna_trials),
            "optuna_timeout_sec": int(args.xgb_optuna_timeout_sec),
            "optuna_holdout": float(args.xgb_optuna_holdout),
            "optuna_best_rmse": xgb_optuna_best_rmse,
            "xgb_params_json": args.xgb_params_json,
        },
        "meta_model": {
            "name": "MultiOutputRegressor(Ridge)",
            "alpha": float(args.meta_ridge_alpha),
            "training": "trained_on_oof_predictions_not_weighted_average",
        },
        "cv": {
            "folds": int(args.cv_folds),
            "base_fold_reports": base_fold_reports,
            "meta_fold_reports": meta_fold_reports,
            "meta_cv_summary": meta_cv_summary,
            "oof_rmse": float(oof_rmse),
        },
        "oof_paths": {
            "oof_et_model_path": str(oof_et_file.relative_to(ROOT)),
            "oof_xgb_model_path": str(oof_xgb_file.relative_to(ROOT)),
            "oof_stack_model_path": str(oof_stack_file.relative_to(ROOT)),
        },
        "final_blend": {
            **final_meta,
            "stacking": "trained_meta_model",
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

    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
