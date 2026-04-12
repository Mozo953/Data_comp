from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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


DEFAULT_PARAMS = {
    "n_estimators": 700,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 14,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

ANCHOR_FEATURES = ["Y1", "Y2", "Y3", "Z", "X4", "X5", "X6", "X7", "X12", "X13", "X14", "X15"]
GROUP_DEFINITIONS = {
    "Y": ["Y1", "Y2", "Y3"],
    "X_CORE": ["X4", "X5", "X6", "X7"],
    "X_TAIL": ["X12", "X13", "X14", "X15"],
}
PAIRWISE_LOG_RATIO_FEATURES = [("Y1", "Y2"), ("Y1", "Y3"), ("Z", "Y1"), ("Z", "Y2"), ("X6", "X4")]
GROUP_RATIO_FEATURES = [("Y1", "Y"), ("Y2", "Y"), ("Y3", "Y"), ("Z", "Y"), ("X4", "X_CORE"), ("X6", "X_CORE"), ("X12", "X_TAIL"), ("X14", "X_TAIL")]


@dataclass(frozen=True)
class FeaturePreprocessor:
    selected_columns: list[str]
    generated_columns: list[str]
    dropped_correlated_columns: list[str]
    dropped_unstable_columns: list[str]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        expanded = build_log_ratio_features(features)
        return expanded[self.selected_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost alone with group-mean / log-ratio features, correlation pruning, and CV3."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data", help="Directory with X_train.csv/X_test.csv/y_train.csv.")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/11_experiment_xgb_alone_groupmean_logratio_cv3", help="Directory to save artifacts.")
    parser.add_argument("--submission-prefix", default="xgb_alone_groupmean_logratio", help="Submission filename prefix.")
    parser.add_argument("--corr-threshold", type=float, default=0.985, help="Feature correlation threshold for pruning.")
    parser.add_argument("--signal-quantile", type=float, default=0.30, help="Keep only features above this signal quantile before correlation pruning.")
    parser.add_argument("--max-selected-features", type=int, default=20, help="Maximum number of features kept after signal and correlation pruning.")
    parser.add_argument("--ratio-eps", type=float, default=1e-3, help="Stability epsilon for ratio denominators.")
    parser.add_argument("--min-feature-std", type=float, default=1e-6, help="Drop features with std below this threshold.")
    parser.add_argument("--max-tail-ratio", type=float, default=250.0, help="Drop unstable features with extreme p99/p50 absolute ratio above this value.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation fold count.")
    parser.add_argument("--optuna-trials", type=int, default=10, help="Optuna trials for hyperparameter optimization.")
    parser.add_argument("--optuna-holdout", type=float, default=0.2, help="Validation fraction for Optuna objective.")
    parser.add_argument("--params-json", default=None, help="Optional path to a previous summary JSON to reuse best_params.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for splits and model.")
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna and use default XGBoost parameters.",
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Skip generating CSV submission file (only JSON summary).",
    )
    args = parser.parse_args()

    if not 0.0 < args.corr_threshold < 1.0:
        raise ValueError("--corr-threshold must be between 0 and 1.")
    if not 0.0 < args.signal_quantile <= 1.0:
        raise ValueError("--signal-quantile must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 < args.optuna_holdout < 1.0:
        raise ValueError("--optuna-holdout must be between 0 and 1.")
    if args.max_selected_features < 1:
        raise ValueError("--max-selected-features must be >= 1.")
    if args.min_feature_std <= 0.0:
        raise ValueError("--min-feature-std must be > 0.")
    if args.max_tail_ratio <= 1.0:
        raise ValueError("--max-tail-ratio must be > 1.")
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


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_log_ratio_features(features: pd.DataFrame, *, eps: float = 1e-3) -> pd.DataFrame:
    base = raw_features(features)
    engineered = pd.DataFrame(index=base.index)

    for column in ANCHOR_FEATURES:
        values = base[column].to_numpy(dtype=float)
        engineered[column] = values
        engineered[f"{column}_logabs"] = _signed_log1p(values)

    group_means: dict[str, np.ndarray] = {}
    for group_name, group_columns in GROUP_DEFINITIONS.items():
        group_values = base[group_columns].abs().to_numpy(dtype=float)
        mean_abs = group_values.mean(axis=1)
        group_means[group_name] = mean_abs
        engineered[f"{group_name}_mean_abs"] = mean_abs
        engineered[f"{group_name}_mean_logabs"] = np.log1p(mean_abs)
        engineered[f"{group_name}_spread_abs"] = group_values.max(axis=1) - group_values.min(axis=1)

    for left, right in PAIRWISE_LOG_RATIO_FEATURES:
        denom = _safe_denominator(base[right], eps)
        ratio = base[left].to_numpy(dtype=float) / denom
        engineered[f"log_ratio_{left}_over_{right}"] = _signed_log1p(ratio)

    for left, group_name in GROUP_RATIO_FEATURES:
        denom = group_means[group_name] + eps
        ratio = base[left].to_numpy(dtype=float) / denom
        engineered[f"log_ratio_{left}_over_{group_name}"] = _signed_log1p(ratio)

    y_mean = group_means["Y"] + eps
    x_core_mean = group_means["X_CORE"] + eps
    x_tail_mean = group_means["X_TAIL"] + eps
    engineered["log_ratio_Y_mean_over_X_core_mean"] = _signed_log1p(y_mean / x_core_mean)
    engineered["log_ratio_X_core_mean_over_X_tail_mean"] = _signed_log1p(x_core_mean / x_tail_mean)

    return engineered


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
    min_feature_std: float,
    max_tail_ratio: float,
) -> FeaturePreprocessor:
    expanded = build_log_ratio_features(X_train, eps=ratio_eps)
    stable_columns: list[str] = []
    dropped_unstable: list[str] = []

    for column in expanded.columns:
        values = expanded[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            dropped_unstable.append(column)
            continue

        std = float(np.std(values))
        if std < min_feature_std:
            dropped_unstable.append(column)
            continue

        abs_values = np.abs(values)
        p50 = float(np.quantile(abs_values, 0.50))
        p99 = float(np.quantile(abs_values, 0.99))
        denom = max(p50, ratio_eps)
        tail_ratio = p99 / denom
        if tail_ratio > max_tail_ratio:
            dropped_unstable.append(column)
            continue

        stable_columns.append(column)

    filtered = expanded[stable_columns].copy()
    signal = feature_target_signal(filtered, y_train_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    if signal.empty:
        signal = feature_target_signal(filtered, y_train_model)
    ordered = list(signal.sort_values(ascending=False).index)
    selected, dropped = _prune_correlated_by_order(filtered, ordered, corr_threshold)
    if not selected:
        ordered = list(feature_target_signal(filtered, y_train_model).sort_values(ascending=False).index)
        selected, dropped = _prune_correlated_by_order(filtered, ordered, corr_threshold)
    if len(selected) > max_selected_features:
        selected = selected[:max_selected_features]

    return FeaturePreprocessor(
        selected_columns=selected,
        generated_columns=list(expanded.columns),
        dropped_correlated_columns=dropped,
        dropped_unstable_columns=dropped_unstable,
    )


def make_model(params: dict) -> XGBRegressor:
    return XGBRegressor(**params)


def optimize_params(
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    ratio_eps: float,
    corr_threshold: float,
    signal_quantile: float,
    max_selected_features: int,
    min_feature_std: float,
    max_tail_ratio: float,
    random_state: int,
    n_trials: int,
    holdout_fraction: float,
) -> dict:
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, using default parameters.")
        return DEFAULT_PARAMS.copy()

    X_fit, X_valid, y_fit_model, y_valid_model, y_fit_full, y_valid_full = train_test_split(
        X_train,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    fit_pre = fit_feature_preprocessor(
        X_fit,
        y_fit_model,
        corr_threshold=corr_threshold,
        ratio_eps=ratio_eps,
        signal_quantile=signal_quantile,
        max_selected_features=max_selected_features,
        min_feature_std=min_feature_std,
        max_tail_ratio=max_tail_ratio,
    )
    X_fit_pre = fit_pre.transform(X_fit)
    X_valid_pre = fit_pre.transform(X_valid)

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 300, 800, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 5)
        learning_rate = trial.suggest_float("learning_rate", 0.025, 0.08)
        subsample = trial.suggest_float("subsample", 0.6, 0.85)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.55, 0.8)
        min_child_weight = trial.suggest_float("min_child_weight", 10.0, 30.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.5, 12.0, log=True)
        gamma = trial.suggest_float("gamma", 0.0, 0.12)

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = make_model(params)
        model.fit(X_fit_pre, y_fit_model)

        pred_valid_model = pd.DataFrame(
            model.predict(X_valid_pre),
            columns=y_fit_model.columns,
            index=X_valid_pre.index,
        )
        pred_valid_full = schema.expand_predictions(pred_valid_model)
        score = competition_rmse(y_valid_full, pred_valid_full)
        trial.report(score, step=0)
        if trial.should_prune():
            raise TrialPruned()
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params.copy()
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    best["objective"] = "reg:squarederror"
    best["eval_metric"] = "rmse"
    best["random_state"] = random_state
    best["n_jobs"] = -1
    best["verbose"] = 0

    print(f"Optuna finished. Best RMSE holdout = {study.best_value:.6f}")
    print(f"Best params = {best}")
    return best


def run_cv(
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    params: dict,
    *,
    cv_folds: int,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
    min_feature_std: float,
    max_tail_ratio: float,
    random_state: int,
) -> tuple[list[dict], dict]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_reports: list[dict] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        pre = fit_feature_preprocessor(
            X_fit,
            y_fit_model,
            corr_threshold=corr_threshold,
            ratio_eps=ratio_eps,
            signal_quantile=signal_quantile,
            max_selected_features=max_selected_features,
            min_feature_std=min_feature_std,
            max_tail_ratio=max_tail_ratio,
        )
        X_fit_pre = pre.transform(X_fit)
        X_valid_pre = pre.transform(X_valid)

        model = make_model(params)
        model.fit(X_fit_pre, y_fit_model)

        pred_valid_model = pd.DataFrame(
            model.predict(X_valid_pre),
            columns=y_fit_model.columns,
            index=X_valid_pre.index,
        )
        pred_valid_full = schema.expand_predictions(pred_valid_model)
        rmse = competition_rmse(y_valid_full, pred_valid_full)

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse": float(rmse),
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "feature_count_after_pruning": int(X_fit_pre.shape[1]),
                "signal_quantile": float(signal_quantile),
                "generated_feature_count": int(len(pre.generated_columns)),
                "dropped_correlated_feature_count": int(len(pre.dropped_correlated_columns)),
                "dropped_unstable_feature_count": int(len(pre.dropped_unstable_columns)),
            }
        )

    scores = [fold["rmse"] for fold in fold_reports]
    cv_summary = {
        "mean_rmse": float(np.mean(scores)),
        "std_rmse": float(np.std(scores)),
        "min_rmse": float(np.min(scores)),
        "max_rmse": float(np.max(scores)),
    }
    return fold_reports, cv_summary


def fit_full_and_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_model: pd.DataFrame,
    params: dict,
    *,
    corr_threshold: float,
    ratio_eps: float,
    signal_quantile: float,
    max_selected_features: int,
    min_feature_std: float,
    max_tail_ratio: float,
) -> tuple[pd.DataFrame, FeaturePreprocessor, XGBRegressor]:
    pre = fit_feature_preprocessor(
        X_train,
        y_train_model,
        corr_threshold=corr_threshold,
        ratio_eps=ratio_eps,
        signal_quantile=signal_quantile,
        max_selected_features=max_selected_features,
        min_feature_std=min_feature_std,
        max_tail_ratio=max_tail_ratio,
    )
    X_train_pre = pre.transform(X_train)
    X_test_pre = pre.transform(X_test)

    model = make_model(params)
    model.fit(X_train_pre, y_train_model)

    pred_test_model = pd.DataFrame(
        model.predict(X_test_pre),
        columns=y_train_model.columns,
        index=X_test_pre.index,
    )
    return pred_test_model, pre, model


def summarize_importance(model: XGBRegressor, columns: list[str]) -> dict[str, float]:
    importance_dict = model.get_booster().get_score(importance_type="weight")
    importance_series = pd.Series(importance_dict).sort_values(ascending=False)
    return {name: float(score) for name, score in importance_series.items() if name in columns}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train = raw_features(data.x_train)
    X_test = raw_features(data.x_test)
    y_train_full = data.y_train.drop(columns=["ID"]) if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    if args.skip_optuna:
        best_params = DEFAULT_PARAMS.copy()
        if args.params_json is not None:
            params_json_path = Path(args.params_json)
            if not params_json_path.is_absolute():
                params_json_path = (ROOT / params_json_path).resolve()
            best_params.update(load_best_params_from_json(params_json_path))
    elif args.params_json is not None:
        params_json_path = Path(args.params_json)
        if not params_json_path.is_absolute():
            params_json_path = (ROOT / params_json_path).resolve()
        best_params = DEFAULT_PARAMS.copy()
        best_params.update(load_best_params_from_json(params_json_path))
    else:
        best_params = optimize_params(
            X_train,
            y_train_model,
            y_train_full,
            schema,
            ratio_eps=args.ratio_eps,
            signal_quantile=args.signal_quantile,
            max_selected_features=args.max_selected_features,
            min_feature_std=args.min_feature_std,
            max_tail_ratio=args.max_tail_ratio,
            corr_threshold=args.corr_threshold,
            random_state=args.random_state,
            n_trials=args.optuna_trials,
            holdout_fraction=args.optuna_holdout,
        )

    best_params["random_state"] = args.random_state
    best_params["n_jobs"] = -1
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["verbosity"] = 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_reports, cv_summary = run_cv(
        X_train,
        y_train_model,
        y_train_full,
        schema,
        best_params,
        cv_folds=args.cv_folds,
        corr_threshold=args.corr_threshold,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.signal_quantile,
        max_selected_features=args.max_selected_features,
        min_feature_std=args.min_feature_std,
        max_tail_ratio=args.max_tail_ratio,
        random_state=args.random_state,
    )

    pred_test_model, full_pre, full_model = fit_full_and_predict(
        X_train,
        X_test,
        y_train_model,
        best_params,
        corr_threshold=args.corr_threshold,
        ratio_eps=args.ratio_eps,
        signal_quantile=args.signal_quantile,
        max_selected_features=args.max_selected_features,
        min_feature_std=args.min_feature_std,
        max_tail_ratio=args.max_tail_ratio,
    )

    pred_test_full = schema.expand_predictions(pred_test_model)
    
    submission_file = None
    if not args.skip_submission:
        submission = build_submission_frame(data.x_test["ID"], pred_test_full)
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)

    summary = {
        "generated_at_utc": timestamp,
        "model": "XGBRegressor",
        "experiment": "xgb_alone_ratio_log",
        "data_dir": str(data_dir),
        "feature_family": {
            "anchors": ANCHOR_FEATURES,
            "log_ratio_pairs": LOG_RATIO_PAIRS,
            "diff_pairs": DIFF_PAIRS,
        },
        "feature_pipeline": {
            "base_feature_count": int(X_train.shape[1]),
            "generated_feature_count": int(len(full_pre.generated_columns)),
            "selected_feature_count_after_pruning": int(len(full_pre.selected_columns)),
            "dropped_correlated_feature_count": int(len(full_pre.dropped_correlated_columns)),
            "dropped_unstable_feature_count": int(len(full_pre.dropped_unstable_columns)),
            "corr_threshold": float(args.corr_threshold),
            "signal_quantile": float(args.signal_quantile),
            "max_selected_features": int(args.max_selected_features),
            "ratio_eps": float(args.ratio_eps),
            "min_feature_std": float(args.min_feature_std),
            "max_tail_ratio": float(args.max_tail_ratio),
        },
        "target_handling": {
            "d15_strategy": "constant_target_removed_from_training_via_schema",
            "modeled_targets": schema.model_targets,
            "duplicate_groups": [group for group in schema.duplicate_groups if len(group) > 1],
            "constant_targets": schema.constant_targets,
        },
        "optuna": {
            "enabled": bool(not args.skip_optuna),
            "available": bool(OPTUNA_AVAILABLE),
            "trials": int(args.optuna_trials if not args.skip_optuna else 0),
            "best_params": best_params,
        },
        "cv": {
            "folds": int(args.cv_folds),
            "fold_reports": fold_reports,
            "summary": cv_summary,
        },
        "submission_path": str(submission_file.relative_to(ROOT)) if submission_file else "skipped",
        "rows_predicted": int(len(submission)) if not args.skip_submission else 0,
        "feature_importance": summarize_importance(full_model, full_pre.selected_columns),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
