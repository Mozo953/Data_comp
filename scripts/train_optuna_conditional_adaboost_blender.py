from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, load_modeling_data  # noqa: E402


RAW_COLUMNS = ["M12", "M13", "M14", "M15", "M4", "M5", "M6", "M7", "R", "S1", "S2", "S3"]
BLOCK_A = ["M12", "M13", "M14", "M15"]
BLOCK_B = ["M4", "M5", "M6", "M7"]
SUPPORT = ["R", "S1", "S2", "S3"]
BASE_MODELS = ["et_rowagg_mf06_bs", "et_allpool_3"]
LOCAL_EXPERT = "rf_local_045_080"
MODEL_ORDER = ["et_rowagg_mf06_bs", "et_allpool_3", LOCAL_EXPERT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna-tuned ET rowagg/allpool blender with a conditional RandomForest expert active only "
            "inside a Humidity interval. RandomForest uses 12 raw + 30 engineered candidates with "
            "fold-local top-35 feature selection."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--reference-dir",
        default="artifacts_extratrees_corr_optuna/Blender_ET3_allpool_+_rowaggbs_0.1391_bins(1ou1.2)",
    )
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/53_optuna_conditional_rf_blender")
    parser.add_argument("--prefix", default="optuna_conditional_rf_blender")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--ada-low", type=float, default=0.45)
    parser.add_argument("--ada-high", type=float, default=0.80)
    parser.add_argument("--selected-features", type=int, default=35)
    parser.add_argument("--optuna-trials", type=int, default=3)
    parser.add_argument("--optuna-timeout-sec", type=int, default=180)
    parser.add_argument("--dirichlet-samples", type=int, default=5000)
    parser.add_argument("--dirichlet-batch-size", type=int, default=1024)
    parser.add_argument("--dirichlet-alpha", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--no-sample-weights", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 <= args.ada_low < args.ada_high <= 1.0:
        raise ValueError("RandomForest Humidity interval must satisfy 0 <= low < high <= 1.")
    if args.selected_features > 42:
        raise ValueError("--selected-features cannot exceed the 42 RandomForest candidate features.")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def verbose_log(enabled: bool, message: str) -> None:
    if enabled:
        log(message)


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_stem(raw_value: str) -> str:
    stem = Path(str(raw_value)).name.strip()
    for char in '<>:"/\\|?*':
        stem = stem.replace(char, "_")
    return stem or "optuna_conditional_rf_blender"


def load_clean_module():
    module_path = ROOT / "scripts" / "train_best_model42_clean.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing helper script: {module_path}")
    spec = importlib.util.spec_from_file_location("train_best_model42_clean", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def maybe_subsample_bundle(bundle, *, max_train_rows: int | None, max_test_rows: int | None):
    if max_train_rows is None and max_test_rows is None:
        return bundle
    return type(bundle)(
        data=type(bundle.data)(
            x_train=bundle.data.x_train if max_train_rows is None else bundle.data.x_train.iloc[:max_train_rows].copy(),
            x_test=bundle.data.x_test if max_test_rows is None else bundle.data.x_test.iloc[:max_test_rows].copy(),
            y_train=bundle.data.y_train if max_train_rows is None else bundle.data.y_train.iloc[:max_train_rows].copy(),
        ),
        schema=bundle.schema,
        x_train_raw=bundle.x_train_raw if max_train_rows is None else bundle.x_train_raw.iloc[:max_train_rows].copy(),
        x_test_raw=bundle.x_test_raw if max_test_rows is None else bundle.x_test_raw.iloc[:max_test_rows].copy(),
        y_train_full=bundle.y_train_full if max_train_rows is None else bundle.y_train_full.iloc[:max_train_rows].copy(),
        y_train_model=bundle.y_train_model if max_train_rows is None else bundle.y_train_model.iloc[:max_train_rows].copy(),
    )


def compute_row_weights(humidity: pd.Series, *, disabled: bool) -> pd.Series:
    if disabled:
        values = np.ones(len(humidity), dtype=np.float32)
    else:
        values = np.where(humidity.to_numpy(dtype=np.float32) >= 0.6, 1.2, 1.0).astype(np.float32)
    return pd.Series(values, index=humidity.index, name="row_weight")


def humidity_interval_mask(humidity: pd.Series, *, low: float, high: float) -> pd.Series:
    return (humidity >= low) & (humidity <= high)


def get_target_multiplicities(schema, modeled_targets: list[str]) -> np.ndarray:
    return np.asarray(
        [
            sum(1 for original in schema.original_targets if schema.representative_for_target[original] == target)
            for target in modeled_targets
        ],
        dtype=np.float32,
    )


def weighted_wrmse(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    *,
    row_weights: pd.Series,
    target_multiplicities: np.ndarray,
    full_target_count: int,
) -> float:
    true_values = y_true.to_numpy(dtype=np.float32)
    pred_values = np.clip(y_pred.to_numpy(dtype=np.float32), 0.0, 1.0)
    row_weight_values = row_weights.to_numpy(dtype=np.float32).reshape(-1, 1)
    target_weight_values = target_multiplicities.reshape(1, -1)
    weighted_sq_error = row_weight_values * target_weight_values * np.square(pred_values - true_values)
    return float(np.sqrt(weighted_sq_error.sum(dtype=np.float64) / float(len(y_true) * full_target_count)))


def subset_wrmse(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    *,
    row_weights: pd.Series,
    mask: pd.Series,
    target_multiplicities: np.ndarray,
    full_target_count: int,
) -> float | None:
    if int(mask.sum()) == 0:
        return None
    idx = mask[mask].index
    return weighted_wrmse(
        y_true.loc[idx],
        y_pred.loc[idx],
        row_weights=row_weights.loc[idx],
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )


def make_rowagg_model(params: dict[str, float | int], *, n_jobs: int, random_state: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        bootstrap=True,
        max_samples=float(params["max_samples"]),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def make_allpool_model(params: dict[str, float | int], *, n_jobs: int, random_state: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        bootstrap=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def make_rf_model(params: dict[str, float | int | str], *, random_state: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=float(params["max_features"]),
        bootstrap=True,
        max_samples=float(params["max_samples"]),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def suggest_rowagg_params(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 260, 520, step=40),
        "max_depth": trial.suggest_int("max_depth", 22, 34),
        "min_samples_split": trial.suggest_int("min_samples_split", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_float("max_features", 0.45, 0.85),
        "max_samples": trial.suggest_float("max_samples", 0.70, 0.95),
    }


def suggest_allpool_params(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 220, 420, step=40),
        "max_depth": trial.suggest_int("max_depth", 14, 24),
        "min_samples_split": trial.suggest_int("min_samples_split", 6, 18),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 8),
        "max_features": trial.suggest_float("max_features", 0.18, 0.45),
    }


def suggest_rf_params(trial: optuna.Trial) -> dict[str, float | int | str]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 120, 420, step=40),
        "max_depth": trial.suggest_int("max_depth", 5, 18),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 80),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "max_features": trial.suggest_float("max_features", 0.35, 1.0),
        "max_samples": trial.suggest_float("max_samples", 0.55, 0.95),
    }


def default_rowagg_params() -> dict[str, float | int]:
    return {
        "n_estimators": 500,
        "max_depth": 28,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.6,
        "max_samples": 0.8051184329566651,
    }


def default_allpool_params() -> dict[str, float | int]:
    return {
        "n_estimators": 300,
        "max_depth": 17,
        "min_samples_split": 13,
        "min_samples_leaf": 5,
        "max_features": 0.25,
    }


def default_rf_params() -> dict[str, float | int | str]:
    return {
        "n_estimators": 240,
        "max_depth": 10,
        "min_samples_leaf": 12,
        "min_samples_split": 6,
        "max_features": 0.7,
        "max_samples": 0.8,
    }


def build_local_candidate_features(raw: pd.DataFrame, *, ratio_eps: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    stats = pd.DataFrame(index=raw.index)
    values = raw[RAW_COLUMNS].to_numpy(dtype=np.float32)
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1)
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90], axis=1)
    stats["row_mean"] = row_mean
    stats["row_std"] = row_std
    stats["row_p10"] = p10
    stats["row_p25"] = p25
    stats["row_p50"] = p50
    stats["row_p75"] = p75
    stats["row_p90"] = p90
    stats["row_iqr"] = p75 - p25
    stats["row_mad"] = np.mean(np.abs(values - row_mean.reshape(-1, 1)), axis=1)
    stats["row_l1"] = np.abs(values).sum(axis=1)
    stats["row_l2"] = np.sqrt(np.square(values).sum(axis=1))

    block_a_mean = raw[BLOCK_A].mean(axis=1)
    block_b_mean = raw[BLOCK_B].mean(axis=1)
    support_mean = raw[SUPPORT].mean(axis=1)
    engineered = pd.DataFrame(index=raw.index)
    engineered["block_a_mean"] = block_a_mean
    engineered["block_b_mean"] = block_b_mean
    engineered["support_mean"] = support_mean
    engineered["block_gap"] = block_a_mean - block_b_mean
    for column in ["M12", "M15", "M4", "M7"]:
        engineered[f"log_{column}"] = np.log1p(np.clip(raw[column].to_numpy(dtype=np.float32), 0.0, None))
    engineered["log_row_l1"] = np.log1p(np.clip(stats["row_l1"].to_numpy(dtype=np.float32), 0.0, None))
    engineered["log_row_l2"] = np.log1p(np.clip(stats["row_l2"].to_numpy(dtype=np.float32), 0.0, None))
    for column in ["M12", "M15", "M4", "R"]:
        engineered[f"{column}_over_row_mean"] = raw[column].to_numpy(dtype=np.float32) / (row_mean + ratio_eps)
    engineered["M12_minus_block_a_mean"] = raw["M12"].to_numpy(dtype=np.float32) - block_a_mean.to_numpy(dtype=np.float32)
    engineered["M4_minus_block_b_mean"] = raw["M4"].to_numpy(dtype=np.float32) - block_b_mean.to_numpy(dtype=np.float32)
    engineered["block_a_std"] = raw[BLOCK_A].std(axis=1).to_numpy(dtype=np.float32)
    engineered["block_b_std"] = raw[BLOCK_B].std(axis=1).to_numpy(dtype=np.float32)
    engineered["support_std"] = raw[SUPPORT].std(axis=1).to_numpy(dtype=np.float32)

    engineered = pd.concat([stats, engineered], axis=1).astype(np.float32)
    if engineered.shape[1] != 30:
        raise ValueError(f"Expected exactly 30 engineered RandomForest features, got {engineered.shape[1]}.")
    candidates = pd.concat([raw[RAW_COLUMNS].astype(np.float32), engineered], axis=1).astype(np.float32)
    if candidates.shape[1] != 42:
        raise ValueError(f"Expected exactly 42 RandomForest candidate features, got {candidates.shape[1]}.")
    return candidates, list(raw[RAW_COLUMNS].columns), list(engineered.columns)


def select_top_features(x_fit: pd.DataFrame, y_fit: pd.DataFrame, *, k: int) -> list[str]:
    if k >= x_fit.shape[1]:
        return list(x_fit.columns)
    x = x_fit.to_numpy(dtype=np.float64)
    y = y_fit.to_numpy(dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0)
    y_std = y.std(axis=0)
    x_std[x_std == 0.0] = np.nan
    y_std[y_std == 0.0] = np.nan
    corr = (x.T @ y) / max(1, (len(x_fit) - 1))
    corr = corr / x_std.reshape(-1, 1)
    corr = corr / y_std.reshape(1, -1)
    scores = np.nanmean(np.abs(corr), axis=1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)[::-1][:k]
    return [x_fit.columns[int(idx)] for idx in order]


def score_model_oof(
    predictions: pd.DataFrame,
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    target_multiplicities: np.ndarray,
    full_target_count: int,
) -> float:
    return weighted_wrmse(
        y_train,
        predictions,
        row_weights=row_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )


def make_oof_for_single_model(
    clean,
    model_name: str,
    params: dict[str, float | int | str],
    x_train_nohum: pd.DataFrame,
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    allowed_mask: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, int]], Counter[str]]:
    cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.random_state))
    oof = pd.DataFrame(index=y_train.index, columns=y_train.columns, dtype=np.float32)
    fold_reports: list[dict[str, int]] = []
    selected_counter: Counter[str] = Counter()

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train_nohum), start=1):
        fit_index = x_train_nohum.index[fit_idx]
        valid_index = x_train_nohum.index[valid_idx]

        if model_name == LOCAL_EXPERT:
            local_fit_index = fit_index.intersection(allowed_mask[allowed_mask].index)
            if len(local_fit_index) < 2:
                raise ValueError("Not enough local RandomForest fit rows inside Humidity interval.")
            views = clean.build_feature_views(
                x_train_nohum.loc[local_fit_index],
                x_train_nohum.loc[valid_index],
                tail_quantile=float(args.tail_quantile),
                ratio_eps=float(args.ratio_eps),
            )
            x_fit_42, _, _ = build_local_candidate_features(views.raw_fit, ratio_eps=float(args.ratio_eps))
            x_valid_42, _, _ = build_local_candidate_features(views.raw_pred, ratio_eps=float(args.ratio_eps))
            selected = select_top_features(x_fit_42, y_train.loc[local_fit_index], k=int(args.selected_features))
            selected_counter.update(selected)
            model = make_rf_model(params, random_state=int(args.random_state) + fold, n_jobs=int(args.n_jobs))
            model.fit(
                x_fit_42[selected],
                y_train.loc[local_fit_index],
                sample_weight=row_weights.loc[local_fit_index].to_numpy(dtype=np.float32),
            )
            pred = np.clip(model.predict(x_valid_42[selected]), 0.0, 1.0)
        else:
            views = clean.build_feature_views(
                x_train_nohum.loc[fit_index],
                x_train_nohum.loc[valid_index],
                tail_quantile=float(args.tail_quantile),
                ratio_eps=float(args.ratio_eps),
            )
            if model_name == "et_rowagg_mf06_bs":
                model = make_rowagg_model(params, n_jobs=int(args.n_jobs), random_state=int(args.random_state) + fold)
                x_fit = views.rowagg_fit
                x_valid = views.rowagg_pred
            elif model_name == "et_allpool_3":
                model = make_allpool_model(params, n_jobs=int(args.n_jobs), random_state=83 + fold)
                x_fit = views.allpool_fit
                x_valid = views.allpool_pred
            else:
                raise ValueError(model_name)
            model.fit(x_fit, y_train.loc[fit_index], sample_weight=row_weights.loc[fit_index].to_numpy(dtype=np.float32))
            pred = np.clip(model.predict(x_valid), 0.0, 1.0)

        oof.loc[valid_index] = pred.astype(np.float32)
        fold_reports.append({"fold": fold, "fit_rows": int(len(fit_index)), "valid_rows": int(len(valid_index))})
        verbose_log(bool(args.verbose), f"{model_name}: OOF fold {fold}/{args.cv_folds} ready")

    if oof.isna().any().any():
        raise RuntimeError(f"Missing OOF predictions for {model_name}.")
    return oof.astype(np.float32), fold_reports, selected_counter


def tune_model(
    clean,
    model_name: str,
    default_params: dict[str, float | int | str],
    x_train_nohum: pd.DataFrame,
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    allowed_mask: pd.Series,
    target_multiplicities: np.ndarray,
    full_target_count: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str], float, list[dict[str, object]]]:
    if int(args.optuna_trials) <= 0:
        return default_params.copy(), float("nan"), []

    def objective(trial: optuna.Trial) -> float:
        if model_name == "et_rowagg_mf06_bs":
            params = suggest_rowagg_params(trial)
        elif model_name == "et_allpool_3":
            params = suggest_allpool_params(trial)
        elif model_name == LOCAL_EXPERT:
            params = suggest_rf_params(trial)
        else:
            raise ValueError(model_name)
        oof, _, _ = make_oof_for_single_model(
            clean,
            model_name,
            params,
            x_train_nohum,
            y_train,
            row_weights,
            allowed_mask,
            args,
        )
        score = score_model_oof(oof, y_train, row_weights, target_multiplicities, full_target_count)
        verbose_log(bool(args.verbose), f"Optuna {model_name}: trial={trial.number}, WRMSE={score:.6f}, params={params}")
        return score

    sampler = optuna.samplers.TPESampler(seed=int(args.random_state))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(default_params)
    study.optimize(
        objective,
        n_trials=int(args.optuna_trials),
        timeout=int(args.optuna_timeout_sec),
        show_progress_bar=False,
    )
    trials = [
        {"number": int(trial.number), "value": None if trial.value is None else float(trial.value), "params": trial.params}
        for trial in study.trials
    ]
    return dict(study.best_params), float(study.best_value), trials


def fit_full_and_predict_test(
    clean,
    model_name: str,
    params: dict[str, float | int | str],
    x_train_nohum: pd.DataFrame,
    x_test_nohum: pd.DataFrame,
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    train_allowed_mask: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Counter[str], dict[str, int]]:
    selected_counter: Counter[str] = Counter()
    if model_name == LOCAL_EXPERT:
        local_index = train_allowed_mask[train_allowed_mask].index
        views = clean.build_feature_views(
            x_train_nohum.loc[local_index],
            x_test_nohum,
            tail_quantile=float(args.tail_quantile),
            ratio_eps=float(args.ratio_eps),
        )
        x_fit_42, _, _ = build_local_candidate_features(views.raw_fit, ratio_eps=float(args.ratio_eps))
        x_test_42, _, _ = build_local_candidate_features(views.raw_pred, ratio_eps=float(args.ratio_eps))
        selected = select_top_features(x_fit_42, y_train.loc[local_index], k=int(args.selected_features))
        selected_counter.update(selected)
        model = make_rf_model(params, random_state=int(args.random_state), n_jobs=int(args.n_jobs))
        model.fit(
            x_fit_42[selected],
            y_train.loc[local_index],
            sample_weight=row_weights.loc[local_index].to_numpy(dtype=np.float32),
        )
        pred = np.clip(model.predict(x_test_42[selected]), 0.0, 1.0)
        info = {"fit_rows": int(len(local_index)), "predict_rows": int(len(x_test_nohum))}
    else:
        views = clean.build_feature_views(
            x_train_nohum,
            x_test_nohum,
            tail_quantile=float(args.tail_quantile),
            ratio_eps=float(args.ratio_eps),
        )
        if model_name == "et_rowagg_mf06_bs":
            model = make_rowagg_model(params, n_jobs=int(args.n_jobs), random_state=int(args.random_state))
            x_fit = views.rowagg_fit
            x_test = views.rowagg_pred
        elif model_name == "et_allpool_3":
            model = make_allpool_model(params, n_jobs=int(args.n_jobs), random_state=83)
            x_fit = views.allpool_fit
            x_test = views.allpool_pred
        else:
            raise ValueError(model_name)
        model.fit(x_fit, y_train, sample_weight=row_weights.to_numpy(dtype=np.float32))
        pred = np.clip(model.predict(x_test), 0.0, 1.0)
        info = {"fit_rows": int(len(x_train_nohum)), "predict_rows": int(len(x_test_nohum))}
    return pd.DataFrame(pred, index=x_test_nohum.index, columns=y_train.columns, dtype=np.float32), selected_counter, info


def build_dirichlet_candidates(alpha: np.ndarray, sample_count: int, random_state: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    alpha = alpha[:width].astype(np.float64)
    candidates = rng.dirichlet(alpha, size=sample_count).astype(np.float32)
    extras = [
        (alpha / alpha.sum()).reshape(1, -1).astype(np.float32),
        np.full((1, width), 1.0 / width, dtype=np.float32),
        np.eye(width, dtype=np.float32),
    ]
    return np.vstack(extras + [candidates]).astype(np.float32)


def optimize_conditional_blend(
    model_oofs: dict[str, pd.DataFrame],
    y_train: pd.DataFrame,
    *,
    row_weights: pd.Series,
    allowed_mask: pd.Series,
    alpha: np.ndarray,
    sample_count: int,
    batch_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inside_candidates = build_dirichlet_candidates(alpha, sample_count, random_state, 3)
    outside_candidates_2 = build_dirichlet_candidates(alpha[:2], sample_count, random_state + 17, 2)
    outside_candidates = np.column_stack(
        [outside_candidates_2[:, 0], outside_candidates_2[:, 1], np.zeros(len(outside_candidates_2), dtype=np.float32)]
    ).astype(np.float32)

    weight_rows: list[dict[str, float | str]] = []
    blended = pd.DataFrame(index=y_train.index, columns=y_train.columns, dtype=np.float32)
    row_weight_values = row_weights.to_numpy(dtype=np.float32)
    inside_bool = allowed_mask.to_numpy(dtype=bool)

    for target in y_train.columns:
        stacked = np.column_stack([model_oofs[name][target].to_numpy(dtype=np.float32) for name in MODEL_ORDER])
        y_true = y_train[target].to_numpy(dtype=np.float32)
        final_pred = np.zeros(len(y_train), dtype=np.float32)

        for zone_name, zone_mask, candidates in [
            ("inside_045_080", inside_bool, inside_candidates),
            ("outside_045_080", ~inside_bool, outside_candidates),
        ]:
            if not np.any(zone_mask):
                continue
            best_mse = float("inf")
            best_weights: np.ndarray | None = None
            best_pred: np.ndarray | None = None
            zone_stacked = stacked[zone_mask]
            zone_y = y_true[zone_mask]
            zone_w = row_weight_values[zone_mask]
            for start in range(0, len(candidates), batch_size):
                chunk = candidates[start : start + batch_size]
                pred_chunk = np.clip(zone_stacked @ chunk.T, 0.0, 1.0)
                mse_chunk = np.mean(zone_w[:, None] * np.square(pred_chunk - zone_y[:, None]), axis=0)
                local_idx = int(np.argmin(mse_chunk))
                if float(mse_chunk[local_idx]) < best_mse:
                    best_mse = float(mse_chunk[local_idx])
                    best_weights = chunk[local_idx].copy()
                    best_pred = pred_chunk[:, local_idx].copy()
            if best_weights is None or best_pred is None:
                raise RuntimeError(f"Blend failed for target={target}, zone={zone_name}.")
            final_pred[zone_mask] = best_pred.astype(np.float32)
            weight_rows.append(
                {
                    "target": target,
                    "zone": zone_name,
                    "et_rowagg_mf06_bs": float(best_weights[0]),
                    "et_allpool_3": float(best_weights[1]),
                    LOCAL_EXPERT: float(best_weights[2]),
                    "weighted_mse": float(best_mse),
                }
            )
        blended[target] = np.clip(final_pred, 0.0, 1.0)
    return pd.DataFrame(weight_rows), blended.astype(np.float32)


def apply_conditional_blend(
    model_predictions: dict[str, pd.DataFrame],
    blend_weights: pd.DataFrame,
    *,
    allowed_mask: pd.Series,
) -> pd.DataFrame:
    blended = pd.DataFrame(index=next(iter(model_predictions.values())).index, columns=blend_weights["target"].unique(), dtype=np.float32)
    inside_bool = allowed_mask.to_numpy(dtype=bool)
    for target in blended.columns:
        final_values = np.zeros(len(blended), dtype=np.float32)
        for zone_name, zone_mask in [("inside_045_080", inside_bool), ("outside_045_080", ~inside_bool)]:
            row = blend_weights[(blend_weights["target"] == target) & (blend_weights["zone"] == zone_name)]
            if row.empty or not np.any(zone_mask):
                continue
            weights = row.iloc[0]
            zone_values = np.zeros(int(zone_mask.sum()), dtype=np.float32)
            for model_name in MODEL_ORDER:
                zone_values += float(weights[model_name]) * model_predictions[model_name].loc[zone_mask, target].to_numpy(dtype=np.float32)
            final_values[zone_mask] = np.clip(zone_values, 0.0, 1.0)
        blended[target] = final_values
    return blended.astype(np.float32)


def latest_reference_summary(reference_dir: Path) -> dict[str, object] | None:
    if not reference_dir.exists():
        return None
    candidates = sorted(reference_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = parse_args()
    clean = load_clean_module()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_stem(args.prefix)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bundle = maybe_subsample_bundle(
        load_modeling_data(resolve_path(args.data_dir)),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    humidity_train = bundle.data.x_train["Humidity"].copy()
    humidity_test = bundle.data.x_test["Humidity"].copy()
    train_allowed = humidity_interval_mask(humidity_train, low=float(args.ada_low), high=float(args.ada_high))
    test_allowed = humidity_interval_mask(humidity_test, low=float(args.ada_low), high=float(args.ada_high))
    row_weights = compute_row_weights(humidity_train, disabled=bool(args.no_sample_weights))
    x_train_nohum = clean.drop_environment_columns(bundle.data.x_train)
    x_test_nohum = clean.drop_environment_columns(bundle.data.x_test)
    clean.validate_no_environment_columns(x_train_nohum, "x_train_nohum")
    clean.validate_no_environment_columns(x_test_nohum, "x_test_nohum")

    target_multiplicities = get_target_multiplicities(bundle.schema, list(bundle.y_train_model.columns))
    full_target_count = len(bundle.schema.original_targets)

    log(
        "Conditional RandomForest blender: ET rowagg/allpool + RandomForest active only in "
        f"Humidity [{args.ada_low}, {args.ada_high}], CV={args.cv_folds}, Optuna trials/model={args.optuna_trials}"
    )
    log(f"Rows: train={len(x_train_nohum)}, test={len(x_test_nohum)}, train_inside={int(train_allowed.sum())}, test_inside={int(test_allowed.sum())}")

    default_params = {
        "et_rowagg_mf06_bs": default_rowagg_params(),
        "et_allpool_3": default_allpool_params(),
        LOCAL_EXPERT: default_rf_params(),
    }
    best_params: dict[str, dict[str, float | int | str]] = {}
    optuna_reports: dict[str, object] = {}

    for model_name in MODEL_ORDER:
        log(f"{model_name}: starting Optuna")
        params, best_score, trials = tune_model(
            clean,
            model_name,
            default_params[model_name],
            x_train_nohum,
            bundle.y_train_model,
            row_weights,
            train_allowed,
            target_multiplicities,
            full_target_count,
            args,
        )
        best_params[model_name] = params
        optuna_reports[model_name] = {"best_score": best_score, "trials": trials}
        log(f"{model_name}: best Optuna WRMSE={best_score:.6f}, best_params={params}")

    model_oofs: dict[str, pd.DataFrame] = {}
    model_tests: dict[str, pd.DataFrame] = {}
    fold_reports: dict[str, list[dict[str, int]]] = {}
    feature_counter: Counter[str] = Counter()
    full_fit_reports: dict[str, dict[str, int]] = {}

    for model_name in MODEL_ORDER:
        log(f"{model_name}: rebuilding final OOF with best params")
        oof, reports, selected_counter = make_oof_for_single_model(
            clean,
            model_name,
            best_params[model_name],
            x_train_nohum,
            bundle.y_train_model,
            row_weights,
            train_allowed,
            args,
        )
        model_oofs[model_name] = oof
        fold_reports[model_name] = reports
        feature_counter.update(selected_counter)
        test_pred, selected_full, fit_info = fit_full_and_predict_test(
            clean,
            model_name,
            best_params[model_name],
            x_train_nohum,
            x_test_nohum,
            bundle.y_train_model,
            row_weights,
            train_allowed,
            args,
        )
        model_tests[model_name] = test_pred
        feature_counter.update(selected_full)
        full_fit_reports[model_name] = fit_info

    base_scores = {}
    for model_name in MODEL_ORDER:
        base_scores[model_name] = {
            "global_wrmse": score_model_oof(
                model_oofs[model_name],
                bundle.y_train_model,
                row_weights,
                target_multiplicities,
                full_target_count,
            ),
            "inside_wrmse": subset_wrmse(
                bundle.y_train_model,
                model_oofs[model_name],
                row_weights=row_weights,
                mask=train_allowed,
                target_multiplicities=target_multiplicities,
                full_target_count=full_target_count,
            ),
            "outside_wrmse": subset_wrmse(
                bundle.y_train_model,
                model_oofs[model_name],
                row_weights=row_weights,
                mask=~train_allowed,
                target_multiplicities=target_multiplicities,
                full_target_count=full_target_count,
            ),
        }
        log(
            f"{model_name}: OOF global={base_scores[model_name]['global_wrmse']:.6f}, "
            f"inside={base_scores[model_name]['inside_wrmse']:.6f}"
        )

    blend_weights, blended_oof_model = optimize_conditional_blend(
        model_oofs,
        bundle.y_train_model,
        row_weights=row_weights,
        allowed_mask=train_allowed,
        alpha=np.asarray(args.dirichlet_alpha, dtype=np.float32),
        sample_count=int(args.dirichlet_samples),
        batch_size=int(args.dirichlet_batch_size),
        random_state=int(args.random_state),
    )
    blended_test_model = apply_conditional_blend(model_tests, blend_weights, allowed_mask=test_allowed)
    blended_oof_full = bundle.schema.expand_predictions(blended_oof_model)
    blended_test_full = bundle.schema.expand_predictions(blended_test_model)
    submission = build_submission_frame(bundle.data.x_test["ID"], blended_test_full)

    blend_scores = {
        "global_wrmse": weighted_wrmse(
            bundle.y_train_model,
            blended_oof_model,
            row_weights=row_weights,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        ),
        "inside_wrmse": subset_wrmse(
            bundle.y_train_model,
            blended_oof_model,
            row_weights=row_weights,
            mask=train_allowed,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        ),
        "outside_wrmse": subset_wrmse(
            bundle.y_train_model,
            blended_oof_model,
            row_weights=row_weights,
            mask=~train_allowed,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        ),
    }
    log(f"Conditional blend OOF global={blend_scores['global_wrmse']:.6f}, inside={blend_scores['inside_wrmse']:.6f}")

    feature_frequency = pd.DataFrame(
        [{"feature": feature, "selection_count": int(count)} for feature, count in feature_counter.most_common()]
    )
    candidate_probe_views = clean.build_feature_views(
        x_train_nohum.iloc[: min(len(x_train_nohum), 1000)],
        x_train_nohum.iloc[: min(len(x_train_nohum), 1000)],
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
    )
    local_candidates, raw_names, engineered_names = build_local_candidate_features(
        candidate_probe_views.raw_fit,
        ratio_eps=float(args.ratio_eps),
    )
    if len(raw_names) != 12 or len(engineered_names) != 30 or local_candidates.shape[1] != 42:
        raise ValueError("RandomForest candidate feature count sanity check failed.")

    paths = {
        "summary": output_dir / f"{prefix}_{timestamp}.json",
        "submission": output_dir / f"{prefix}_{timestamp}.csv",
        "blend_weights": output_dir / f"{prefix}_{timestamp}_conditional_blend_weights.csv",
        "oof_blend_modelspace": output_dir / f"{prefix}_{timestamp}_oof_blend_modelspace.csv",
        "oof_blend_full": output_dir / f"{prefix}_{timestamp}_oof_blend_full.csv",
        "test_blend_modelspace": output_dir / f"{prefix}_{timestamp}_test_blend_modelspace.csv",
        "feature_frequency": output_dir / f"{prefix}_{timestamp}_rf_feature_frequency.csv",
        "optuna_trials": output_dir / f"{prefix}_{timestamp}_optuna_trials.json",
    }
    for model_name in MODEL_ORDER:
        paths[f"{model_name}_oof"] = output_dir / f"{prefix}_{timestamp}_{model_name}_oof.csv"
        paths[f"{model_name}_test"] = output_dir / f"{prefix}_{timestamp}_{model_name}_test.csv"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(paths["submission"], index=False)
    blend_weights.to_csv(paths["blend_weights"], index=False)
    blended_oof_model.to_csv(paths["oof_blend_modelspace"], index=True)
    blended_oof_full.to_csv(paths["oof_blend_full"], index=True)
    blended_test_model.to_csv(paths["test_blend_modelspace"], index=True)
    feature_frequency.to_csv(paths["feature_frequency"], index=False)
    paths["optuna_trials"].write_text(json.dumps(optuna_reports, indent=2), encoding="utf-8")
    for model_name in MODEL_ORDER:
        model_oofs[model_name].to_csv(paths[f"{model_name}_oof"], index=True)
        model_tests[model_name].to_csv(paths[f"{model_name}_test"], index=True)

    reference_summary = latest_reference_summary(resolve_path(args.reference_dir))
    summary = {
        "generated_at_utc": timestamp,
        "model": "Optuna ET rowagg/allpool + conditional RandomForest local expert blender",
        "source_reference_dir": display_path(resolve_path(args.reference_dir)),
        "reference_summary_loaded": reference_summary is not None,
        "training": {
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "sample_weighting": "disabled" if args.no_sample_weights else "1.2 if Humidity>=0.6 else 1.0",
            "postprocessing": False,
        },
        "humidity_mask": {
            "ada_active_low": float(args.ada_low),
            "ada_active_high": float(args.ada_high),
            "train_inside_rows": int(train_allowed.sum()),
            "test_inside_rows": int(test_allowed.sum()),
            "rf_weight_outside_interval": 0.0,
        },
        "feature_engineering": {
            "extra_trees_views": "reference rowagg(26) + allpool(197), both no Humidity",
            "rf_raw_features": raw_names,
            "rf_engineered_feature_count": len(engineered_names),
            "rf_candidate_count": int(local_candidates.shape[1]),
            "rf_selected_per_fold": int(args.selected_features),
            "selection_method": "mean absolute Pearson correlation over modeled targets, fit-fold only",
        },
        "best_params": best_params,
        "base_scores": base_scores,
        "blend_scores": blend_scores,
        "blend_diagnostics": {
            "inside_mean_rf_weight": float(
                blend_weights.loc[blend_weights["zone"] == "inside_045_080", LOCAL_EXPERT].mean()
            ),
            "outside_max_rf_weight": float(
                blend_weights.loc[blend_weights["zone"] == "outside_045_080", LOCAL_EXPERT].max()
            ),
        },
        "fold_reports": fold_reports,
        "full_fit_reports": full_fit_reports,
        "reference_comparison": {
            "loaded_reference_summary": reference_summary,
            "note": "Use blend_scores.global_wrmse and inside_wrmse against the reference folder summary if present.",
        },
        "artifacts": {key: display_path(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Submission written to {paths['submission']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
