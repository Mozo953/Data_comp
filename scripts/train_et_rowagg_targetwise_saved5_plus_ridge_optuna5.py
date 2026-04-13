from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from odor_competition.data import build_submission_frame, load_modeling_data  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402

VERBOSE = False
RIDGE_MODEL_NAME = "ridge_rowagg_optuna5"


@dataclass(frozen=True)
class CalibrationParams:
    slope: float
    intercept: float
    target_mean: float


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def verbose_log(message: str) -> None:
    if VERBOSE:
        log_progress(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse the 5 saved ExtraTrees from model20, train one Ridge model with Optuna(5 trials), "
            "then fit the usual target-wise Dirichlet blender and post-process."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--input-dir",
        default="artifacts_extratrees_corr_optuna/19_et_rowagg_dirichlet_shrinkage_0.1413",
        help="Directory containing the saved ET *_oof.csv / *_test.csv predictions.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional source ET summary JSON. If omitted, the latest JSON in --input-dir is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/22_et_rowagg_targetwise_saved5_plus_ridge_optuna5",
    )
    parser.add_argument("--submission-prefix", default="et_rowagg_targetwise_saved5_plus_ridge_optuna5")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--ridge-optuna-trials", type=int, default=5)
    parser.add_argument("--ridge-holdout-fraction", type=float, default=0.2)
    parser.add_argument("--dirichlet-samples", type=int, default=12000)
    parser.add_argument("--dirichlet-alpha", type=float, default=2.5)
    parser.add_argument("--dirichlet-batch-size", type=int, default=512)
    parser.add_argument("--shrinkage-grid-size", type=int, default=25)
    parser.add_argument(
        "--top-n-models",
        type=int,
        default=5,
        help="If --model-names is not provided, keep the top-N ET variants from the source summary by OOF score.",
    )
    parser.add_argument(
        "--model-names",
        nargs="*",
        default=None,
        help="Optional explicit list of saved ET model names to reuse exactly.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.cv_folds != 3:
        raise ValueError("Ce script est fixe pour une CV=3.")
    if args.ridge_optuna_trials < 1:
        raise ValueError("--ridge-optuna-trials must be >= 1.")
    if not 0.0 < args.ridge_holdout_fraction < 1.0:
        raise ValueError("--ridge-holdout-fraction must be in (0,1).")
    if args.dirichlet_samples < 1000:
        raise ValueError("--dirichlet-samples must be >= 1000.")
    if args.dirichlet_alpha <= 0.0:
        raise ValueError("--dirichlet-alpha must be > 0.")
    if args.dirichlet_batch_size < 1:
        raise ValueError("--dirichlet-batch-size must be >= 1.")
    if args.shrinkage_grid_size < 2:
        raise ValueError("--shrinkage-grid-size must be >= 2.")
    if args.top_n_models < 2:
        raise ValueError("--top-n-models must be >= 2.")

    return args


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def resolve_summary_json(input_dir: Path, summary_json_arg: str | None) -> Path:
    if summary_json_arg:
        summary_path = resolve_path(summary_json_arg)
        if not summary_path.exists():
            raise FileNotFoundError(f"summary JSON not found: {summary_path}")
        return summary_path

    candidates = sorted(input_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No JSON summary found in {input_dir}")
    return candidates[0]


def load_source_summary(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text())
    if "variants" not in payload or not isinstance(payload["variants"], dict):
        raise KeyError(f"Invalid source summary, missing 'variants': {summary_path}")
    return payload


def select_model_names(summary: dict, *, top_n_models: int, explicit_names: list[str] | None) -> list[str]:
    available_names = list(summary["variants"].keys())
    if explicit_names:
        missing = [name for name in explicit_names if name not in summary["variants"]]
        if missing:
            raise KeyError(f"Unknown model names requested: {missing}. Available={available_names}")
        if len(explicit_names) < 2:
            raise ValueError("At least two model names are required for blending.")
        return explicit_names

    ordered = sorted(
        summary["variants"].items(),
        key=lambda item: float(item[1].get("oof_score", item[1].get("optuna_best_score", float("inf")))),
    )
    selected = [name for name, _ in ordered[:top_n_models]]
    if len(selected) < 2:
        raise ValueError("Need at least two saved models to build a blender.")
    return selected


def build_rowwise_aggregated_features(features: pd.DataFrame) -> pd.DataFrame:
    values = features.to_numpy(dtype=float)
    p10 = np.percentile(values, 10, axis=1)
    p25 = np.percentile(values, 25, axis=1)
    p50 = np.percentile(values, 50, axis=1)
    p75 = np.percentile(values, 75, axis=1)
    p90 = np.percentile(values, 90, axis=1)

    aggregated = pd.DataFrame(
        {
            "row_mean": np.mean(values, axis=1),
            "row_std": np.std(values, axis=1),
            "row_min": np.min(values, axis=1),
            "row_max": np.max(values, axis=1),
            "row_range": np.max(values, axis=1) - np.min(values, axis=1),
            "row_p10": p10,
            "row_p25": p25,
            "row_p50": p50,
            "row_p75": p75,
            "row_p90": p90,
            "row_iqr": p75 - p25,
            "row_mad": np.median(np.abs(values - p50[:, None]), axis=1),
            "row_l1": np.linalg.norm(values, ord=1, axis=1),
            "row_l2": np.linalg.norm(values, ord=2, axis=1),
        },
        index=features.index,
    )
    return pd.concat([features, aggregated], axis=1)


def build_ridge_features(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        build_rowwise_aggregated_features(X_train_raw).astype(np.float32),
        build_rowwise_aggregated_features(X_test_raw).astype(np.float32),
    )


def read_prediction_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "row_index"})
    return frame


def normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_index: pd.Index,
    expected_columns: list[str],
    kind: str,
    model_name: str,
) -> pd.DataFrame:
    working = frame.copy()
    if "row_index" in working.columns:
        working = working.set_index("row_index")
    elif len(working) == len(expected_index):
        working.index = expected_index
    else:
        raise ValueError(
            f"{kind} frame for {model_name} has {len(working)} rows, expected {len(expected_index)} rows."
        )

    missing_columns = [column for column in expected_columns if column not in working.columns]
    if missing_columns:
        raise ValueError(f"{kind} frame for {model_name} is missing columns: {missing_columns}")

    working = working[expected_columns].copy().reindex(expected_index)
    if working.isnull().any().any():
        raise ValueError(f"{kind} frame for {model_name} contains NaNs after reindexing.")
    return working.astype(np.float32)


def load_saved_predictions(
    input_dir: Path,
    model_names: list[str],
    *,
    train_index: pd.Index,
    test_index: pd.Index,
    model_targets: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    model_oofs: dict[str, pd.DataFrame] = {}
    model_tests: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        oof_path = input_dir / f"{model_name}_oof.csv"
        test_path = input_dir / f"{model_name}_test.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing OOF file for {model_name}: {oof_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file for {model_name}: {test_path}")

        model_oofs[model_name] = normalize_prediction_frame(
            read_prediction_csv(oof_path),
            expected_index=train_index,
            expected_columns=model_targets,
            kind="OOF",
            model_name=model_name,
        )
        model_tests[model_name] = normalize_prediction_frame(
            read_prediction_csv(test_path),
            expected_index=test_index,
            expected_columns=model_targets,
            kind="test",
            model_name=model_name,
        )

    return model_oofs, model_tests


def make_ridge_pipeline(alpha: float, fit_intercept: bool) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)),
        ]
    )


def optimize_ridge_params(
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    random_state: int,
    holdout_fraction: float,
    n_trials: int,
) -> tuple[dict, float]:
    X_fit, X_valid, y_fit_model, _, y_fit_full, y_valid_full = train_test_split(
        X_train,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        model = make_ridge_pipeline(alpha=alpha, fit_intercept=fit_intercept)
        model.fit(X_fit, y_fit_model)
        pred_valid = pd.DataFrame(
            model.predict(X_valid),
            columns=y_train_model.columns,
            index=X_valid.index,
        ).astype(np.float32)
        score = float(competition_rmse(y_valid_full, schema.expand_predictions(pred_valid)))
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {
        "alpha": float(study.best_params["alpha"]),
        "fit_intercept": bool(study.best_params["fit_intercept"]),
    }
    return best_params, float(study.best_value)


def fit_ridge_cv_and_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    ridge_params: dict,
    cv_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], float]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_pred = pd.DataFrame(index=X_train.index, columns=y_train_model.columns, dtype=np.float32)
    fold_reports: list[dict] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        model = make_ridge_pipeline(
            alpha=float(ridge_params["alpha"]),
            fit_intercept=bool(ridge_params["fit_intercept"]),
        )
        model.fit(X_fit, y_fit_model)
        fold_pred = pd.DataFrame(
            model.predict(X_valid),
            columns=y_train_model.columns,
            index=X_valid.index,
        ).clip(0.0, 1.0).astype(np.float32)
        oof_pred.loc[X_valid.index, :] = fold_pred

        fold_score = float(competition_rmse(y_valid_full, schema.expand_predictions(fold_pred)))
        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse": fold_score,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "feature_count": int(X_fit.shape[1]),
            }
        )
        verbose_log(f"{RIDGE_MODEL_NAME}: fold {fold_idx}/{cv_folds} rmse={fold_score:.6f}")

    full_model = make_ridge_pipeline(
        alpha=float(ridge_params["alpha"]),
        fit_intercept=bool(ridge_params["fit_intercept"]),
    )
    full_model.fit(X_train, y_train_model)
    test_pred = pd.DataFrame(
        full_model.predict(X_test),
        columns=y_train_model.columns,
        index=X_test.index,
    ).clip(0.0, 1.0).astype(np.float32)

    oof_score = float(competition_rmse(y_train_full, schema.expand_predictions(oof_pred)))
    return oof_pred, test_pred, fold_reports, oof_score


def weighted_rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weights = np.where(y_true >= 0.5, 1.2, 1.0).astype(np.float32)
    clipped = np.clip(y_pred, 0.0, 1.0).astype(np.float32)
    return float(np.sqrt(np.mean(weights * np.square(clipped - y_true))))


def optimize_targetwise_dirichlet_blend(
    model_oofs: dict[str, pd.DataFrame],
    y_train_model: pd.DataFrame,
    *,
    sample_count: int,
    alpha: float,
    batch_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model_names = list(model_oofs.keys())
    target_names = list(y_train_model.columns)
    rng = np.random.default_rng(random_state)

    candidate_weights = rng.dirichlet(
        np.full(len(model_names), alpha, dtype=np.float32),
        size=sample_count,
    ).astype(np.float32)
    equal_weights = np.full((1, len(model_names)), 1.0 / len(model_names), dtype=np.float32)
    unit_weights = np.eye(len(model_names), dtype=np.float32)
    candidate_weights = np.vstack([equal_weights, unit_weights, candidate_weights]).astype(np.float32)

    target_weights = pd.DataFrame(index=target_names, columns=model_names, dtype=np.float32)
    blended_oof = pd.DataFrame(index=y_train_model.index, columns=target_names, dtype=np.float32)
    target_scores: dict[str, float] = {}

    for target in target_names:
        stacked = np.column_stack([model_oofs[name][target].to_numpy(dtype=np.float32) for name in model_names])
        y_true = y_train_model[target].to_numpy(dtype=np.float32)
        competition_weights = np.where(y_true >= 0.5, 1.2, 1.0).astype(np.float32)
        best_weights: np.ndarray | None = None
        best_pred: np.ndarray | None = None
        best_mse = float("inf")

        for start in range(0, len(candidate_weights), batch_size):
            stop = min(start + batch_size, len(candidate_weights))
            weight_chunk = candidate_weights[start:stop]
            blended_chunk = np.clip(stacked @ weight_chunk.T, 0.0, 1.0)
            weighted_mse_chunk = np.mean(
                competition_weights[:, None] * np.square(blended_chunk - y_true[:, None]),
                axis=0,
            )
            best_local_idx = int(np.argmin(weighted_mse_chunk))
            best_local_mse = float(weighted_mse_chunk[best_local_idx])

            if best_local_mse < best_mse:
                best_mse = best_local_mse
                best_weights = weight_chunk[best_local_idx].copy()
                best_pred = blended_chunk[:, best_local_idx].copy()

        if best_weights is None or best_pred is None:
            raise RuntimeError(f"Target-wise Dirichlet search failed for target {target}.")

        target_weights.loc[target] = best_weights
        blended_oof[target] = best_pred.astype(np.float32)
        target_scores[target] = weighted_rmse_1d(y_true, best_pred)
        verbose_log(f"Target blend {target}: rmse={target_scores[target]:.6f}, weights={target_weights.loc[target].to_dict()}")

    return target_weights, blended_oof, target_scores


def apply_targetwise_blend(model_predictions: dict[str, pd.DataFrame], target_weights: pd.DataFrame) -> pd.DataFrame:
    target_names = list(target_weights.index)
    blended = pd.DataFrame(
        index=next(iter(model_predictions.values())).index,
        columns=target_names,
        dtype=np.float32,
    )

    for target in target_names:
        total = np.zeros(len(blended), dtype=np.float32)
        for model_name, predictions in model_predictions.items():
            total += float(target_weights.loc[target, model_name]) * predictions[target].to_numpy(dtype=np.float32)
        blended[target] = np.clip(total, 0.0, 1.0).astype(np.float32)

    return blended


def fit_linear_calibration(
    predictions: pd.DataFrame,
    y_train_model: pd.DataFrame,
) -> tuple[dict[str, CalibrationParams], pd.DataFrame]:
    calibration: dict[str, CalibrationParams] = {}
    calibrated = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=np.float32)

    for target in predictions.columns:
        x = predictions[target].to_numpy(dtype=np.float32)
        y = y_train_model[target].to_numpy(dtype=np.float32)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_centered = x - x_mean
        variance = float(np.mean(np.square(x_centered)))

        if variance <= 1e-12:
            slope = 1.0
            intercept = y_mean - x_mean
        else:
            covariance = float(np.mean(x_centered * (y - y_mean)))
            slope = covariance / variance
            intercept = y_mean - (slope * x_mean)

        calibrated[target] = np.clip((slope * x) + intercept, 0.0, 1.0).astype(np.float32)
        calibration[target] = CalibrationParams(
            slope=float(slope),
            intercept=float(intercept),
            target_mean=float(y_mean),
        )

    return calibration, calibrated


def apply_linear_calibration(predictions: pd.DataFrame, calibration: dict[str, CalibrationParams]) -> pd.DataFrame:
    calibrated = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=np.float32)
    for target in predictions.columns:
        params = calibration[target]
        raw = predictions[target].to_numpy(dtype=np.float32)
        calibrated[target] = np.clip((params.slope * raw) + params.intercept, 0.0, 1.0).astype(np.float32)
    return calibrated


def optimize_global_shrinkage(
    calibrated_oof: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    grid_size: int,
) -> tuple[float, float, pd.DataFrame]:
    alpha_grid = np.linspace(0.92, 0.99, grid_size, dtype=np.float32)
    target_means = y_train_model.mean(axis=0)
    best_alpha = float(alpha_grid[0])
    best_score = float("inf")
    best_pred: pd.DataFrame | None = None

    for alpha in alpha_grid:
        shrunk = calibrated_oof.mul(float(alpha)).add(target_means.mul(float(1.0 - alpha)), axis="columns")
        shrunk = shrunk.clip(0.0, 1.0).astype(np.float32)
        score = float(competition_rmse(y_train_full, schema.expand_predictions(shrunk)))
        if score < best_score:
            best_alpha = float(alpha)
            best_score = score
            best_pred = shrunk

    if best_pred is None:
        raise RuntimeError("Shrinkage optimization failed.")
    return best_alpha, best_score, best_pred


def apply_global_shrinkage(
    predictions: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    target_means = y_train_model.mean(axis=0)
    return predictions.mul(alpha).add(target_means.mul(1.0 - alpha), axis="columns").clip(0.0, 1.0).astype(np.float32)


def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = bool(args.verbose)

    data_dir = resolve_path(args.data_dir)
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolve_summary_json(input_dir, args.summary_json)
    source_summary = load_source_summary(summary_path)
    selected_model_names = select_model_names(
        source_summary,
        top_n_models=args.top_n_models,
        explicit_names=args.model_names,
    )

    log_progress(f"Loading cleaned modeling data and saved predictions for {len(selected_model_names)} ET models")
    bundle = load_modeling_data(data_dir)
    schema = bundle.schema
    y_train_model = bundle.y_train_model
    y_train_full = bundle.y_train_full
    test_ids = bundle.data.x_test["ID"]

    model_oofs, model_tests = load_saved_predictions(
        input_dir,
        selected_model_names,
        train_index=y_train_model.index,
        test_index=bundle.x_test_raw.index,
        model_targets=list(y_train_model.columns),
    )

    log_progress("Building Ridge features from cleaned raw + rowagg")
    X_train_ridge, X_test_ridge = build_ridge_features(bundle.x_train_raw, bundle.x_test_raw)

    log_progress("Optuna(5 trials) for the Ridge model")
    ridge_params, ridge_holdout_score = optimize_ridge_params(
        X_train_ridge,
        y_train_model,
        y_train_full,
        schema,
        random_state=args.random_state,
        holdout_fraction=args.ridge_holdout_fraction,
        n_trials=args.ridge_optuna_trials,
    )

    log_progress("Running Ridge in CV=3 and fitting full-data predictions")
    ridge_oof, ridge_test, ridge_fold_reports, ridge_oof_score = fit_ridge_cv_and_test(
        X_train_ridge,
        X_test_ridge,
        y_train_model,
        y_train_full,
        schema,
        ridge_params=ridge_params,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    model_oofs[RIDGE_MODEL_NAME] = ridge_oof
    model_tests[RIDGE_MODEL_NAME] = ridge_test

    log_progress("Training the target-wise Dirichlet simplex blender")
    target_weights, blended_oof, per_target_blend_scores = optimize_targetwise_dirichlet_blend(
        model_oofs,
        y_train_model,
        sample_count=args.dirichlet_samples,
        alpha=args.dirichlet_alpha,
        batch_size=args.dirichlet_batch_size,
        random_state=args.random_state + 5000,
    )
    blended_oof_score = float(competition_rmse(y_train_full, schema.expand_predictions(blended_oof)))
    blended_test = apply_targetwise_blend(model_tests, target_weights)

    log_progress("Applying the usual post-process")
    calibration, calibrated_oof = fit_linear_calibration(blended_oof, y_train_model)
    calibrated_test = apply_linear_calibration(blended_test, calibration)
    calibrated_oof_score = float(competition_rmse(y_train_full, schema.expand_predictions(calibrated_oof)))
    best_alpha, shrunk_oof_score, shrunk_oof = optimize_global_shrinkage(
        calibrated_oof,
        y_train_model,
        y_train_full,
        schema,
        grid_size=args.shrinkage_grid_size,
    )
    shrunk_test = apply_global_shrinkage(calibrated_test, y_train_model, alpha=best_alpha)

    submission = build_submission_frame(test_ids, schema.expand_predictions(shrunk_test))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    target_weights_path = output_dir / f"{args.submission_prefix}_{timestamp}_target_weights.csv"
    blended_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_blend.csv"
    calibrated_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_calibrated.csv"
    shrunk_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_shrunk.csv"
    ridge_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_{RIDGE_MODEL_NAME}_oof.csv"
    ridge_test_path = output_dir / f"{args.submission_prefix}_{timestamp}_{RIDGE_MODEL_NAME}_test.csv"
    summary_out_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"

    submission.to_csv(submission_path, index=False)
    target_weights.to_csv(target_weights_path)
    blended_oof.to_csv(blended_oof_path, index=True)
    calibrated_oof.to_csv(calibrated_oof_path, index=True)
    shrunk_oof.to_csv(shrunk_oof_path, index=True)
    ridge_oof.to_csv(ridge_oof_path, index=True)
    ridge_test.to_csv(ridge_test_path, index=False)

    selected_variant_reports = {
        model_name: source_summary["variants"][model_name]
        for model_name in selected_model_names
    }

    summary = {
        "generated_at_utc": timestamp,
        "model": "5 saved ET + Ridge(optuna5) -> target-wise Dirichlet simplex blender + usual postprocess",
        "data_dir": str(data_dir),
        "source_artifacts": {
            "input_dir": str(input_dir),
            "summary_json": str(summary_path),
            "selected_model_names": selected_model_names,
            "selection_strategy": (
                "explicit_model_names"
                if args.model_names
                else f"top_{int(args.top_n_models)}_by_source_oof_score"
            ),
            "added_model_name": RIDGE_MODEL_NAME,
        },
        "notes": {
            "base_model_training": "The 5 ET base models are reused from saved OOF/test predictions.",
            "ridge_training": "A new Ridge base model is optimized with Optuna(5 trials), then rerun in CV=3.",
            "ridge_features": "cleaned raw + row-wise aggregated features",
            "blend_mode": "target_wise_dirichlet_simplex",
            "postprocess": "same as train_et_rowagg_dirichlet_shrinkage.py: per-target linear calibration + global shrinkage",
            "d15_removed_from_training": bool("d15" not in y_train_model.columns),
        },
        "training": {
            "cv_folds": int(args.cv_folds),
            "ridge_optuna_trials": int(args.ridge_optuna_trials),
            "ridge_holdout_fraction": float(args.ridge_holdout_fraction),
            "dirichlet_samples": int(args.dirichlet_samples),
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "dirichlet_batch_size": int(args.dirichlet_batch_size),
            "shrinkage_grid_size": int(args.shrinkage_grid_size),
            "random_state": int(args.random_state),
        },
        "selected_variants": selected_variant_reports,
        "ridge": {
            "best_params": ridge_params,
            "best_holdout_rmse": float(ridge_holdout_score),
            "cv_fold_reports": ridge_fold_reports,
            "oof_score": float(ridge_oof_score),
            "feature_count": int(X_train_ridge.shape[1]),
        },
        "blend": {
            "model_names": list(model_oofs.keys()),
            "oof_score_before_calibration": float(blended_oof_score),
            "per_target_rmse_model_space": per_target_blend_scores,
            "target_weights_path": str(target_weights_path.relative_to(ROOT)),
            "target_weights": {
                target: {
                    model_name: float(target_weights.loc[target, model_name])
                    for model_name in target_weights.columns
                }
                for target in target_weights.index
            },
        },
        "calibration": {
            "oof_score_after_linear_calibration": float(calibrated_oof_score),
            "per_target": {target: asdict(params) for target, params in calibration.items()},
        },
        "shrinkage": {
            "alpha_search_interval": [0.92, 0.99],
            "grid_size": int(args.shrinkage_grid_size),
            "best_alpha": float(best_alpha),
            "oof_score_after_shrinkage": float(shrunk_oof_score),
        },
        "artifacts": {
            "submission_path": str(submission_path.relative_to(ROOT)),
            "target_weights_path": str(target_weights_path.relative_to(ROOT)),
            "blended_oof_path": str(blended_oof_path.relative_to(ROOT)),
            "calibrated_oof_path": str(calibrated_oof_path.relative_to(ROOT)),
            "shrunk_oof_path": str(shrunk_oof_path.relative_to(ROOT)),
            "ridge_oof_path": str(ridge_oof_path.relative_to(ROOT)),
            "ridge_test_path": str(ridge_test_path.relative_to(ROOT)),
            "summary_path": str(summary_out_path.relative_to(ROOT)),
        },
    }

    summary_out_path.write_text(json.dumps(summary, indent=2))
    log_progress(f"Finished. Submission written to {submission_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
