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
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from odor_competition.data import build_submission_frame, compress_targets, load_modeling_data  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


@dataclass(frozen=True)
class CalibrationParams:
    slope: float
    intercept: float
    shrinkage: float
    target_mean: float


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_named_paths(items: list[str], *, flag_name: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{flag_name} entries must look like name=path, got: {item}")
        name, raw_path = item.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"{flag_name} contains an empty model name.")
        if key in parsed:
            raise ValueError(f"{flag_name} contains duplicate model name: {key}")
        parsed[key] = Path(raw_path.strip())
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend precomputed OOF/test predictions with a constrained Dirichlet target-wise ensemble and Optuna shrinkage calibration."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/15_blend_saved_oofs")
    parser.add_argument("--submission-prefix", default="blend_saved_oofs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dirichlet-samples", type=int, default=2000)
    parser.add_argument("--dirichlet-alpha", type=float, default=4.0)
    parser.add_argument("--blend-min-weight", type=float, default=0.05)
    parser.add_argument("--calibration-optuna-trials", type=int, default=25)
    parser.add_argument(
        "--model-oof",
        action="append",
        default=[],
        help="Named OOF file in the form model_name=path/to/oof.csv. Repeat once per model.",
    )
    parser.add_argument(
        "--model-test",
        action="append",
        default=[],
        help="Named test-prediction file in the form model_name=path/to/test_preds.csv. Repeat once per model.",
    )
    args = parser.parse_args()

    if len(args.model_oof) < 2:
        raise ValueError("Provide at least two --model-oof entries.")
    if len(args.model_test) < 2:
        raise ValueError("Provide at least two --model-test entries.")
    if args.dirichlet_samples < 32:
        raise ValueError("--dirichlet-samples must be >= 32.")
    if args.dirichlet_alpha <= 0.0:
        raise ValueError("--dirichlet-alpha must be > 0.")
    if not 0.0 <= args.blend_min_weight < 1.0:
        raise ValueError("--blend-min-weight must be in [0,1).")
    if args.calibration_optuna_trials < 1:
        raise ValueError("--calibration-optuna-trials must be >= 1.")

    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_prediction_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "row_index"})
    return frame


def normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    schema,
    expected_ids: pd.Series,
    kind: str,
) -> pd.DataFrame:
    working = frame.copy()

    if "ID" in working.columns:
        working = working.set_index("ID").reindex(expected_ids).reset_index(drop=True)
    elif "row_index" in working.columns:
        working = working.drop(columns=["row_index"])

    if list(working.columns) == schema.model_targets:
        return working.astype(np.float32)

    original_targets = [column for column in schema.original_targets if column in working.columns]
    if len(original_targets) == len(schema.original_targets):
        compressed = compress_targets(working[schema.original_targets], schema)
        return compressed.astype(np.float32)

    raise ValueError(
        f"{kind} predictions must contain either modeled targets {schema.model_targets} "
        f"or all original targets {schema.original_targets}. Got columns: {list(working.columns)}"
    )


def load_named_prediction_frames(
    named_paths: dict[str, Path],
    *,
    schema,
    expected_ids: pd.Series,
    kind: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for model_name, raw_path in named_paths.items():
        path = resolve_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{kind} file not found for {model_name}: {path}")
        frame = read_prediction_csv(path)
        frames[model_name] = normalize_prediction_frame(frame, schema=schema, expected_ids=expected_ids, kind=kind)
    return frames


def weighted_rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    weights = np.where(y_true >= 0.5, 1.2, 1.0).astype(np.float32)
    return float(np.sqrt(np.mean(weights * np.square(y_pred - y_true))))


def optimize_dirichlet_blend_per_target(
    model_oofs: dict[str, pd.DataFrame],
    y_train_model: pd.DataFrame,
    *,
    sample_count: int,
    alpha: float,
    min_weight: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_progress(
        f"Blend: optimizing target-wise Dirichlet weights with {sample_count} samples, alpha={alpha:.2f}, min_weight={min_weight:.3f}"
    )
    model_names = list(model_oofs.keys())
    target_names = list(y_train_model.columns)
    model_count = len(model_names)

    if model_count * min_weight >= 1.0:
        raise ValueError("blend_min_weight is too large for the number of models.")

    rng = np.random.default_rng(random_state)
    raw_weights = rng.dirichlet(np.full(model_count, alpha, dtype=np.float32), size=sample_count).astype(np.float32)
    scaled_weights = np.full((sample_count, model_count), min_weight, dtype=np.float32)
    scaled_weights += (1.0 - (model_count * min_weight)) * raw_weights
    equal_weights = np.full((1, model_count), 1.0 / model_count, dtype=np.float32)
    candidate_weights = np.vstack([equal_weights, scaled_weights]).astype(np.float32)

    weight_frame = pd.DataFrame(index=target_names, columns=model_names, dtype=np.float32)
    blended_oof = pd.DataFrame(index=y_train_model.index, columns=target_names, dtype=np.float32)

    for target in target_names:
        stacked = np.column_stack([model_oofs[name][target].to_numpy(dtype=np.float32) for name in model_names])
        y_true = y_train_model[target].to_numpy(dtype=np.float32)
        weights = np.where(y_true >= 0.5, 1.2, 1.0).astype(np.float32)
        blended_candidates = np.clip(stacked @ candidate_weights.T, 0.0, 1.0)
        weighted_mse = np.mean(weights[:, None] * np.square(blended_candidates - y_true[:, None]), axis=0)
        best_idx = int(np.argmin(weighted_mse))
        best_weights = candidate_weights[best_idx]
        weight_frame.loc[target] = best_weights
        blended_oof[target] = np.clip(stacked @ best_weights, 0.0, 1.0)

    log_progress("Blend: target-wise Dirichlet optimization done")
    return weight_frame, blended_oof


def fit_target_calibration(
    blended_oof: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    n_trials: int,
    random_state: int,
) -> tuple[dict[str, CalibrationParams], pd.DataFrame]:
    log_progress("Calibration: fitting per-target linear calibration and Optuna shrinkage")
    calibration: dict[str, CalibrationParams] = {}
    calibrated = pd.DataFrame(index=blended_oof.index, columns=blended_oof.columns, dtype=np.float32)

    for target_idx, target in enumerate(blended_oof.columns):
        x = np.clip(blended_oof[target].to_numpy(dtype=np.float32), 0.0, 1.0)
        y = y_train_model[target].to_numpy(dtype=np.float32)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_centered = x - x_mean
        variance = float(np.mean(np.square(x_centered)))
        if variance <= 1e-12:
            slope = 1.0
            intercept = 0.0
        else:
            covariance = float(np.mean(x_centered * (y - y_mean)))
            slope = covariance / variance
            intercept = y_mean - (slope * x_mean)

        linear = np.clip((slope * x) + intercept, 0.0, 1.0).astype(np.float32)

        def objective(trial: optuna.Trial) -> float:
            shrinkage = trial.suggest_float("shrinkage", 0.0, 0.15)
            candidate = np.clip(((1.0 - shrinkage) * linear) + (shrinkage * y_mean), 0.0, 1.0)
            return weighted_rmse_1d(y, candidate)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=random_state + 10_000 + target_idx),
        )
        study.enqueue_trial({"shrinkage": 0.0})
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_shrinkage = float(study.best_params["shrinkage"])
        best_prediction = np.clip(((1.0 - best_shrinkage) * linear) + (best_shrinkage * y_mean), 0.0, 1.0)

        calibration[target] = CalibrationParams(
            slope=float(slope),
            intercept=float(intercept),
            shrinkage=best_shrinkage,
            target_mean=float(y_mean),
        )
        calibrated[target] = best_prediction.astype(np.float32)

    log_progress("Calibration: done")
    return calibration, calibrated


def apply_target_calibration(predictions: pd.DataFrame, calibration: dict[str, CalibrationParams]) -> pd.DataFrame:
    calibrated = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=np.float32)
    for target in predictions.columns:
        params = calibration[target]
        raw = predictions[target].to_numpy(dtype=np.float32)
        linear = (params.slope * raw) + params.intercept
        shrunk = ((1.0 - params.shrinkage) * linear) + (params.shrinkage * params.target_mean)
        calibrated[target] = np.clip(shrunk, 0.0, 1.0).astype(np.float32)
    return calibrated


def build_blended_test_predictions(
    model_test_predictions: dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
) -> pd.DataFrame:
    target_names = list(target_weights.index)
    blended = pd.DataFrame(index=next(iter(model_test_predictions.values())).index, columns=target_names, dtype=np.float32)

    for target in target_names:
        total = np.zeros(len(blended), dtype=np.float32)
        for model_name, model_predictions in model_test_predictions.items():
            total += target_weights.loc[target, model_name] * model_predictions[target].to_numpy(dtype=np.float32)
        blended[target] = np.clip(total, 0.0, 1.0)
    return blended


def main() -> None:
    args = parse_args()
    log_progress("Blend-only pipeline: loading data and prediction files")

    data_dir = resolve_path(Path(args.data_dir))
    output_dir = resolve_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_paths = parse_named_paths(args.model_oof, flag_name="--model-oof")
    test_paths = parse_named_paths(args.model_test, flag_name="--model-test")
    if set(oof_paths) != set(test_paths):
        raise ValueError("The model names passed to --model-oof and --model-test must match exactly.")

    bundle = load_modeling_data(data_dir)
    schema = bundle.schema
    y_train_full = bundle.y_train_full
    y_train_model = bundle.y_train_model
    train_ids = bundle.data.x_train["ID"]
    test_ids = bundle.data.x_test["ID"]

    model_oofs = load_named_prediction_frames(oof_paths, schema=schema, expected_ids=train_ids, kind="OOF")
    model_test_predictions = load_named_prediction_frames(test_paths, schema=schema, expected_ids=test_ids, kind="test")

    for model_name in model_oofs:
        if len(model_oofs[model_name]) != len(y_train_model):
            raise ValueError(f"OOF length mismatch for {model_name}.")
        if len(model_test_predictions[model_name]) != len(test_ids):
            raise ValueError(f"Test prediction length mismatch for {model_name}.")

    target_weights, blended_oof = optimize_dirichlet_blend_per_target(
        model_oofs,
        y_train_model,
        sample_count=args.dirichlet_samples,
        alpha=args.dirichlet_alpha,
        min_weight=args.blend_min_weight,
        random_state=args.random_state + 999,
    )
    blended_oof_full = schema.expand_predictions(blended_oof)
    blend_oof_rmse = float(competition_rmse(y_train_full, blended_oof_full))
    log_progress(f"Blend: OOF RMSE before calibration = {blend_oof_rmse:.6f}")

    calibration, calibrated_oof = fit_target_calibration(
        blended_oof,
        y_train_model,
        n_trials=args.calibration_optuna_trials,
        random_state=args.random_state + 2026,
    )
    calibrated_oof_full = schema.expand_predictions(calibrated_oof)
    calibrated_oof_rmse = float(competition_rmse(y_train_full, calibrated_oof_full))
    log_progress(f"Blend: OOF RMSE after calibration = {calibrated_oof_rmse:.6f}")

    blended_test = build_blended_test_predictions(model_test_predictions, target_weights)
    calibrated_test = apply_target_calibration(blended_test, calibration)
    final_predictions_full = schema.expand_predictions(calibrated_test)
    submission = build_submission_frame(test_ids, final_predictions_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    weights_path = output_dir / f"{args.submission_prefix}_{timestamp}_target_weights.csv"
    target_weights.to_csv(weights_path)

    oof_raw_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_raw_model.csv"
    blended_oof.to_csv(oof_raw_path, index=True)

    oof_post_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_postprocessed_model.csv"
    calibrated_oof.to_csv(oof_post_path, index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Blend-only target-wise constrained Dirichlet ensemble from saved OOF/test predictions",
        "data_dir": str(data_dir),
        "notes": {
            "base_model_training": "No base model is refit here. The script only consumes precomputed CV OOF and test predictions.",
            "loss": "Dirichlet blending and post-processing are evaluated with competition weighted RMSE.",
            "blend_constraint": "Target-wise Dirichlet blend keeps alpha=4, minimum per-model weight=0.05, and never includes identity/single-model candidates.",
            "cv_assumption": "The provided OOF files are assumed to come from CV=3 base-model runs.",
        },
        "input_models": {
            model_name: {
                "oof_path": str(resolve_path(oof_paths[model_name])),
                "test_path": str(resolve_path(test_paths[model_name])),
            }
            for model_name in oof_paths
        },
        "training": {
            "dirichlet_samples": int(args.dirichlet_samples),
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "blend_min_weight": float(args.blend_min_weight),
            "calibration_optuna_trials": int(args.calibration_optuna_trials),
            "random_state": int(args.random_state),
        },
        "blend": {
            "oof_rmse_before_calibration": float(blend_oof_rmse),
            "oof_rmse_after_calibration": float(calibrated_oof_rmse),
            "target_weight_path": str(weights_path.relative_to(ROOT)),
            "target_weights": {
                target: {model_name: float(target_weights.loc[target, model_name]) for model_name in target_weights.columns}
                for target in target_weights.index
            },
        },
        "calibration": {target: asdict(params) for target, params in calibration.items()},
        "artifacts": {
            "submission_path": str(submission_path.relative_to(ROOT)),
            "oof_raw_model_path": str(oof_raw_path.relative_to(ROOT)),
            "oof_postprocessed_model_path": str(oof_post_path.relative_to(ROOT)),
        },
    }

    summary_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log_progress(f"Blend-only pipeline: finished. Submission written to {submission_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
