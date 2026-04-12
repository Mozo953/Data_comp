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
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from odor_competition.data import build_submission_frame, load_modeling_data  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402

VERBOSE = False


@dataclass(frozen=True)
class VariantSpec:
    name: str
    dataset_name: str
    max_features: float
    bootstrap: bool
    random_state: int
    n_estimators: int = 500


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
        description="Diversified ExtraTrees ensemble on cleaned raw and row-wise aggregated features with Dirichlet blending, linear calibration, and shrinkage."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/19_et_rowagg_dirichlet_shrinkage")
    parser.add_argument("--submission-prefix", default="et_rowagg_dirichlet_shrinkage")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=3)
    parser.add_argument("--optuna-timeout-sec", type=int, default=900)
    parser.add_argument("--dirichlet-samples", type=int, default=12000)
    parser.add_argument("--dirichlet-alpha", type=float, default=2.5)
    parser.add_argument("--shrinkage-grid-size", type=int, default=25)
    parser.add_argument("--et-n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.cv_folds != 3:
        raise ValueError("Ce pipeline est parametre pour une CV=3 pendant Optuna.")
    if args.optuna_trials < 1:
        raise ValueError("--optuna-trials must be >= 1.")
    if args.optuna_timeout_sec < 60:
        raise ValueError("--optuna-timeout-sec must be >= 60.")
    if args.dirichlet_samples < 1000:
        raise ValueError("--dirichlet-samples must be >= 1000.")
    if args.dirichlet_alpha <= 0.0:
        raise ValueError("--dirichlet-alpha must be > 0.")
    if args.shrinkage_grid_size < 2:
        raise ValueError("--shrinkage-grid-size must be >= 2.")

    return args


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


def build_variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(name="et_raw_mf06_nb", dataset_name="raw_clean", max_features=0.6, bootstrap=False, random_state=42),
        VariantSpec(name="et_raw_mf07_bs", dataset_name="raw_clean", max_features=0.7, bootstrap=True, random_state=314),
        VariantSpec(name="et_raw_mf08_nb", dataset_name="raw_clean", max_features=0.8, bootstrap=False, random_state=2718),
        VariantSpec(name="et_rowagg_mf06_bs", dataset_name="rowagg_clean", max_features=0.6, bootstrap=True, random_state=123),
        VariantSpec(name="et_rowagg_mf07_nb", dataset_name="rowagg_clean", max_features=0.7, bootstrap=False, random_state=777),
        VariantSpec(name="et_rowagg_mf08_bs", dataset_name="rowagg_clean", max_features=0.8, bootstrap=True, random_state=2024),
    ]


def make_et_model(spec: VariantSpec, tuned_params: dict, *, n_jobs: int) -> ExtraTreesRegressor:
    params = {
        "n_estimators": spec.n_estimators,
        "criterion": "squared_error",
        "bootstrap": spec.bootstrap,
        "max_features": spec.max_features,
        "random_state": spec.random_state,
        "n_jobs": n_jobs,
        **tuned_params,
    }
    if not spec.bootstrap:
        params.pop("max_samples", None)
    return ExtraTreesRegressor(**params)


def optimize_variant_params(
    spec: VariantSpec,
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    cv_folds: int,
    n_trials: int,
    timeout_sec: int,
    n_jobs: int,
) -> tuple[dict, float, pd.DataFrame, list[dict]]:
    log_progress(f"{spec.name}: starting Optuna on {spec.dataset_name} ({X_train.shape[1]} features)")
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=spec.random_state)
    trial_oofs: dict[int, pd.DataFrame] = {}
    trial_fold_reports: dict[int, list[dict]] = {}

    def objective(trial: optuna.Trial) -> float:
        tuned_params = {
            "max_depth": trial.suggest_int("max_depth", 14, 30, step=2),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 18, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        }
        if spec.bootstrap:
            tuned_params["max_samples"] = trial.suggest_float("max_samples", 0.65, 0.95)
        verbose_log(f"{spec.name}: trial {trial.number} params={tuned_params}")

        oof_pred = pd.DataFrame(index=X_train.index, columns=y_train_model.columns, dtype=np.float32)
        fold_reports: list[dict] = []
        for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
            X_fit = X_train.iloc[fit_idx]
            X_valid = X_train.iloc[valid_idx]
            y_fit = y_train_model.iloc[fit_idx]
            y_valid_full = y_train_full.iloc[valid_idx]

            model = make_et_model(spec, tuned_params, n_jobs=n_jobs)
            model.fit(X_fit, y_fit)
            fold_pred = pd.DataFrame(
                model.predict(X_valid),
                columns=y_train_model.columns,
                index=X_valid.index,
            )
            oof_pred.iloc[valid_idx] = fold_pred.to_numpy(dtype=np.float32)

            fold_full = schema.expand_predictions(fold_pred)
            partial_score = competition_rmse(y_valid_full, fold_full)
            fold_reports.append(
                {
                    "fold": fold_idx,
                    "rmse": float(partial_score),
                    "fit_rows": int(len(fit_idx)),
                    "valid_rows": int(len(valid_idx)),
                }
            )
            verbose_log(f"{spec.name}: trial {trial.number} fold {fold_idx}/{cv_folds} rmse={partial_score:.6f}")
            trial.report(partial_score, step=fold_idx)
            if trial.should_prune():
                verbose_log(f"{spec.name}: trial {trial.number} pruned at fold {fold_idx}")
                raise optuna.TrialPruned()

        oof_full = schema.expand_predictions(oof_pred)
        score = float(competition_rmse(y_train_full, oof_full))
        trial_oofs[trial.number] = oof_pred.copy()
        trial_fold_reports[trial.number] = fold_reports
        verbose_log(f"{spec.name}: trial {trial.number} final OOF WRMSE={score:.6f}")
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=spec.random_state),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    best_params = dict(study.best_params)
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])
    if "min_samples_split" in best_params:
        best_params["min_samples_split"] = int(best_params["min_samples_split"])
    if "min_samples_leaf" in best_params:
        best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

    log_progress(f"{spec.name}: Optuna done, best OOF WRMSE={study.best_value:.6f}")
    best_trial_number = study.best_trial.number
    verbose_log(f"{spec.name}: best trial={best_trial_number}, best_params={best_params}")
    return best_params, float(study.best_value), trial_oofs[best_trial_number], trial_fold_reports[best_trial_number]


def fit_variant_full_and_test(
    spec: VariantSpec,
    tuned_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    n_jobs: int,
) -> pd.DataFrame:
    verbose_log(f"{spec.name}: fitting final full-data model with params={tuned_params}")
    full_model = make_et_model(spec, tuned_params, n_jobs=n_jobs)
    full_model.fit(X_train, y_train_model)
    test_pred = pd.DataFrame(
        full_model.predict(X_test),
        columns=y_train_model.columns,
        index=X_test.index,
    ).astype(np.float32)
    verbose_log(f"{spec.name}: final test prediction shape={test_pred.shape}")
    return test_pred


def optimize_global_dirichlet_blend(
    model_oofs: dict[str, pd.DataFrame],
    y_train_full: pd.DataFrame,
    schema,
    *,
    sample_count: int,
    alpha: float,
    random_state: int,
) -> tuple[pd.Series, pd.DataFrame, float]:
    model_names = list(model_oofs.keys())
    verbose_log(f"Dirichlet blend: models={model_names}, samples={sample_count}, alpha={alpha}")
    rng = np.random.default_rng(random_state)
    candidate_weights = rng.dirichlet(np.full(len(model_names), alpha, dtype=np.float32), size=sample_count).astype(np.float32)
    unit_weights = np.eye(len(model_names), dtype=np.float32)
    equal_weights = np.full((1, len(model_names)), 1.0 / len(model_names), dtype=np.float32)
    candidate_weights = np.vstack([equal_weights, unit_weights, candidate_weights]).astype(np.float32)

    stacked = np.stack([model_oofs[name].to_numpy(dtype=np.float32) for name in model_names], axis=2)
    best_score = float("inf")
    best_weights = candidate_weights[0]
    best_pred: pd.DataFrame | None = None

    for weights in candidate_weights:
        blended_array = np.tensordot(stacked, weights, axes=([2], [0]))
        blended_model = pd.DataFrame(blended_array, columns=model_oofs[model_names[0]].columns, index=model_oofs[model_names[0]].index)
        score = float(competition_rmse(y_train_full, schema.expand_predictions(blended_model)))
        if score < best_score:
            best_score = score
            best_weights = weights.copy()
            best_pred = blended_model.astype(np.float32)

    if best_pred is None:
        raise RuntimeError("Dirichlet blending failed to produce a prediction frame.")

    weight_series = pd.Series(best_weights, index=model_names, dtype=np.float32)
    verbose_log(f"Dirichlet blend: best_score={best_score:.6f}, best_weights={weight_series.to_dict()}")
    return weight_series, best_pred, best_score


def apply_blend_weights(model_predictions: dict[str, pd.DataFrame], weights: pd.Series) -> pd.DataFrame:
    blended = pd.DataFrame(
        0.0,
        index=next(iter(model_predictions.values())).index,
        columns=next(iter(model_predictions.values())).columns,
        dtype=np.float32,
    )
    for model_name, predictions in model_predictions.items():
        blended += float(weights[model_name]) * predictions.astype(np.float32)
    return blended.astype(np.float32)


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
        verbose_log(
            f"Calibration {target}: slope={slope:.6f}, intercept={intercept:.6f}, target_mean={y_mean:.6f}"
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
        verbose_log(f"Shrinkage alpha={float(alpha):.6f} -> OOF WRMSE={score:.6f}")
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


def maybe_trim_rows(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    test_ids: pd.Series,
    *,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    if max_train_rows is not None:
        X_train = X_train.iloc[:max_train_rows].copy()
        y_train_model = y_train_model.iloc[:max_train_rows].copy()
        y_train_full = y_train_full.iloc[:max_train_rows].copy()
    if max_test_rows is not None:
        X_test = X_test.iloc[:max_test_rows].copy()
        test_ids = test_ids.iloc[:max_test_rows].copy()
    return X_train, X_test, y_train_model, y_train_full, test_ids


def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = bool(args.verbose)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_progress("Loading cleaned modeling data")
    bundle = load_modeling_data(data_dir)
    schema = bundle.schema
    X_train_raw_clean = bundle.x_train_raw
    X_test_raw_clean = bundle.x_test_raw
    y_train_model = bundle.y_train_model
    y_train_full = bundle.y_train_full
    test_ids = bundle.data.x_test["ID"]

    X_train_raw_clean, X_test_raw_clean, y_train_model, y_train_full, test_ids = maybe_trim_rows(
        X_train_raw_clean,
        X_test_raw_clean,
        y_train_model,
        y_train_full,
        test_ids,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    log_progress("Building row-wise aggregated feature view from cleaned raw features")
    X_train_rowagg = build_rowwise_aggregated_features(X_train_raw_clean)
    X_test_rowagg = build_rowwise_aggregated_features(X_test_raw_clean)
    verbose_log(f"raw_clean shape train/test = {X_train_raw_clean.shape}/{X_test_raw_clean.shape}")
    verbose_log(f"rowagg_clean shape train/test = {X_train_rowagg.shape}/{X_test_rowagg.shape}")

    feature_sets = {
        "raw_clean": (X_train_raw_clean, X_test_raw_clean),
        "rowagg_clean": (X_train_rowagg, X_test_rowagg),
    }

    variant_specs = build_variant_specs()
    variant_reports: dict[str, dict] = {}
    model_oofs: dict[str, pd.DataFrame] = {}
    model_tests: dict[str, pd.DataFrame] = {}

    log_progress(f"Humidity/Env dropped from modeling features: {'Env' not in X_train_raw_clean.columns}")
    for spec in variant_specs:
        train_view, test_view = feature_sets[spec.dataset_name]
        tuned_params, optuna_score, best_oof_pred, fold_reports = optimize_variant_params(
            spec,
            train_view,
            y_train_model,
            y_train_full,
            schema,
            cv_folds=args.cv_folds,
            n_trials=args.optuna_trials,
            timeout_sec=args.optuna_timeout_sec,
            n_jobs=args.et_n_jobs,
        )
        test_pred = fit_variant_full_and_test(
            spec,
            tuned_params,
            train_view,
            test_view,
            y_train_model,
            n_jobs=args.et_n_jobs,
        )
        model_oofs[spec.name] = best_oof_pred
        model_tests[spec.name] = test_pred
        variant_reports[spec.name] = {
            "spec": asdict(spec),
            "optuna_best_score": float(optuna_score),
            "oof_score": float(optuna_score),
            "feature_count": int(train_view.shape[1]),
            "tuned_params": tuned_params,
            "fold_reports": fold_reports,
            "oof_origin": "best_optuna_trial",
        }
        oof_path = output_dir / f"{spec.name}_oof.csv"
        test_path = output_dir / f"{spec.name}_test.csv"
        best_oof_pred.to_csv(oof_path, index=True)
        test_pred.to_csv(test_path, index=False)
        log_progress(f"{spec.name}: OOF WRMSE from best Optuna trial = {optuna_score:.6f}")

    log_progress("Searching global Dirichlet blend weights on OOF predictions")
    blend_weights, blended_oof, blended_oof_score = optimize_global_dirichlet_blend(
        model_oofs,
        y_train_full,
        schema,
        sample_count=args.dirichlet_samples,
        alpha=args.dirichlet_alpha,
        random_state=args.random_state + 5000,
    )
    blended_test = apply_blend_weights(model_tests, blend_weights)

    log_progress("Fitting per-target linear calibration")
    calibration, calibrated_oof = fit_linear_calibration(blended_oof, y_train_model)
    calibrated_test = apply_linear_calibration(blended_test, calibration)
    calibrated_oof_score = float(competition_rmse(y_train_full, schema.expand_predictions(calibrated_oof)))

    log_progress("Optimizing global shrinkage alpha on calibrated OOF")
    best_alpha, shrunk_oof_score, shrunk_oof = optimize_global_shrinkage(
        calibrated_oof,
        y_train_model,
        y_train_full,
        schema,
        grid_size=args.shrinkage_grid_size,
    )
    shrunk_test = apply_global_shrinkage(calibrated_test, y_train_model, alpha=best_alpha)
    verbose_log(f"Final shrinkage alpha selected = {best_alpha:.6f}")

    final_predictions = schema.expand_predictions(shrunk_test)
    submission = build_submission_frame(test_ids, final_predictions)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    blend_weights_path = output_dir / f"{args.submission_prefix}_{timestamp}_blend_weights.csv"
    blend_weights.to_frame(name="weight").to_csv(blend_weights_path)

    blended_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_blend.csv"
    blended_oof.to_csv(blended_oof_path, index=True)

    calibrated_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_calibrated.csv"
    calibrated_oof.to_csv(calibrated_oof_path, index=True)

    shrunk_oof_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_shrunk.csv"
    shrunk_oof.to_csv(shrunk_oof_path, index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Diversified ExtraTrees ensemble on cleaned raw + rowwise aggregated features with Dirichlet blend, linear calibration, and global shrinkage",
        "data_dir": str(data_dir),
        "notes": {
            "modeling_features_source": "Cleaned train-only clipped dataset produced by load_modeling_data",
            "humidity_env_dropped": bool("Env" not in X_train_raw_clean.columns),
            "d15_removed_from_training": bool("d15" not in y_train_model.columns),
            "tree_loss": "ExtraTreesRegressor criterion=squared_error (MSE)",
            "oof_source": "best Optuna trial CV predictions reused directly for blend, calibration, and shrinkage",
            "variant_count": int(len(variant_specs)),
            "raw_feature_count": int(X_train_raw_clean.shape[1]),
            "rowagg_feature_count": int(X_train_rowagg.shape[1]),
            "rowwise_added_features": [
                "row_mean",
                "row_std",
                "row_min",
                "row_max",
                "row_range",
                "row_p10",
                "row_p25",
                "row_p50",
                "row_p75",
                "row_p90",
                "row_iqr",
                "row_mad",
                "row_l1",
                "row_l2",
            ],
        },
        "training": {
            "cv_folds": int(args.cv_folds),
            "optuna_trials_per_variant": int(args.optuna_trials),
            "optuna_timeout_sec_per_variant": int(args.optuna_timeout_sec),
            "dirichlet_samples": int(args.dirichlet_samples),
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "et_n_jobs": int(args.et_n_jobs),
            "random_state": int(args.random_state),
        },
        "variants": variant_reports,
        "blend": {
            "weights": {model_name: float(weight) for model_name, weight in blend_weights.items()},
            "oof_score_before_calibration": float(blended_oof_score),
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
            "blend_weights_path": str(blend_weights_path.relative_to(ROOT)),
            "blended_oof_path": str(blended_oof_path.relative_to(ROOT)),
            "calibrated_oof_path": str(calibrated_oof_path.relative_to(ROOT)),
            "shrunk_oof_path": str(shrunk_oof_path.relative_to(ROOT)),
        },
    }

    summary_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log_progress(f"Finished. Submission written to {submission_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
