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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import ExtraTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from odor_competition.data import load_modeling_data  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    random_state: int


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two AdaBoost regressors with ExtraTree weak learners on "
            "raw_clean and rowagg_clean features, using a light Optuna search and CV=3."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/23_adaboost_raw_vs_rowagg_optuna_cv3",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--optuna-trials", type=int, default=3)
    parser.add_argument("--optuna-timeout-sec", type=int, default=1200)
    parser.add_argument(
        "--optuna-max-train-rows",
        type=int,
        default=60000,
        help="Max row count used only during Optuna to keep the search light. Final CV=3 still uses full data.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Passed to MultiOutputRegressor. Keep 1 in restricted environments.",
    )
    args = parser.parse_args()

    if args.cv_folds != 3:
        raise ValueError("Ce script est volontairement fixe sur une CV=3.")
    if args.optuna_trials < 1:
        raise ValueError("--optuna-trials must be >= 1.")
    if args.optuna_timeout_sec < 60:
        raise ValueError("--optuna-timeout-sec must be >= 60.")
    if args.optuna_max_train_rows is not None and args.optuna_max_train_rows < args.cv_folds * 1000:
        raise ValueError("--optuna-max-train-rows is too small for a stable CV=3 search.")

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


def build_dataset_specs(random_state: int) -> list[DatasetSpec]:
    return [
        DatasetSpec(name="raw_clean", random_state=random_state),
        DatasetSpec(name="rowagg_clean", random_state=random_state + 101),
    ]


def cast_params(params: dict) -> dict:
    casted = dict(params)
    for key in [
        "n_estimators",
        "base_max_depth",
        "base_min_samples_split",
        "base_min_samples_leaf",
        "base_max_leaf_nodes",
    ]:
        if key in casted:
            casted[key] = int(casted[key])
    if "base_max_features" in casted:
        casted["base_max_features"] = float(casted["base_max_features"])
    if "learning_rate" in casted:
        casted["learning_rate"] = float(casted["learning_rate"])
    return casted


def make_model(params: dict, *, random_state: int, n_jobs: int) -> MultiOutputRegressor:
    base_estimator = ExtraTreeRegressor(
        criterion="squared_error",
        splitter="random",
        max_depth=int(params["base_max_depth"]),
        min_samples_split=int(params["base_min_samples_split"]),
        min_samples_leaf=int(params["base_min_samples_leaf"]),
        max_features=float(params["base_max_features"]),
        max_leaf_nodes=int(params["base_max_leaf_nodes"]),
        random_state=random_state,
    )
    booster = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        loss=params["loss"],
        random_state=random_state,
    )
    return MultiOutputRegressor(booster, n_jobs=n_jobs)


def sample_for_optuna(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows is None or len(X_train) <= max_rows:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    sampled_idx = np.sort(rng.choice(len(X_train), size=max_rows, replace=False))
    return X_train.iloc[sampled_idx].copy(), y_train.iloc[sampled_idx].copy()


def evaluate_cv(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict,
    *,
    cv_folds: int,
    random_state: int,
    n_jobs: int,
    trial: optuna.Trial | None = None,
) -> dict:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_mse: list[float] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train.iloc[fit_idx]
        y_valid = y_train.iloc[valid_idx]

        model = make_model(params, random_state=random_state + fold_idx, n_jobs=n_jobs)
        model.fit(X_fit, y_fit)
        predictions = np.clip(model.predict(X_valid), 0.0, 1.0)
        fold_score = float(mean_squared_error(y_valid, predictions))
        fold_mse.append(fold_score)

        if trial is not None:
            trial.report(float(np.mean(fold_mse)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        "fold_mse": fold_mse,
        "mean_mse": float(np.mean(fold_mse)),
        "std_mse": float(np.std(fold_mse)),
    }


def optimize_dataset(
    dataset_name: str,
    X_optuna: pd.DataFrame,
    y_optuna: pd.DataFrame,
    *,
    cv_folds: int,
    n_trials: int,
    timeout_sec: int,
    random_state: int,
    n_jobs: int,
) -> tuple[dict, float]:
    log_progress(
        f"{dataset_name}: Optuna light sur {len(X_optuna)} lignes, {X_optuna.shape[1]} features, {n_trials} trial(s)"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.30, log=True),
            "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
            "base_max_depth": trial.suggest_int("base_max_depth", 2, 6),
            "base_min_samples_split": trial.suggest_int("base_min_samples_split", 2, 16, step=2),
            "base_min_samples_leaf": trial.suggest_int("base_min_samples_leaf", 1, 6),
            "base_max_features": trial.suggest_float("base_max_features", 0.5, 1.0, step=0.1),
            "base_max_leaf_nodes": trial.suggest_int("base_max_leaf_nodes", 16, 64, step=8),
        }
        result = evaluate_cv(
            X_optuna,
            y_optuna,
            params,
            cv_folds=cv_folds,
            random_state=random_state,
            n_jobs=n_jobs,
            trial=trial,
        )
        return float(result["mean_mse"])

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    best_params = cast_params(study.best_params)
    log_progress(f"{dataset_name}: meilleur MSE Optuna={study.best_value:.6f}")
    return best_params, float(study.best_value)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_progress("Chargement des donnees nettoyees")
    bundle = load_modeling_data(data_dir)
    X_train_raw_clean = bundle.x_train_raw.astype(np.float32)
    y_train_model = bundle.y_train_model.astype(np.float32)

    log_progress("Construction de la vue rowagg_clean")
    X_train_rowagg = build_rowwise_aggregated_features(X_train_raw_clean).astype(np.float32)

    feature_sets = {
        "raw_clean": X_train_raw_clean,
        "rowagg_clean": X_train_rowagg,
    }

    dataset_reports: dict[str, dict] = {}
    dataset_specs = build_dataset_specs(args.random_state)

    for spec in dataset_specs:
        X_full = feature_sets[spec.name]
        X_optuna, y_optuna = sample_for_optuna(
            X_full,
            y_train_model,
            max_rows=args.optuna_max_train_rows,
            random_state=spec.random_state,
        )

        best_params, optuna_best_mse = optimize_dataset(
            spec.name,
            X_optuna,
            y_optuna,
            cv_folds=args.cv_folds,
            n_trials=args.optuna_trials,
            timeout_sec=args.optuna_timeout_sec,
            random_state=spec.random_state,
            n_jobs=args.n_jobs,
        )

        log_progress(f"{spec.name}: reevaluation finale sur tout le train en CV=3")
        final_cv = evaluate_cv(
            X_full,
            y_train_model,
            best_params,
            cv_folds=args.cv_folds,
            random_state=spec.random_state,
            n_jobs=args.n_jobs,
        )

        dataset_reports[spec.name] = {
            "dataset_name": spec.name,
            "random_state": int(spec.random_state),
            "feature_count": int(X_full.shape[1]),
            "full_train_rows": int(len(X_full)),
            "optuna_rows": int(len(X_optuna)),
            "optuna_trials": int(args.optuna_trials),
            "optuna_timeout_sec": int(args.optuna_timeout_sec),
            "optuna_best_mean_mse": float(optuna_best_mse),
            "best_params": best_params,
            "final_cv": final_cv,
        }
        log_progress(
            f"{spec.name}: MSE final CV=3 = {final_cv['mean_mse']:.6f} "
            f"(folds={', '.join(f'{score:.6f}' for score in final_cv['fold_mse'])})"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "generated_at_utc": timestamp,
        "model": "AdaBoostRegressor(MultiOutput, ExtraTree weak learners) on raw_clean vs rowagg_clean",
        "data_dir": str(data_dir),
        "notes": {
            "rowagg_source": "Same row-wise aggregated transformation as train_et_rowagg_dirichlet_shrinkage.py",
            "target_space": "y_train_model from load_modeling_data",
            "metric": "sklearn.metrics.mean_squared_error averaged over all modeled outputs",
            "final_evaluation": "Best parameters reevaluated with CV=3 on the full training set",
        },
        "training": {
            "cv_folds": int(args.cv_folds),
            "optuna_trials": int(args.optuna_trials),
            "optuna_timeout_sec": int(args.optuna_timeout_sec),
            "optuna_max_train_rows": None if args.optuna_max_train_rows is None else int(args.optuna_max_train_rows),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
        },
        "datasets": dataset_reports,
    }

    summary_path = output_dir / f"adaboost_raw_vs_rowagg_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    log_progress(f"Resume ecrit dans {summary_path}")


if __name__ == "__main__":
    main()
