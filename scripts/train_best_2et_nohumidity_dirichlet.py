from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, load_modeling_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Best reproducible model: CV3 ExtraTrees rowagg + ExtraTrees allpool, "
            "Humidity removed from features, model50 piecewise sample weights, "
            "and target-wise Dirichlet blend."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391")
    parser.add_argument("--submission-prefix", default="best_2et_nohumidity_dirichlet")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--dirichlet-samples", type=int, default=5000)
    parser.add_argument("--dirichlet-batch-size", type=int, default=1024)
    parser.add_argument("--dirichlet-alpha-vector", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument(
        "--weight-preset",
        choices=["model49", "model50", "model50_low02_115"],
        default="model50",
        help=(
            "model49: previous piecewise weights. "
            "model50: same except 0.39-0.50=1.35, 0.50-0.68=1.0, 0.80-1.00=1.25. "
            "model50_low02_115: model50 with 0.00-0.20=1.15."
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cv_folds != 3:
        raise ValueError("Cette pipeline est definie pour CV=3.")
    return args


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_file_stem(raw_value: str) -> str:
    stem = Path(str(raw_value)).name.strip()
    for char in '<>:"/\\|?*':
        stem = stem.replace(char, "_")
    return stem or "blender_et3_rowaggbs_piecewise_bins_cv3"


def load_best_model_core():
    module_path = ROOT / "scripts" / "best_2et_nohumidity_core.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing best-model helper: {module_path}. "
            "Regenerate scripts/best_2et_nohumidity_core.py first."
        )
    spec = importlib.util.spec_from_file_location("best_2et_nohumidity_core", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_piecewise_specs(preset: str):
    if preset == "model50":
        return [
            ("[0.00, 0.20)", lambda s: s < 0.2, 1.0),
            ("[0.20, 0.39)", lambda s: (s >= 0.2) & (s < 0.39), 1.1),
            ("[0.39, 0.50)", lambda s: (s >= 0.39) & (s < 0.50), 1.35),
            ("[0.50, 0.68)", lambda s: (s >= 0.50) & (s < 0.68), 1.0),
            ("[0.68, 0.80)", lambda s: (s >= 0.68) & (s < 0.8), 1.25),
            ("[0.80, 1.00]", lambda s: s >= 0.8, 1.25),
        ]
    if preset == "model50_low02_115":
        return [
            ("[0.00, 0.20)", lambda s: s < 0.2, 1.15),
            ("[0.20, 0.39)", lambda s: (s >= 0.2) & (s < 0.39), 1.1),
            ("[0.39, 0.50)", lambda s: (s >= 0.39) & (s < 0.50), 1.35),
            ("[0.50, 0.68)", lambda s: (s >= 0.50) & (s < 0.68), 1.0),
            ("[0.68, 0.80)", lambda s: (s >= 0.68) & (s < 0.8), 1.25),
            ("[0.80, 1.00]", lambda s: s >= 0.8, 1.25),
        ]
    if preset == "model49":
        return [
            ("[0.00, 0.20)", lambda s: s < 0.2, 1.0),
            ("[0.20, 0.39)", lambda s: (s >= 0.2) & (s < 0.39), 1.1),
            ("[0.39, 0.49)", lambda s: (s >= 0.39) & (s < 0.49), 1.1),
            ("[0.49, 0.52)", lambda s: (s >= 0.49) & (s < 0.52), 1.25),
            ("[0.52, 0.68)", lambda s: (s >= 0.52) & (s < 0.68), 1.0),
            ("[0.68, 0.80)", lambda s: (s >= 0.68) & (s < 0.8), 1.25),
            ("[0.80, 1.00]", lambda s: s >= 0.8, 1.1),
        ]
    raise ValueError(f"Unknown piecewise preset: {preset}")


def compute_piecewise_env_weights(env: pd.Series, *, preset: str) -> pd.Series:
    values = np.full(len(env), 1.1, dtype=np.float32)
    for _, mask_fn, weight in get_piecewise_specs(preset):
        values[mask_fn(env).to_numpy()] = float(weight)
    return pd.Series(values, index=env.index, name=f"piecewise_humidity_weight_{preset}")


def summarize_piecewise_bins(train_env: pd.Series, test_env: pd.Series, *, preset: str) -> pd.DataFrame:
    specs = get_piecewise_specs(preset)
    rows = []
    for interval, mask_fn, weight in specs:
        train_mask = mask_fn(train_env)
        test_mask = mask_fn(test_env)
        rows.append(
            {
                "env_interval": interval,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "fixed_weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def make_oof_predictions_only(
    clean,
    x_train_noenv: pd.DataFrame,
    y_train: pd.DataFrame,
    weights_train: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    tail_quantile: float,
    ratio_eps: float,
    n_jobs: int,
    verbose: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, int]]]]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_predictions = {
        model_name: pd.DataFrame(index=y_train.index, columns=y_train.columns, dtype=np.float32)
        for model_name in clean.MODEL_ORDER
    }
    fold_reports = {model_name: [] for model_name in clean.MODEL_ORDER}

    for fold_number, (fit_idx, valid_idx) in enumerate(cv.split(x_train_noenv), start=1):
        fit_index = x_train_noenv.index[fit_idx]
        valid_index = x_train_noenv.index[valid_idx]
        views = clean.build_feature_views(
            x_train_noenv.loc[fit_index],
            x_train_noenv.loc[valid_index],
            tail_quantile=tail_quantile,
            ratio_eps=ratio_eps,
        )
        fold_predictions = clean.fit_models_and_predict(
            views,
            y_train.loc[fit_index],
            weights_train.loc[fit_index],
            n_jobs=n_jobs,
        )
        for model_name, predictions in fold_predictions.items():
            oof_predictions[model_name].loc[valid_index] = predictions.to_numpy(dtype=np.float32)
            fold_reports[model_name].append(
                {"fold": fold_number, "fit_rows": int(len(fit_index)), "valid_rows": int(len(valid_index))}
            )
        if verbose:
            log_progress(f"OOF fold {fold_number}/{cv_folds} ready")

    for model_name, predictions in oof_predictions.items():
        if predictions.isna().any().any():
            raise RuntimeError(f"Missing OOF predictions for {model_name}.")
        oof_predictions[model_name] = predictions.astype(np.float32)
    return oof_predictions, fold_reports


def make_full_test_predictions(
    clean,
    x_train_noenv: pd.DataFrame,
    y_train: pd.DataFrame,
    weights_train: pd.Series,
    x_test_noenv: pd.DataFrame,
    *,
    tail_quantile: float,
    ratio_eps: float,
    n_jobs: int,
):
    full_views = clean.build_feature_views(
        x_train_noenv,
        x_test_noenv,
        tail_quantile=tail_quantile,
        ratio_eps=ratio_eps,
    )
    model_tests = clean.fit_models_and_predict(
        full_views,
        y_train,
        weights_train,
        n_jobs=n_jobs,
    )
    return model_tests, full_views


def main() -> None:
    args = parse_args()
    clean = load_best_model_core()

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = safe_file_stem(args.submission_prefix)

    bundle = clean.maybe_subsample_bundle(
        load_modeling_data(data_dir),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    env_train = bundle.data.x_train["Humidity"].copy()
    env_test = bundle.data.x_test["Humidity"].copy()
    row_weights = compute_piecewise_env_weights(env_train, preset=str(args.weight_preset))
    env_weight_bins = summarize_piecewise_bins(env_train, env_test, preset=str(args.weight_preset))

    x_train_noenv = clean.drop_environment_columns(bundle.data.x_train)
    x_test_noenv = clean.drop_environment_columns(bundle.data.x_test)
    clean.validate_no_environment_columns(x_train_noenv, "x_train_noenv")
    clean.validate_no_environment_columns(x_test_noenv, "x_test_noenv")

    target_multiplicities = clean.get_target_multiplicities(bundle.schema, list(bundle.y_train_model.columns))
    full_target_count = len(bundle.schema.original_targets)
    alpha_vector = np.asarray(args.dirichlet_alpha_vector, dtype=np.float32)
    if len(alpha_vector) != 2 or np.any(alpha_vector <= 0):
        raise ValueError("--dirichlet-alpha-vector must contain exactly 2 positive values.")

    log_progress(
        f"Piecewise bins model ({args.weight_preset}): CV=3, Humidity dropped before FE, "
        "fixed sample weights by Humidity bins"
    )
    log_progress(
        "Weights: "
        + ", ".join(f"{interval}={weight}" for interval, _, weight in get_piecewise_specs(str(args.weight_preset)))
    )
    model_oofs, fold_reports = make_oof_predictions_only(
        clean,
        x_train_noenv,
        bundle.y_train_model,
        row_weights,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        n_jobs=int(args.n_jobs),
        verbose=bool(args.verbose),
    )
    for model_name in clean.MODEL_ORDER:
        checkpoint_path = output_dir / f"{model_name}_oof_checkpoint_{timestamp}.csv"
        model_oofs[model_name].to_csv(checkpoint_path, index=True)
    env_weight_bins.to_csv(output_dir / f"{prefix}_{timestamp}_env_weight_bins_checkpoint.csv", index=False)
    pd.DataFrame({"Humidity": env_train, "sample_weight": row_weights}, index=bundle.y_train_model.index).to_csv(
        output_dir / f"{prefix}_{timestamp}_sample_weights_checkpoint.csv",
        index=True,
    )
    log_progress(f"OOF checkpoints written to {output_dir}")

    base_scores = {
        model_name: clean.weighted_wrmse(
            bundle.y_train_model,
            model_oofs[model_name],
            row_weights=row_weights,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        )
        for model_name in clean.MODEL_ORDER
    }
    for model_name, score in base_scores.items():
        log_progress(f"{model_name}: CV3 weighted WRMSE={score:.6f}")

    log_progress("Optimizing target-wise Dirichlet/simplex blend on OOF predictions")
    blend_weights, blended_oof_model = clean.optimize_dirichlet_blend(
        model_oofs,
        bundle.y_train_model,
        row_weights=row_weights,
        alpha_vector=alpha_vector,
        sample_count=int(args.dirichlet_samples),
        batch_size=int(args.dirichlet_batch_size),
        random_state=int(args.random_state),
    )
    log_progress("Refitting both base models on full train for test predictions")
    model_tests, full_views = make_full_test_predictions(
        clean,
        x_train_noenv,
        bundle.y_train_model,
        row_weights,
        x_test_noenv,
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        n_jobs=int(args.n_jobs),
    )
    feature_manifest = {
        "raw_clean_count": int(full_views.raw_fit.shape[1]),
        "rowagg_clean_count": int(full_views.rowagg_fit.shape[1]),
        "allpool_noenv_count": int(full_views.allpool_fit.shape[1]),
        "rowagg_feature_names": list(full_views.rowagg_fit.columns),
        "allpool_first_40_feature_names": list(full_views.allpool_fit.columns[:40]),
    }
    if (
        feature_manifest["raw_clean_count"] != 12
        or feature_manifest["rowagg_clean_count"] != 26
        or feature_manifest["allpool_noenv_count"] != 197
    ):
        raise ValueError(f"Unexpected feature counts: {feature_manifest}")

    blended_test_model = clean.apply_targetwise_blend(model_tests, blend_weights)
    blended_oof_score = clean.weighted_wrmse(
        bundle.y_train_model,
        blended_oof_model,
        row_weights=row_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )

    blended_oof_full = bundle.schema.expand_predictions(blended_oof_model)
    blended_test_full = bundle.schema.expand_predictions(blended_test_model)
    submission = build_submission_frame(bundle.data.x_test["ID"], blended_test_full)

    submission_path = output_dir / f"{prefix}_{timestamp}.csv"
    summary_path = output_dir / f"{prefix}_{timestamp}.json"
    env_bins_path = output_dir / f"{prefix}_{timestamp}_env_weight_bins.csv"
    blend_weights_path = output_dir / f"{prefix}_{timestamp}_target_simplex_weights.csv"
    blended_oof_model_path = output_dir / f"{prefix}_{timestamp}_oof_blend_modelspace.csv"
    blended_oof_full_path = output_dir / f"{prefix}_{timestamp}_oof_blend_full.csv"
    blended_test_model_path = output_dir / f"{prefix}_{timestamp}_test_blend_modelspace.csv"
    feature_manifest_path = output_dir / f"{prefix}_{timestamp}_feature_manifest.json"
    sample_weights_path = output_dir / f"{prefix}_{timestamp}_sample_weights.csv"

    for path in [
        submission_path,
        summary_path,
        env_bins_path,
        blend_weights_path,
        blended_oof_model_path,
        blended_oof_full_path,
        blended_test_model_path,
        feature_manifest_path,
        sample_weights_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)
    env_weight_bins.to_csv(env_bins_path, index=False)
    blend_weights.to_csv(blend_weights_path, index=True)
    blended_oof_model.to_csv(blended_oof_model_path, index=True)
    blended_oof_full.to_csv(blended_oof_full_path, index=True)
    blended_test_model.to_csv(blended_test_model_path, index=True)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    pd.DataFrame({"Humidity": env_train, "sample_weight": row_weights}, index=bundle.y_train_model.index).to_csv(sample_weights_path, index=True)

    for model_name in clean.MODEL_ORDER:
        model_oofs[model_name].to_csv(output_dir / f"{model_name}_oof.csv", index=True)
        model_tests[model_name].to_csv(output_dir / f"{model_name}_test.csv", index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": f"Best 2ET nohumidity Dirichlet blend with fixed piecewise Humidity sample weights ({args.weight_preset})",
        "training": {
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "metric": "WRMSE = sqrt(mean_over_rows_targets(row_weight * squared_error)) with fixed piecewise Humidity sample weights",
            "postprocessing": False,
        },
        "preprocessing": {
            "clipping": full_views.clipping_profile,
            "environment_removed_before_feature_engineering": True,
            "forbidden_feature_patterns": ["Humidity", "humidity_*", "humidity_times_*", "support_gap"],
        },
        "feature_views": feature_manifest,
        "sample_weighting": {
            "preset": str(args.weight_preset),
            "rule_order": [
                {interval: float(weight)}
                for interval, _, weight in get_piecewise_specs(str(args.weight_preset))
            ],
            "env_weight_bins_path": display_path(env_bins_path),
            "sample_weights_path": display_path(sample_weights_path),
        },
        "base_models": {
            "et_rowagg_mf06_bs": {
                "dataset": "rowagg_clean",
                "params": {
                    "n_estimators": 500,
                    "max_features": 0.6,
                    "bootstrap": True,
                    "random_state": 42,
                    "max_depth": 28,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_samples": 0.8051184329566651,
                },
                "cv3_weighted_wrmse": base_scores["et_rowagg_mf06_bs"],
                "oof_path": display_path(output_dir / "et_rowagg_mf06_bs_oof.csv"),
                "test_path": display_path(output_dir / "et_rowagg_mf06_bs_test.csv"),
            },
            "et_allpool_3": {
                "dataset": "allpool_noenv",
                "params": {
                    "n_estimators": 300,
                    "max_depth": 17,
                    "min_samples_split": 13,
                    "min_samples_leaf": 5,
                    "max_features": 0.25,
                    "bootstrap": False,
                    "random_state": 83,
                    "n_jobs": int(args.n_jobs),
                },
                "cv3_weighted_wrmse": base_scores["et_allpool_3"],
                "oof_path": display_path(output_dir / "et_allpool_3_oof.csv"),
                "test_path": display_path(output_dir / "et_allpool_3_test.csv"),
            },
        },
        "simplex": {
            "alpha_vector": [float(value) for value in alpha_vector],
            "dirichlet_samples": int(args.dirichlet_samples),
            "dirichlet_batch_size": int(args.dirichlet_batch_size),
            "oof_weighted_wrmse": float(blended_oof_score),
            "winner_counts": {
                model_name: int(count)
                for model_name, count in blend_weights.idxmax(axis=1).value_counts().to_dict().items()
            },
            "target_weights_path": display_path(blend_weights_path),
            "blended_oof_modelspace_path": display_path(blended_oof_model_path),
            "blended_oof_full_path": display_path(blended_oof_full_path),
            "blended_test_modelspace_path": display_path(blended_test_model_path),
        },
        "artifacts": {
            "submission_path": display_path(submission_path),
            "summary_path": display_path(summary_path),
            "feature_manifest_path": display_path(feature_manifest_path),
            "blended_oof_full_path": display_path(blended_oof_full_path),
            "base_model_predictions_dir": display_path(output_dir),
        },
        "fold_reports": fold_reports,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_progress(f"Final blended OOF weighted WRMSE={blended_oof_score:.6f}")
    log_progress(f"Submission written to {submission_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
