from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, load_modeling_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Model 42 clean rerun with custom sample weights: "
            "weight=1.0 if Humidity<0.4 else 1.4. CV=3, no post-processing."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/46_model42_env04_weight14_cv3")
    parser.add_argument("--submission-prefix", default="model42_env04_weight14_cv3")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--dirichlet-samples", type=int, default=5000)
    parser.add_argument("--dirichlet-batch-size", type=int, default=1024)
    parser.add_argument("--dirichlet-alpha-vector", nargs=2, type=float, default=[1.0, 1.0])
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


def load_clean_model42_module():
    module_path = ROOT / "scripts" / "train_best_model42_clean.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing clean model42 helper: {module_path}. "
            "Regenerate scripts/train_best_model42_clean.py first."
        )
    spec = importlib.util.spec_from_file_location("train_best_model42_clean", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_env04_weights(env: pd.Series) -> pd.Series:
    values = np.where(env.to_numpy(dtype=np.float32) >= 0.4, 1.4, 1.0).astype(np.float32)
    return pd.Series(values, index=env.index, name="humidity04_weight14")


def summarize_env04_bins(train_env: pd.Series, test_env: pd.Series) -> pd.DataFrame:
    specs = [
        ("[0.0, 0.4)", train_env < 0.4, test_env < 0.4, 1.0),
        ("[0.4, 1.0]", train_env >= 0.4, test_env >= 0.4, 1.4),
    ]
    return pd.DataFrame(
        [
            {
                "env_interval": interval,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "fixed_weight": float(weight),
            }
            for interval, train_mask, test_mask, weight in specs
        ]
    )


def main() -> None:
    args = parse_args()
    clean = load_clean_model42_module()

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = clean.maybe_subsample_bundle(
        load_modeling_data(data_dir),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    env_train = bundle.data.x_train["Humidity"].copy()
    env_test = bundle.data.x_test["Humidity"].copy()
    row_weights = compute_env04_weights(env_train)
    env_weight_bins = summarize_env04_bins(env_train, env_test)

    x_train_noenv = clean.drop_environment_columns(bundle.data.x_train)
    x_test_noenv = clean.drop_environment_columns(bundle.data.x_test)
    clean.validate_no_environment_columns(x_train_noenv, "x_train_noenv")
    clean.validate_no_environment_columns(x_test_noenv, "x_test_noenv")

    target_multiplicities = clean.get_target_multiplicities(bundle.schema, list(bundle.y_train_model.columns))
    full_target_count = len(bundle.schema.original_targets)
    alpha_vector = np.asarray(args.dirichlet_alpha_vector, dtype=np.float32)
    if len(alpha_vector) != 2 or np.any(alpha_vector <= 0):
        raise ValueError("--dirichlet-alpha-vector must contain exactly 2 positive values.")

    log_progress("Model 42 clean variant: CV=3, Humidity dropped before FE, sample_weight=1.4 if Humidity>=0.4")
    model_oofs, model_tests, full_views, fold_reports = clean.make_oof_and_test_predictions(
        x_train_noenv,
        bundle.y_train_model,
        row_weights,
        x_test_noenv,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        n_jobs=int(args.n_jobs),
        verbose=bool(args.verbose),
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

    blend_weights, blended_oof_model = clean.optimize_dirichlet_blend(
        model_oofs,
        bundle.y_train_model,
        row_weights=row_weights,
        alpha_vector=alpha_vector,
        sample_count=int(args.dirichlet_samples),
        batch_size=int(args.dirichlet_batch_size),
        random_state=int(args.random_state),
    )
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    summary_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    env_bins_path = output_dir / f"{args.submission_prefix}_{timestamp}_env_weight_bins.csv"
    blend_weights_path = output_dir / f"{args.submission_prefix}_{timestamp}_target_simplex_weights.csv"
    blended_oof_model_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_blend_modelspace.csv"
    blended_oof_full_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_blend_full.csv"
    blended_test_model_path = output_dir / f"{args.submission_prefix}_{timestamp}_test_blend_modelspace.csv"
    feature_manifest_path = output_dir / f"{args.submission_prefix}_{timestamp}_feature_manifest.json"

    submission.to_csv(submission_path, index=False)
    env_weight_bins.to_csv(env_bins_path, index=False)
    blend_weights.to_csv(blend_weights_path, index=True)
    blended_oof_model.to_csv(blended_oof_model_path, index=True)
    blended_oof_full.to_csv(blended_oof_full_path, index=True)
    blended_test_model.to_csv(blended_test_model_path, index=True)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    for model_name in clean.MODEL_ORDER:
        model_oofs[model_name].to_csv(output_dir / f"{model_name}_oof.csv", index=True)
        model_tests[model_name].to_csv(output_dir / f"{model_name}_test.csv", index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Clean CV3 model42 variant with Humidity>=0.4 weight=1.4",
        "training": {
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "metric": "WRMSE = sqrt(mean_over_rows_targets(row_weight * squared_error)) with row_weight=1.4 if Humidity>=0.4 else 1.0",
            "postprocessing": False,
        },
        "preprocessing": {
            "clipping": full_views.clipping_profile,
            "environment_removed_before_feature_engineering": True,
            "forbidden_feature_patterns": ["Humidity", "humidity_*", "humidity_times_*", "support_gap"],
        },
        "feature_views": feature_manifest,
        "sample_weighting": {
            "rule": {"Humidity < 0.4": 1.0, "Humidity >= 0.4": 1.4},
            "env_weight_bins_path": str(env_bins_path.relative_to(ROOT)),
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
                "oof_path": str((output_dir / "et_rowagg_mf06_bs_oof.csv").relative_to(ROOT)),
                "test_path": str((output_dir / "et_rowagg_mf06_bs_test.csv").relative_to(ROOT)),
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
                "oof_path": str((output_dir / "et_allpool_3_oof.csv").relative_to(ROOT)),
                "test_path": str((output_dir / "et_allpool_3_test.csv").relative_to(ROOT)),
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
            "target_weights_path": str(blend_weights_path.relative_to(ROOT)),
            "blended_oof_modelspace_path": str(blended_oof_model_path.relative_to(ROOT)),
            "blended_test_modelspace_path": str(blended_test_model_path.relative_to(ROOT)),
        },
        "artifacts": {
            "submission_path": str(submission_path.relative_to(ROOT)),
            "summary_path": str(summary_path.relative_to(ROOT)),
            "feature_manifest_path": str(feature_manifest_path.relative_to(ROOT)),
            "blended_oof_full_path": str(blended_oof_full_path.relative_to(ROOT)),
            "base_model_predictions_dir": str(output_dir.relative_to(ROOT)),
        },
        "fold_reports": fold_reports,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_progress(f"Final blended OOF weighted WRMSE={blended_oof_score:.6f}")
    log_progress(f"Submission written to {submission_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
