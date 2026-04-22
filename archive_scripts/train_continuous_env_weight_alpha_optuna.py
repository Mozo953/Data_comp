from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaz_competition.data import build_submission_frame, load_modeling_data  # noqa: E402


EPS = 1e-12
W_MIN = 1.0
W_MAX = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous Humidity-weighted rowagg/allpool Dirichlet pipeline with OOF baseline "
            "difficulty and alpha-only Optuna weighting."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/48_continuous_env_weight_alpha_optuna")
    parser.add_argument("--submission-prefix", default="continuous_env_weight_alpha_optuna")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--env-quantile-bins", type=int, default=20)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--alpha-low", type=float, default=0.0)
    parser.add_argument("--alpha-high", type=float, default=3.0)
    parser.add_argument("--optuna-timeout-sec", type=int, default=180)
    parser.add_argument("--dirichlet-samples", type=int, default=5000)
    parser.add_argument("--dirichlet-batch-size", type=int, default=1024)
    parser.add_argument("--dirichlet-alpha-vector", nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cv_folds != 3:
        raise ValueError("Cette pipeline est definie pour CV=3.")
    if args.env_quantile_bins < 2:
        raise ValueError("--env-quantile-bins must be >= 2.")
    if args.alpha_low < 0 or args.alpha_high < args.alpha_low:
        raise ValueError("Alpha search range must satisfy 0 <= alpha_low <= alpha_high.")
    return args


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def verbose_log(enabled: bool, message: str) -> None:
    if enabled:
        log_progress(message)


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def safe_file_stem(raw_value: str) -> str:
    stem = Path(str(raw_value)).name.strip()
    for char in '<>:"/\\|?*':
        stem = stem.replace(char, "_")
    return stem or "continuous_env_weight_alpha_optuna"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_clean_model42_module():
    module_path = ROOT / "scripts" / "best_2et_nohumidity_core.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing clean model42 helper: {module_path}. "
            "Regenerate scripts/best_2et_nohumidity_core.py first."
        )
    spec = importlib.util.spec_from_file_location("best_2et_nohumidity_core", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0.0 or len(values) < 3:
        return values
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / sigma))
    kernel /= kernel.sum()
    padded = np.pad(values, pad_width=radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def sample_rmse_model_space(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    *,
    target_multiplicities: np.ndarray,
    full_target_count: int,
) -> pd.Series:
    true_values = y_true.to_numpy(dtype=np.float32)
    pred_values = np.clip(y_pred.to_numpy(dtype=np.float32), 0.0, 1.0)
    target_weight_values = target_multiplicities.reshape(1, -1)
    sample_mse = np.sum(target_weight_values * np.square(pred_values - true_values), axis=1) / float(full_target_count)
    return pd.Series(np.sqrt(sample_mse, dtype=np.float32), index=y_true.index, name="sample_rmse")


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
    label: str,
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
        verbose_log(verbose, f"{label}: OOF fold {fold_number}/{cv_folds} ready")

    for model_name, predictions in oof_predictions.items():
        if predictions.isna().any().any():
            raise RuntimeError(f"Missing OOF predictions for {model_name} in {label}.")
        oof_predictions[model_name] = predictions.astype(np.float32)
    return oof_predictions, fold_reports


def build_env_bin_frame(train_env: pd.Series, test_env: pd.Series, sample_error: pd.Series, *, bin_count: int) -> pd.DataFrame:
    train_values = train_env.to_numpy(dtype=np.float64)
    test_values = test_env.to_numpy(dtype=np.float64)
    if np.unique(train_values).size < 2:
        raise ValueError("Humidity has fewer than 2 unique train values; quantile binning is impossible.")

    _, edges = pd.qcut(train_values, q=bin_count, retbins=True, duplicates="drop")
    edges = np.unique(edges.astype(np.float64))
    if len(edges) < 3:
        edges = np.linspace(float(np.nanmin(train_values)), float(np.nanmax(train_values)), min(bin_count, 2) + 1)
    cut_edges = edges.copy()
    cut_edges[0] = -np.inf
    cut_edges[-1] = np.inf

    train_codes = pd.cut(train_values, bins=cut_edges, labels=False, include_lowest=True, right=True)
    test_codes = pd.cut(test_values, bins=cut_edges, labels=False, include_lowest=True, right=True)
    train_codes = np.asarray(train_codes, dtype=np.int64)
    test_codes = np.asarray(test_codes, dtype=np.int64)
    effective_bins = len(edges) - 1
    rows: list[dict[str, float | int]] = []
    errors = sample_error.loc[train_env.index].to_numpy(dtype=np.float64)

    for bin_id in range(effective_bins):
        train_mask = train_codes == bin_id
        test_mask = test_codes == bin_id
        left = float(edges[bin_id])
        right = float(edges[bin_id + 1])
        width = max(right - left, EPS)
        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())
        train_freq = train_count / max(len(train_values), 1)
        test_freq = test_count / max(len(test_values), 1)
        train_density = train_freq / width
        test_density = test_freq / width
        loss_raw = float(np.mean(errors[train_mask])) if train_count else float("nan")
        rows.append(
            {
                "bin_id": bin_id,
                "env_left": left,
                "env_right": right,
                "env_mid": (left + right) / 2.0,
                "train_count": train_count,
                "test_count": test_count,
                "train_freq": train_freq,
                "test_freq": test_freq,
                "train_density": train_density,
                "test_density": test_density,
                "loss_raw": loss_raw,
            }
        )

    frame = pd.DataFrame(rows)
    if frame["loss_raw"].isna().any():
        frame["loss_raw"] = frame["loss_raw"].interpolate(limit_direction="both")
    frame.attrs["cut_edges"] = cut_edges
    return frame


def build_weight_curves_and_samples(
    train_env: pd.Series,
    test_env: pd.Series,
    sample_error: pd.Series,
    *,
    alpha: float,
    bin_count: int,
    smooth_sigma: float,
) -> tuple[pd.DataFrame, pd.Series]:
    curves = build_env_bin_frame(train_env, test_env, sample_error, bin_count=bin_count)
    loss_smooth = gaussian_smooth(curves["loss_raw"].to_numpy(dtype=np.float64), smooth_sigma)
    loss_min = float(np.min(loss_smooth))
    loss_max = float(np.max(loss_smooth))
    difficulty = (loss_smooth - loss_min) / (loss_max - loss_min + EPS)

    ratio_raw = np.maximum(
        0.0,
        curves["test_density"].to_numpy(dtype=np.float64) / (curves["train_density"].to_numpy(dtype=np.float64) + EPS) - 1.0,
    )
    ratio_shift = gaussian_smooth(ratio_raw, smooth_sigma)
    ratio_shift = np.maximum(0.0, ratio_shift)
    weights_by_bin = np.clip(W_MIN + float(alpha) * difficulty * ratio_shift, W_MIN, W_MAX)

    curves["loss_smooth"] = loss_smooth
    curves["difficulty_D"] = difficulty
    curves["ratio_raw"] = ratio_raw
    curves["ratio_R"] = ratio_shift
    curves["alpha"] = float(alpha)
    curves["weight"] = weights_by_bin

    cut_edges = curves.attrs["cut_edges"]
    train_codes = pd.cut(
        train_env.to_numpy(dtype=np.float64),
        bins=cut_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    train_codes = np.asarray(train_codes, dtype=np.int64)
    sample_weights = pd.Series(weights_by_bin[train_codes], index=train_env.index, name="continuous_env_weight").astype(np.float32)
    return curves, sample_weights


def score_oofs(
    clean,
    model_oofs: dict[str, pd.DataFrame],
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    target_multiplicities: np.ndarray,
    full_target_count: int,
) -> dict[str, float]:
    return {
        model_name: clean.weighted_wrmse(
            y_train,
            model_oofs[model_name],
            row_weights=row_weights,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        )
        for model_name in clean.MODEL_ORDER
    }


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
    test_predictions = clean.fit_models_and_predict(
        full_views,
        y_train,
        weights_train,
        n_jobs=n_jobs,
    )
    return test_predictions, full_views


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
    y_train = bundle.y_train_model
    env_train = bundle.data.x_train["Humidity"].copy()
    env_test = bundle.data.x_test["Humidity"].copy()
    x_train_noenv = clean.drop_environment_columns(bundle.data.x_train)
    x_test_noenv = clean.drop_environment_columns(bundle.data.x_test)
    clean.validate_no_environment_columns(x_train_noenv, "x_train_noenv")
    clean.validate_no_environment_columns(x_test_noenv, "x_test_noenv")

    alpha_vector = np.asarray(args.dirichlet_alpha_vector, dtype=np.float32)
    if len(alpha_vector) != 2 or np.any(alpha_vector <= 0):
        raise ValueError("--dirichlet-alpha-vector must contain exactly 2 positive values.")

    target_multiplicities = clean.get_target_multiplicities(bundle.schema, list(y_train.columns))
    full_target_count = len(bundle.schema.original_targets)
    baseline_unit_weights = pd.Series(np.ones(len(y_train), dtype=np.float32), index=y_train.index, name="baseline_unit_weight")

    log_progress("Continuous alpha pipeline: baseline OOF without sample_weight")
    baseline_oofs, baseline_fold_reports = make_oof_predictions_only(
        clean,
        x_train_noenv,
        y_train,
        baseline_unit_weights,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        n_jobs=int(args.n_jobs),
        verbose=bool(args.verbose),
        label="baseline",
    )
    baseline_scores = score_oofs(
        clean,
        baseline_oofs,
        y_train,
        baseline_unit_weights,
        target_multiplicities,
        full_target_count,
    )
    for model_name, score in baseline_scores.items():
        log_progress(f"{model_name}: baseline OOF RMSE={score:.6f}")

    baseline_errors = [
        sample_rmse_model_space(
            y_train,
            baseline_oofs[model_name],
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        )
        for model_name in clean.MODEL_ORDER
    ]
    baseline_sample_error = pd.concat(baseline_errors, axis=1).mean(axis=1).astype(np.float32)
    baseline_sample_error.name = "baseline_sample_error"

    trial_cache: dict[
        int,
        tuple[float, float, pd.DataFrame, pd.Series, dict[str, pd.DataFrame], dict[str, list[dict[str, int]]]]
    ] = {}

    def objective(trial: optuna.Trial) -> float:
        alpha = float(trial.suggest_float("alpha", float(args.alpha_low), float(args.alpha_high)))
        curves, sample_weights = build_weight_curves_and_samples(
            env_train,
            env_test,
            baseline_sample_error,
            alpha=alpha,
            bin_count=int(args.env_quantile_bins),
            smooth_sigma=float(args.smooth_sigma),
        )
        weighted_oofs, weighted_reports = make_oof_predictions_only(
            clean,
            x_train_noenv,
            y_train,
            sample_weights,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            tail_quantile=float(args.tail_quantile),
            ratio_eps=float(args.ratio_eps),
            n_jobs=int(args.n_jobs),
            verbose=False,
            label=f"alpha={alpha:.6f}",
        )
        blend_weights, blended_oof = clean.optimize_dirichlet_blend(
            weighted_oofs,
            y_train,
            row_weights=sample_weights,
            alpha_vector=alpha_vector,
            sample_count=int(args.dirichlet_samples),
            batch_size=int(args.dirichlet_batch_size),
            random_state=int(args.random_state),
        )
        score = clean.weighted_wrmse(
            y_train,
            blended_oof,
            row_weights=sample_weights,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        )
        trial_cache[int(trial.number)] = (float(score), alpha, curves, sample_weights, weighted_oofs, weighted_reports)
        log_progress(f"Optuna trial {trial.number}: alpha={alpha:.6f}, weighted blend OOF WRMSE={score:.6f}")
        return float(score)

    log_progress(
        f"Optimizing alpha only with Optuna TPE for max {int(args.optuna_timeout_sec)} sec "
        f"in [{float(args.alpha_low)}, {float(args.alpha_high)}]"
    )
    sampler = optuna.samplers.TPESampler(seed=int(args.random_state))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, timeout=int(args.optuna_timeout_sec), show_progress_bar=False)
    if len(study.trials) == 0 or study.best_trial is None:
        raise RuntimeError("Optuna did not complete any alpha trial.")

    alpha_best = float(study.best_params["alpha"])
    log_progress(f"Best alpha={alpha_best:.6f}, best validation WRMSE={float(study.best_value):.6f}")

    cached_best = trial_cache.get(int(study.best_trial.number))
    if cached_best is not None:
        _, _, final_curves, final_sample_weights, weighted_oofs, weighted_fold_reports = cached_best
        log_progress("Reusing weighted OOF predictions from the best Optuna trial")
    else:
        final_curves, final_sample_weights = build_weight_curves_and_samples(
            env_train,
            env_test,
            baseline_sample_error,
            alpha=alpha_best,
            bin_count=int(args.env_quantile_bins),
            smooth_sigma=float(args.smooth_sigma),
        )
        log_progress("Final weighted CV=3 OOF predictions")
        weighted_oofs, weighted_fold_reports = make_oof_predictions_only(
            clean,
            x_train_noenv,
            y_train,
            final_sample_weights,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            tail_quantile=float(args.tail_quantile),
            ratio_eps=float(args.ratio_eps),
            n_jobs=int(args.n_jobs),
            verbose=bool(args.verbose),
            label="weighted-final",
        )

    log_progress("Final full-train fit for test predictions")
    weighted_tests, full_views = make_full_test_predictions(
        clean,
        x_train_noenv,
        y_train,
        final_sample_weights,
        x_test_noenv,
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        n_jobs=int(args.n_jobs),
    )
    weighted_scores = score_oofs(
        clean,
        weighted_oofs,
        y_train,
        final_sample_weights,
        target_multiplicities,
        full_target_count,
    )
    for model_name, score in weighted_scores.items():
        log_progress(f"{model_name}: weighted OOF WRMSE={score:.6f}")

    blend_weights, blended_oof_model = clean.optimize_dirichlet_blend(
        weighted_oofs,
        y_train,
        row_weights=final_sample_weights,
        alpha_vector=alpha_vector,
        sample_count=int(args.dirichlet_samples),
        batch_size=int(args.dirichlet_batch_size),
        random_state=int(args.random_state),
    )
    blended_test_model = clean.apply_targetwise_blend(weighted_tests, blend_weights)
    blended_oof_score = clean.weighted_wrmse(
        y_train,
        blended_oof_model,
        row_weights=final_sample_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )

    blended_oof_full = bundle.schema.expand_predictions(blended_oof_model)
    blended_test_full = bundle.schema.expand_predictions(blended_test_model)
    submission = build_submission_frame(bundle.data.x_test["ID"], blended_test_full)

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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_prefix = safe_file_stem(str(args.submission_prefix))
    paths = {
        "submission": output_dir / f"{submission_prefix}_{timestamp}.csv",
        "summary": output_dir / f"{submission_prefix}_{timestamp}.json",
        "loss_curve": output_dir / f"loss_curve_{timestamp}.csv",
        "ratio_curve": output_dir / f"ratio_curve_{timestamp}.csv",
        "weight_curve": output_dir / f"weight_curve_{timestamp}.csv",
        "sample_weights": output_dir / f"sample_weights_{timestamp}.csv",
        "blend_model": output_dir / f"blend_model_{timestamp}.csv",
        "blend_full": output_dir / f"blend_full_{timestamp}.csv",
        "blend_test_model": output_dir / f"blend_test_model_{timestamp}.csv",
        "target_simplex": output_dir / f"simplex_{timestamp}.csv",
        "feature_manifest": output_dir / f"feature_manifest_{timestamp}.json",
        "optuna_trials": output_dir / f"optuna_trials_{timestamp}.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(paths["submission"], index=False)
    final_curves[
        [
            "bin_id",
            "env_left",
            "env_right",
            "env_mid",
            "train_count",
            "test_count",
            "loss_raw",
            "loss_smooth",
            "difficulty_D",
        ]
    ].to_csv(paths["loss_curve"], index=False)
    final_curves[
        [
            "bin_id",
            "env_left",
            "env_right",
            "env_mid",
            "train_count",
            "test_count",
            "train_density",
            "test_density",
            "ratio_raw",
            "ratio_R",
        ]
    ].to_csv(paths["ratio_curve"], index=False)
    final_curves.to_csv(paths["weight_curve"], index=False)
    pd.DataFrame(
        {
            "Humidity": env_train.to_numpy(dtype=np.float32),
            "baseline_sample_error": baseline_sample_error.to_numpy(dtype=np.float32),
            "sample_weight": final_sample_weights.to_numpy(dtype=np.float32),
        },
        index=y_train.index,
    ).to_csv(paths["sample_weights"], index=True)
    blend_weights.to_csv(paths["target_simplex"], index=True)
    blended_oof_model.to_csv(paths["blend_model"], index=True)
    blended_oof_full.to_csv(paths["blend_full"], index=True)
    blended_test_model.to_csv(paths["blend_test_model"], index=True)
    paths["feature_manifest"].write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    for model_name in clean.MODEL_ORDER:
        (output_dir / f"{model_name}_baseline_oof.csv").parent.mkdir(parents=True, exist_ok=True)
        baseline_oofs[model_name].to_csv(output_dir / f"{model_name}_baseline_oof.csv", index=True)
        weighted_oofs[model_name].to_csv(output_dir / f"{model_name}_weighted_oof.csv", index=True)
        weighted_tests[model_name].to_csv(output_dir / f"{model_name}_weighted_test.csv", index=True)

    study.trials_dataframe().to_csv(paths["optuna_trials"], index=False)

    relative_paths = {name: display_path(path) for name, path in paths.items()}
    summary = {
        "generated_at_utc": timestamp,
        "model": "Continuous Humidity-weighted rowagg/allpool Dirichlet pipeline with alpha-optimized D*R weights",
        "training": {
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "metric": "WRMSE = sqrt(1/N * sum_j w_j * mean_targets((y_j - yhat_j)^2)) in model target space expanded by target multiplicity",
            "postprocessing": False,
        },
        "preprocessing": {
            "clipping": full_views.clipping_profile,
            "environment_removed_before_feature_engineering": True,
            "forbidden_feature_patterns": ["Humidity", "humidity_*", "humidity_times_*", "support_gap"],
        },
        "feature_views": feature_manifest,
        "sample_weighting": {
            "formula": "w(b)=clip(w_min + alpha * D(b) * R(b), w_min, w_max)",
            "sample_formula": "w_j = w(b(Humidity_j))",
            "difficulty_formula": "D(b)=(ell(b)-min_b ell(b))/(max_b ell(b)-min_b ell(b)+eps), ell(b)=mean OOF baseline sample RMSE in bin b",
            "ratio_formula": "R(b)=max(0, p_test(b)/(p_train(b)+eps)-1)",
            "sample_error_formula": "e_j=sqrt((1/T)*sum_i(y_ji-yhat_ji)^2), averaged across the two baseline OOF models",
            "w_min": W_MIN,
            "w_max": W_MAX,
            "eps": EPS,
            "alpha_best": alpha_best,
            "alpha_search_range": [float(args.alpha_low), float(args.alpha_high)],
            "optuna_timeout_sec": int(args.optuna_timeout_sec),
            "optuna_trials_completed": int(len(study.trials)),
            "effective_bins": int(len(final_curves)),
            "requested_bins": int(args.env_quantile_bins),
            "smooth_sigma": float(args.smooth_sigma),
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
                "baseline_oof_rmse": float(baseline_scores["et_rowagg_mf06_bs"]),
                "weighted_oof_wrmse": float(weighted_scores["et_rowagg_mf06_bs"]),
                "baseline_oof_path": display_path(output_dir / "et_rowagg_mf06_bs_baseline_oof.csv"),
                "weighted_oof_path": display_path(output_dir / "et_rowagg_mf06_bs_weighted_oof.csv"),
                "weighted_test_path": display_path(output_dir / "et_rowagg_mf06_bs_weighted_test.csv"),
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
                "baseline_oof_rmse": float(baseline_scores["et_allpool_3"]),
                "weighted_oof_wrmse": float(weighted_scores["et_allpool_3"]),
                "baseline_oof_path": display_path(output_dir / "et_allpool_3_baseline_oof.csv"),
                "weighted_oof_path": display_path(output_dir / "et_allpool_3_weighted_oof.csv"),
                "weighted_test_path": display_path(output_dir / "et_allpool_3_weighted_test.csv"),
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
            "target_weights_path": relative_paths["target_simplex"],
            "blended_oof_modelspace_path": relative_paths["blend_model"],
            "blended_oof_full_path": relative_paths["blend_full"],
            "blended_test_modelspace_path": relative_paths["blend_test_model"],
        },
        "artifacts": relative_paths,
        "fold_reports": {
            "baseline": baseline_fold_reports,
            "weighted": weighted_fold_reports,
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_progress(f"Final blended OOF weighted WRMSE={blended_oof_score:.6f}")
    log_progress(f"Submission written to {paths['submission']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
