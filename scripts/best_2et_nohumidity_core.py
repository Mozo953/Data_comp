from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaz_competition.data import build_submission_frame, load_modeling_data  # noqa: E402


RAW_COLUMNS = ["M12", "M13", "M14", "M15", "M4", "M5", "M6", "M7", "R", "S1", "S2", "S3"]
BLOCK_A = ["M12", "M13", "M14", "M15"]
BLOCK_B = ["M4", "M5", "M6", "M7"]
SUPPORT = ["R", "S1", "S2", "S3"]
MODEL_ORDER = ["et_rowagg_mf06_bs", "et_allpool_3"]


@dataclass(frozen=True)
class FeatureViews:
    rowagg_fit: pd.DataFrame
    rowagg_pred: pd.DataFrame
    allpool_fit: pd.DataFrame
    allpool_pred: pd.DataFrame
    raw_fit: pd.DataFrame
    raw_pred: pd.DataFrame
    clipping_profile: dict[str, float | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Core helpers for the 2ET nohumidity Dirichlet blend: CV3 ExtraTrees rowagg + allpool, "
            "Humidity dropped before FE, Humidity>=0.6 sample_weight=1.2, Dirichlet blend."
        )
    )
    parser.add_argument("--data-dir", default="src/gaz_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391/core_baseline")
    parser.add_argument("--submission-prefix", default="core_2et_nohumidity_dirichlet")
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
        raise ValueError("The 2ET nohumidity core pipeline is defined with CV=3.")
    return args


def log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def verbose_log(enabled: bool, message: str) -> None:
    if enabled:
        log_progress(message)


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


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


def drop_humidity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        column
        for column in frame.columns
        if column != "ID"
        and column != "Humidity"
        and column != "support_gap"
        and not column.startswith("humidity_")
        and not column.startswith("humidity_times_")
    ]
    return frame.loc[:, keep_columns].copy()


def validate_no_humidity_columns(frame: pd.DataFrame, label: str) -> None:
    forbidden = [
        column
        for column in frame.columns
        if column == "Humidity"
        or column == "support_gap"
        or column.startswith("humidity_")
        or column.startswith("humidity_times_")
    ]
    if forbidden:
        raise ValueError(f"Humidity leakage detected in {label}: {forbidden}")


def compute_humidity_weights(humidity: pd.Series) -> pd.Series:
    values = np.where(humidity.to_numpy(dtype=np.float32) >= 0.6, 1.2, 1.0).astype(np.float32)
    return pd.Series(values, index=humidity.index, name="humidity_weight")


def summarize_humidity_bins(train_humidity: pd.Series, test_humidity: pd.Series) -> pd.DataFrame:
    bins = [
        ("[0.0, 0.2)", train_humidity < 0.2, test_humidity < 0.2, 1.0),
        ("[0.2, 0.6)", (train_humidity >= 0.2) & (train_humidity < 0.6), (test_humidity >= 0.2) & (test_humidity < 0.6), 1.0),
        ("[0.6, 1.0]", train_humidity >= 0.6, test_humidity >= 0.6, 1.2),
    ]
    return pd.DataFrame(
        [
            {
                "humidity_interval": interval,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "fixed_weight": float(weight),
            }
            for interval, train_mask, test_mask, weight in bins
        ]
    )


def clip_raw_features(
    fit_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    *,
    tail_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    fit_raw = fit_frame[RAW_COLUMNS].copy()
    pred_raw = pred_frame[RAW_COLUMNS].copy()
    lower = fit_raw.quantile(tail_quantile)
    upper = fit_raw.quantile(1.0 - tail_quantile)
    clipped_fit = fit_raw.clip(lower=lower, upper=upper, axis="columns").astype(np.float32)
    clipped_pred = pred_raw.clip(lower=lower, upper=upper, axis="columns").astype(np.float32)
    profile = {
        "tail_quantile_low": float(tail_quantile),
        "tail_quantile_high": float(1.0 - tail_quantile),
        "feature_count": int(clipped_fit.shape[1]),
    }
    return clipped_fit, clipped_pred, profile


def safe_log(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=np.float32), 0.0, None))


def add_row_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw[RAW_COLUMNS].to_numpy(dtype=np.float32)
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1)
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90], axis=1)
    mad = np.mean(np.abs(values - row_mean.reshape(-1, 1)), axis=1)
    l1 = np.abs(values).sum(axis=1)
    l2 = np.sqrt(np.square(values).sum(axis=1))
    return pd.DataFrame(
        {
            "row_mean": row_mean,
            "row_std": row_std,
            "row_p10": p10,
            "row_p25": p25,
            "row_p50": p50,
            "row_p75": p75,
            "row_p90": p90,
            "row_iqr": p75 - p25,
            "row_mad": mad,
            "row_l1": l1,
            "row_l2": l2,
        },
        index=raw.index,
        dtype=np.float32,
    )


def add_block_features(raw: pd.DataFrame) -> pd.DataFrame:
    block_a_mean = raw[BLOCK_A].mean(axis=1)
    block_b_mean = raw[BLOCK_B].mean(axis=1)
    support_mean = raw[SUPPORT].mean(axis=1)
    return pd.DataFrame(
        {
            "block_a_mean": block_a_mean,
            "block_b_mean": block_b_mean,
            "support_mean": support_mean,
            "block_gap": block_a_mean - block_b_mean,
        },
        index=raw.index,
        dtype=np.float32,
    )


def build_rowagg_features(raw: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
    stats = add_row_statistics(raw)
    blocks = add_block_features(raw)
    row_mean = stats["row_mean"].to_numpy(dtype=np.float32)
    row_l1 = stats["row_l1"].to_numpy(dtype=np.float32)
    row_l2 = stats["row_l2"].to_numpy(dtype=np.float32)
    block_a_mean = blocks["block_a_mean"].to_numpy(dtype=np.float32)
    block_b_mean = blocks["block_b_mean"].to_numpy(dtype=np.float32)

    rowagg = pd.concat([stats, blocks], axis=1)
    rowagg["log_M12"] = safe_log(raw["M12"])
    rowagg["log_M15"] = safe_log(raw["M15"])
    rowagg["log_M4"] = safe_log(raw["M4"])
    rowagg["log_M7"] = safe_log(raw["M7"])
    rowagg["log_row_l1"] = safe_log(row_l1)
    rowagg["log_row_l2"] = safe_log(row_l2)
    rowagg["M12_over_row_mean"] = raw["M12"].to_numpy(dtype=np.float32) / (row_mean + ratio_eps)
    rowagg["M4_over_row_mean"] = raw["M4"].to_numpy(dtype=np.float32) / (row_mean + ratio_eps)
    rowagg["R_over_row_mean"] = raw["R"].to_numpy(dtype=np.float32) / (row_mean + ratio_eps)
    rowagg["M12_minus_block_a_mean"] = raw["M12"].to_numpy(dtype=np.float32) - block_a_mean
    rowagg["M4_minus_block_b_mean"] = raw["M4"].to_numpy(dtype=np.float32) - block_b_mean
    return rowagg.astype(np.float32)


def block_mean_for_column(column: str, blocks: pd.DataFrame) -> np.ndarray:
    if column in BLOCK_A:
        return blocks["block_a_mean"].to_numpy(dtype=np.float32)
    if column in BLOCK_B:
        return blocks["block_b_mean"].to_numpy(dtype=np.float32)
    return blocks["support_mean"].to_numpy(dtype=np.float32)


def block_mean_suffix_for_column(column: str) -> str:
    if column in BLOCK_A:
        return "block_a_mean"
    if column in BLOCK_B:
        return "block_b_mean"
    return "support_mean"


def build_allpool_features(raw: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
    stats = add_row_statistics(raw)
    blocks = add_block_features(raw)
    row_mean = stats["row_mean"].to_numpy(dtype=np.float32)

    engineered: dict[str, np.ndarray] = {
        "log_row_l1": safe_log(stats["row_l1"]),
        "log_row_l2": safe_log(stats["row_l2"]),
    }

    for column in RAW_COLUMNS:
        values = raw[column].to_numpy(dtype=np.float32)
        engineered[f"log_{column}"] = safe_log(values)
        engineered[f"{column}_over_row_mean"] = values / (row_mean + ratio_eps)
        engineered[f"{column}_minus_{block_mean_suffix_for_column(column)}"] = values - block_mean_for_column(column, blocks)

    for left in RAW_COLUMNS:
        left_values = raw[left].to_numpy(dtype=np.float32)
        for right in RAW_COLUMNS:
            if left == right:
                continue
            right_values = raw[right].to_numpy(dtype=np.float32)
            engineered[f"{left}_over_{right}"] = left_values / (right_values + ratio_eps)

    engineered_frame = pd.DataFrame(engineered, index=raw.index, dtype=np.float32)
    return pd.concat([raw[RAW_COLUMNS].astype(np.float32), stats, blocks, engineered_frame], axis=1).astype(np.float32)


def build_feature_views(
    fit_nohumidity: pd.DataFrame,
    pred_nohumidity: pd.DataFrame,
    *,
    tail_quantile: float,
    ratio_eps: float,
) -> FeatureViews:
    raw_fit, raw_pred, clipping_profile = clip_raw_features(
        fit_nohumidity,
        pred_nohumidity,
        tail_quantile=tail_quantile,
    )
    rowagg_fit = build_rowagg_features(raw_fit, ratio_eps=ratio_eps)
    rowagg_pred = build_rowagg_features(raw_pred, ratio_eps=ratio_eps)
    allpool_fit = build_allpool_features(raw_fit, ratio_eps=ratio_eps)
    allpool_pred = build_allpool_features(raw_pred, ratio_eps=ratio_eps)

    for label, frame in {
        "raw_fit": raw_fit,
        "raw_pred": raw_pred,
        "rowagg_fit": rowagg_fit,
        "rowagg_pred": rowagg_pred,
        "allpool_fit": allpool_fit,
        "allpool_pred": allpool_pred,
    }.items():
        validate_no_humidity_columns(frame, label)

    return FeatureViews(
        rowagg_fit=rowagg_fit,
        rowagg_pred=rowagg_pred,
        allpool_fit=allpool_fit,
        allpool_pred=allpool_pred,
        raw_fit=raw_fit,
        raw_pred=raw_pred,
        clipping_profile=clipping_profile,
    )


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


def make_rowagg_model(n_jobs: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=28,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.6,
        bootstrap=True,
        max_samples=0.8051184329566651,
        random_state=42,
        n_jobs=n_jobs,
    )


def make_allpool_model(n_jobs: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=17,
        min_samples_split=13,
        min_samples_leaf=5,
        max_features=0.25,
        bootstrap=False,
        random_state=83,
        n_jobs=n_jobs,
    )


def fit_models_and_predict(
    views: FeatureViews,
    y_fit: pd.DataFrame,
    weights_fit: pd.Series,
    *,
    n_jobs: int,
) -> dict[str, pd.DataFrame]:
    rowagg_model = make_rowagg_model(n_jobs)
    rowagg_model.fit(views.rowagg_fit, y_fit, sample_weight=weights_fit.to_numpy(dtype=np.float32))
    rowagg_pred = pd.DataFrame(
        rowagg_model.predict(views.rowagg_pred),
        index=views.rowagg_pred.index,
        columns=y_fit.columns,
        dtype=np.float32,
    ).clip(0.0, 1.0)

    allpool_model = make_allpool_model(n_jobs)
    allpool_model.fit(views.allpool_fit, y_fit, sample_weight=weights_fit.to_numpy(dtype=np.float32))
    allpool_pred = pd.DataFrame(
        allpool_model.predict(views.allpool_pred),
        index=views.allpool_pred.index,
        columns=y_fit.columns,
        dtype=np.float32,
    ).clip(0.0, 1.0)

    return {
        "et_rowagg_mf06_bs": rowagg_pred.astype(np.float32),
        "et_allpool_3": allpool_pred.astype(np.float32),
    }


def make_oof_and_test_predictions(
    x_train_nohumidity: pd.DataFrame,
    y_train: pd.DataFrame,
    weights_train: pd.Series,
    x_test_nohumidity: pd.DataFrame,
    *,
    cv_folds: int,
    random_state: int,
    tail_quantile: float,
    ratio_eps: float,
    n_jobs: int,
    verbose: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], FeatureViews, dict[str, list[dict[str, int]]]]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_predictions = {
        model_name: pd.DataFrame(index=y_train.index, columns=y_train.columns, dtype=np.float32)
        for model_name in MODEL_ORDER
    }
    fold_reports = {model_name: [] for model_name in MODEL_ORDER}

    for fold_number, (fit_idx, valid_idx) in enumerate(cv.split(x_train_nohumidity), start=1):
        fit_index = x_train_nohumidity.index[fit_idx]
        valid_index = x_train_nohumidity.index[valid_idx]
        views = build_feature_views(
            x_train_nohumidity.loc[fit_index],
            x_train_nohumidity.loc[valid_index],
            tail_quantile=tail_quantile,
            ratio_eps=ratio_eps,
        )
        fold_predictions = fit_models_and_predict(
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
        verbose_log(verbose, f"OOF fold {fold_number}/{cv_folds} ready")

    full_views = build_feature_views(
        x_train_nohumidity,
        x_test_nohumidity,
        tail_quantile=tail_quantile,
        ratio_eps=ratio_eps,
    )
    test_predictions = fit_models_and_predict(
        full_views,
        y_train,
        weights_train,
        n_jobs=n_jobs,
    )
    return oof_predictions, test_predictions, full_views, fold_reports


def optimize_dirichlet_blend(
    model_oofs: dict[str, pd.DataFrame],
    y_train: pd.DataFrame,
    *,
    row_weights: pd.Series,
    alpha_vector: np.ndarray,
    sample_count: int,
    batch_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    candidates = rng.dirichlet(alpha_vector.astype(np.float64), size=sample_count).astype(np.float32)
    candidates = np.vstack(
        [
            (alpha_vector / alpha_vector.sum()).reshape(1, -1),
            np.full((1, len(MODEL_ORDER)), 1.0 / len(MODEL_ORDER), dtype=np.float32),
            np.eye(len(MODEL_ORDER), dtype=np.float32),
            candidates,
        ]
    ).astype(np.float32)

    target_weights = pd.DataFrame(index=y_train.columns, columns=MODEL_ORDER, dtype=np.float32)
    blended = pd.DataFrame(index=y_train.index, columns=y_train.columns, dtype=np.float32)
    row_weight_values = row_weights.to_numpy(dtype=np.float32)

    for target in y_train.columns:
        stacked = np.column_stack([model_oofs[model_name][target].to_numpy(dtype=np.float32) for model_name in MODEL_ORDER])
        y_true = y_train[target].to_numpy(dtype=np.float32)
        best_mse = float("inf")
        best_weights: np.ndarray | None = None
        best_pred: np.ndarray | None = None
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            pred_chunk = np.clip(stacked @ chunk.T, 0.0, 1.0)
            mse_chunk = np.mean(row_weight_values[:, None] * np.square(pred_chunk - y_true[:, None]), axis=0)
            local_idx = int(np.argmin(mse_chunk))
            if float(mse_chunk[local_idx]) < best_mse:
                best_mse = float(mse_chunk[local_idx])
                best_weights = chunk[local_idx].copy()
                best_pred = pred_chunk[:, local_idx].copy()
        if best_weights is None or best_pred is None:
            raise RuntimeError(f"Dirichlet blend failed for target {target}.")
        target_weights.loc[target, :] = best_weights
        blended[target] = best_pred.astype(np.float32)

    return target_weights.astype(np.float32), blended.astype(np.float32)


def apply_targetwise_blend(model_predictions: dict[str, pd.DataFrame], target_weights: pd.DataFrame) -> pd.DataFrame:
    blended = pd.DataFrame(index=next(iter(model_predictions.values())).index, columns=target_weights.index, dtype=np.float32)
    for target in target_weights.index:
        values = np.zeros(len(blended), dtype=np.float32)
        for model_name in MODEL_ORDER:
            values += float(target_weights.loc[target, model_name]) * model_predictions[model_name][target].to_numpy(dtype=np.float32)
        blended[target] = np.clip(values, 0.0, 1.0).astype(np.float32)
    return blended


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = maybe_subsample_bundle(
        load_modeling_data(data_dir),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    humidity_train = bundle.data.x_train["Humidity"].copy()
    humidity_test = bundle.data.x_test["Humidity"].copy()
    row_weights = compute_humidity_weights(humidity_train)
    humidity_weight_bins = summarize_humidity_bins(humidity_train, humidity_test)

    x_train_nohumidity = drop_humidity_columns(bundle.data.x_train)
    x_test_nohumidity = drop_humidity_columns(bundle.data.x_test)
    validate_no_humidity_columns(x_train_nohumidity, "x_train_nohumidity")
    validate_no_humidity_columns(x_test_nohumidity, "x_test_nohumidity")

    target_multiplicities = get_target_multiplicities(bundle.schema, list(bundle.y_train_model.columns))
    full_target_count = len(bundle.schema.original_targets)
    alpha_vector = np.asarray(args.dirichlet_alpha_vector, dtype=np.float32)

    log_progress("2ET nohumidity core: CV=3, Humidity dropped before FE, sample_weight=1.2 if Humidity>=0.6")
    model_oofs, model_tests, full_views, fold_reports = make_oof_and_test_predictions(
        x_train_nohumidity,
        bundle.y_train_model,
        row_weights,
        x_test_nohumidity,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        n_jobs=int(args.n_jobs),
        verbose=bool(args.verbose),
    )

    base_scores = {
        model_name: weighted_wrmse(
            bundle.y_train_model,
            model_oofs[model_name],
            row_weights=row_weights,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        )
        for model_name in MODEL_ORDER
    }
    for model_name, score in base_scores.items():
        log_progress(f"{model_name}: CV3 weighted WRMSE={score:.6f}")

    blend_weights, blended_oof_model = optimize_dirichlet_blend(
        model_oofs,
        bundle.y_train_model,
        row_weights=row_weights,
        alpha_vector=alpha_vector,
        sample_count=int(args.dirichlet_samples),
        batch_size=int(args.dirichlet_batch_size),
        random_state=int(args.random_state),
    )
    blended_test_model = apply_targetwise_blend(model_tests, blend_weights)
    blended_oof_score = weighted_wrmse(
        bundle.y_train_model,
        blended_oof_model,
        row_weights=row_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )

    blended_oof_full = bundle.schema.expand_predictions(blended_oof_model)
    blended_test_full = bundle.schema.expand_predictions(blended_test_model)
    submission = build_submission_frame(bundle.data.x_test["ID"], blended_test_full)

    feature_manifest = {
        "raw_clean_count": int(full_views.raw_fit.shape[1]),
        "rowagg_clean_count": int(full_views.rowagg_fit.shape[1]),
        "allpool_nohumidity_count": int(full_views.allpool_fit.shape[1]),
        "rowagg_feature_names": list(full_views.rowagg_fit.columns),
        "allpool_first_40_feature_names": list(full_views.allpool_fit.columns[:40]),
    }
    if feature_manifest["raw_clean_count"] != 12 or feature_manifest["rowagg_clean_count"] != 26 or feature_manifest["allpool_nohumidity_count"] != 197:
        raise ValueError(f"Unexpected feature counts: {feature_manifest}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    summary_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    humidity_bins_path = output_dir / f"{args.submission_prefix}_{timestamp}_humidity_weight_bins.csv"
    blend_weights_path = output_dir / f"{args.submission_prefix}_{timestamp}_target_simplex_weights.csv"
    blended_oof_model_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_blend_modelspace.csv"
    blended_oof_full_path = output_dir / f"{args.submission_prefix}_{timestamp}_oof_blend_full.csv"
    blended_test_model_path = output_dir / f"{args.submission_prefix}_{timestamp}_test_blend_modelspace.csv"
    feature_manifest_path = output_dir / f"{args.submission_prefix}_{timestamp}_feature_manifest.json"

    submission.to_csv(submission_path, index=False)
    humidity_weight_bins.to_csv(humidity_bins_path, index=False)
    blend_weights.to_csv(blend_weights_path, index=True)
    blended_oof_model.to_csv(blended_oof_model_path, index=True)
    blended_oof_full.to_csv(blended_oof_full_path, index=True)
    blended_test_model.to_csv(blended_test_model_path, index=True)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    for model_name in MODEL_ORDER:
        model_oofs[model_name].to_csv(output_dir / f"{model_name}_oof.csv", index=True)
        model_tests[model_name].to_csv(output_dir / f"{model_name}_test.csv", index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Core CV3 2ET nohumidity blend: et_rowagg_mf06_bs + et_allpool_3",
        "training": {
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "metric": "WRMSE = sqrt(mean_over_rows_targets(row_weight * squared_error)) with row_weight=1.2 if Humidity>=0.6 else 1.0",
            "postprocessing": False,
        },
        "preprocessing": {
            "clipping": full_views.clipping_profile,
            "humidity_removed_before_feature_engineering": True,
            "forbidden_feature_patterns": ["Humidity", "humidity_*", "humidity_times_*", "support_gap"],
        },
        "feature_views": feature_manifest,
        "sample_weighting": {
            "rule": {"Humidity < 0.6": 1.0, "Humidity >= 0.6": 1.2},
            "humidity_weight_bins_path": str(humidity_bins_path.relative_to(ROOT)),
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
                "dataset": "allpool_nohumidity",
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

