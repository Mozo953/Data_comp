from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaz_competition.data import load_modeling_data  # noqa: E402


RAW_COLUMNS = ["M12", "M13", "M14", "M15", "M4", "M5", "M6", "M7", "R", "S1", "S2", "S3"]
BLOCK_A = ["M12", "M13", "M14", "M15"]
BLOCK_B = ["M4", "M5", "M6", "M7"]
SUPPORT = ["R", "S1", "S2", "S3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local AdaBoost model trained only on Humidity in [0.45, 0.80], using 201 no-Humidity "
            "features, with shallow trees to reduce overfit. Test rows outside the bin are predicted as 1.0."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/52_adaboost_humidity_045_080")
    parser.add_argument("--prefix", default="adaboost_humidity_045_080")
    parser.add_argument("--humidity-low", type=float, default=0.45)
    parser.add_argument("--humidity-high", type=float, default=0.80)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=128)
    parser.add_argument("--loss", choices=["linear", "square", "exponential"], default="linear")
    parser.add_argument(
        "--use-piecewise-sample-weights",
        action="store_true",
        help="Optional: use model50-style piecewise weights inside the local humidity bin. Default is unweighted.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 <= args.humidity_low < args.humidity_high <= 1.0:
        raise ValueError("Humidity bounds must satisfy 0 <= low < high <= 1.")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


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
    return stem or "adaboost_humidity_045_080"


def load_clean_model42_module():
    module_path = ROOT / "scripts" / "best_2et_nohumidity_core.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing helper script: {module_path}")
    spec = importlib.util.spec_from_file_location("best_2et_nohumidity_core", module_path)
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


def compute_model50_weights(humidity: pd.Series) -> pd.Series:
    values = np.full(len(humidity), 1.1, dtype=np.float32)
    h = humidity.to_numpy(dtype=np.float32)
    values[h < 0.2] = 1.0
    values[(h >= 0.39) & (h < 0.50)] = 1.35
    values[(h >= 0.50) & (h < 0.68)] = 1.0
    values[(h >= 0.68) & (h < 0.80)] = 1.25
    values[h >= 0.80] = 1.25
    return pd.Series(values, index=humidity.index, name="model50_sample_weight")


def add_four_extra_features(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_a_std": raw[BLOCK_A].std(axis=1).to_numpy(dtype=np.float32),
            "block_b_std": raw[BLOCK_B].std(axis=1).to_numpy(dtype=np.float32),
            "support_std": raw[SUPPORT].std(axis=1).to_numpy(dtype=np.float32),
            "support_range": (raw[SUPPORT].max(axis=1) - raw[SUPPORT].min(axis=1)).to_numpy(dtype=np.float32),
        },
        index=raw.index,
        dtype=np.float32,
    )


def build_201_features(
    clean,
    fit_nohum: pd.DataFrame,
    pred_nohum: pd.DataFrame,
    *,
    tail_quantile: float,
    ratio_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    views = clean.build_feature_views(
        fit_nohum,
        pred_nohum,
        tail_quantile=tail_quantile,
        ratio_eps=ratio_eps,
    )
    fit_201 = pd.concat([views.allpool_fit, add_four_extra_features(views.raw_fit)], axis=1).astype(np.float32)
    pred_201 = pd.concat([views.allpool_pred, add_four_extra_features(views.raw_pred)], axis=1).astype(np.float32)
    forbidden = [
        column
        for column in fit_201.columns
        if column == "Humidity"
        or column.startswith("humidity_")
        or column.startswith("env_")
        or column == "support_gap"
    ]
    if forbidden:
        raise ValueError(f"Humidity leakage detected in AdaBoost features: {forbidden}")
    if fit_201.shape[1] != 201 or pred_201.shape[1] != 201:
        raise ValueError(f"Expected 201 features, got fit={fit_201.shape[1]}, pred={pred_201.shape[1]}")
    return fit_201, pred_201, views.clipping_profile


def make_adaboost(args: argparse.Namespace) -> MultiOutputRegressor:
    tree = DecisionTreeRegressor(
        max_depth=int(args.tree_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.random_state),
    )
    ada = AdaBoostRegressor(
        estimator=tree,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        loss=str(args.loss),
        random_state=int(args.random_state),
    )
    return MultiOutputRegressor(ada, n_jobs=int(args.n_jobs))


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


def make_local_oof(
    clean,
    x_local_nohum: pd.DataFrame,
    y_local: pd.DataFrame,
    weights_local: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, int]], dict[str, float | int]]:
    cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.random_state))
    oof = pd.DataFrame(index=y_local.index, columns=y_local.columns, dtype=np.float32)
    fold_reports: list[dict[str, int]] = []
    last_profile: dict[str, float | int] = {}

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_local_nohum), start=1):
        fit_index = x_local_nohum.index[fit_idx]
        valid_index = x_local_nohum.index[valid_idx]
        x_fit, x_valid, last_profile = build_201_features(
            clean,
            x_local_nohum.loc[fit_index],
            x_local_nohum.loc[valid_index],
            tail_quantile=float(args.tail_quantile),
            ratio_eps=float(args.ratio_eps),
        )
        model = make_adaboost(args)
        fit_kwargs = {}
        if args.use_piecewise_sample_weights:
            fit_kwargs["sample_weight"] = weights_local.loc[fit_index].to_numpy(dtype=np.float32)
        model.fit(x_fit, y_local.loc[fit_index], **fit_kwargs)
        pred = np.clip(model.predict(x_valid), 0.0, 1.0)
        oof.loc[valid_index] = pred.astype(np.float32)
        fold_reports.append({"fold": fold, "fit_rows": int(len(fit_index)), "valid_rows": int(len(valid_index))})
        if args.verbose:
            log(f"OOF fold {fold}/{args.cv_folds} ready")

    if oof.isna().any().any():
        raise RuntimeError("Missing local AdaBoost OOF predictions.")
    return oof.astype(np.float32), fold_reports, last_profile


def main() -> None:
    args = parse_args()
    clean = load_clean_model42_module()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_stem(args.prefix)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bundle = maybe_subsample_bundle(
        load_modeling_data(data_dir),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    humidity_train = bundle.data.x_train["Humidity"]
    humidity_test = bundle.data.x_test["Humidity"]
    train_mask = (humidity_train >= float(args.humidity_low)) & (humidity_train <= float(args.humidity_high))
    test_mask = (humidity_test >= float(args.humidity_low)) & (humidity_test <= float(args.humidity_high))
    if int(train_mask.sum()) < int(args.cv_folds):
        raise ValueError(f"Not enough local train rows in humidity bin: {int(train_mask.sum())}")

    x_train_nohum = clean.drop_environment_columns(bundle.data.x_train)
    x_test_nohum = clean.drop_environment_columns(bundle.data.x_test)
    y_local = bundle.y_train_model.loc[train_mask].copy()
    x_local_nohum = x_train_nohum.loc[train_mask].copy()
    if args.use_piecewise_sample_weights:
        local_weights = compute_model50_weights(humidity_train.loc[train_mask])
    else:
        local_weights = pd.Series(np.ones(len(y_local), dtype=np.float32), index=y_local.index, name="unit_weight")

    log(
        f"Training local AdaBoost on Humidity in [{args.humidity_low}, {args.humidity_high}] "
        f"with train_rows={int(train_mask.sum())}, test_rows={int(test_mask.sum())}"
    )
    local_oof, fold_reports, cv_profile = make_local_oof(clean, x_local_nohum, y_local, local_weights, args)
    target_multiplicities = get_target_multiplicities(bundle.schema, list(bundle.y_train_model.columns))
    full_target_count = len(bundle.schema.original_targets)
    local_oof_score = weighted_wrmse(
        y_local,
        local_oof,
        row_weights=local_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )
    log(f"Local AdaBoost CV WRMSE={local_oof_score:.6f}")

    x_fit_201, x_test_201, full_profile = build_201_features(
        clean,
        x_local_nohum,
        x_test_nohum.loc[test_mask],
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
    )
    model = make_adaboost(args)
    fit_kwargs = {}
    if args.use_piecewise_sample_weights:
        fit_kwargs["sample_weight"] = local_weights.to_numpy(dtype=np.float32)
    model.fit(x_fit_201, y_local, **fit_kwargs)
    local_test_pred = pd.DataFrame(
        np.clip(model.predict(x_test_201), 0.0, 1.0),
        index=x_test_201.index,
        columns=bundle.y_train_model.columns,
        dtype=np.float32,
    )
    local_test_full = bundle.schema.expand_predictions(local_test_pred)

    final_submission = pd.DataFrame({"ID": bundle.data.x_test["ID"].to_numpy()})
    for target in bundle.schema.original_targets:
        final_submission[target] = 1.0
    final_submission.loc[test_mask.to_numpy(), local_test_full.columns] = local_test_full.to_numpy(dtype=np.float32)

    paths = {
        "summary": output_dir / f"{prefix}_{timestamp}.json",
        "local_oof_modelspace": output_dir / f"{prefix}_{timestamp}_local_oof_modelspace.csv",
        "local_test_modelspace": output_dir / f"{prefix}_{timestamp}_local_test_modelspace.csv",
        "local_test_full": output_dir / f"{prefix}_{timestamp}_local_test_full.csv",
        "submission": output_dir / f"{prefix}_{timestamp}.csv",
        "feature_manifest": output_dir / f"{prefix}_{timestamp}_feature_manifest.csv",
        "local_rows": output_dir / f"{prefix}_{timestamp}_local_rows.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    local_oof.to_csv(paths["local_oof_modelspace"], index=True)
    local_test_pred.to_csv(paths["local_test_modelspace"], index=True)
    local_test_full.to_csv(paths["local_test_full"], index=True)
    final_submission.to_csv(paths["submission"], index=False)
    pd.DataFrame({"feature": list(x_fit_201.columns)}).to_csv(paths["feature_manifest"], index=False)
    pd.DataFrame(
        {
            "train_rows_in_bin": [int(train_mask.sum())],
            "test_rows_in_bin": [int(test_mask.sum())],
            "humidity_low": [float(args.humidity_low)],
            "humidity_high": [float(args.humidity_high)],
        }
    ).to_csv(paths["local_rows"], index=False)

    summary = {
        "generated_at_utc": timestamp,
        "model": "Local AdaBoost on Humidity bin [0.45, 0.80] with 201 no-Humidity features",
        "humidity_bin": {
            "low": float(args.humidity_low),
            "high": float(args.humidity_high),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        },
        "anti_overfit_settings": {
            "base_tree_max_depth": int(args.tree_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "loss": str(args.loss),
            "cv_folds": int(args.cv_folds),
        },
        "features": {
            "count": int(x_fit_201.shape[1]),
            "humidity_removed": True,
            "source": "allpool 197 + block_a_std + block_b_std + support_std + support_range",
            "clipping_cv_profile": cv_profile,
            "clipping_full_profile": full_profile,
        },
        "sample_weighting": {
            "used": bool(args.use_piecewise_sample_weights),
            "rule": "unit weights by default; optional model50-style piecewise weights inside local bin",
        },
        "local_cv": {
            "weighted_wrmse": float(local_oof_score),
            "fold_reports": fold_reports,
        },
        "outside_bin_prediction": {
            "value": 1.0,
            "rule": "All test rows outside the humidity bin are forced to 1.0 for every target.",
        },
        "artifacts": {key: display_path(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Final submission written to {paths['submission']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
