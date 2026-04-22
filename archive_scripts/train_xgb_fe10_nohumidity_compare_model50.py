from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaz_competition.data import build_submission_frame, load_modeling_data  # noqa: E402


RAW_COLUMNS = ["M12", "M13", "M14", "M15", "M4", "M5", "M6", "M7", "R", "S1", "S2", "S3"]
MODEL_ORDER_NAME = "xgb_fe10_nohumidity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a shallow XGBoost no-Humidity model with FE10 logs/poly2, max 15 selected features, "
            "CV=3, model50 sample weights, and compare OOF correlation with model50 blend."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--model50-dir", default="artifacts_extratrees_corr_optuna/50_blender_et3_rowaggbs_piecewise_model50_cv3")
    parser.add_argument("--model50-oof-file", default=None)
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/51_xgb_fe10_nohumidity_compare_model50")
    parser.add_argument("--prefix", default="xgb_fe10_nohumidity_compare_model50")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--max-selected-features", type=int, default=15)
    parser.add_argument("--optuna-trials", type=int, default=10)
    parser.add_argument("--optuna-timeout-per-trial-sec", type=int, default=180)
    parser.add_argument("--xgb-n-estimators", type=int, default=450)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cv_folds != 3:
        raise ValueError("Cette pipeline est definie pour CV=3.")
    if args.max_selected_features < 1:
        raise ValueError("--max-selected-features must be >= 1.")
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
    return stem or "xgb_fe10_nohumidity_compare_model50"


def require_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for this script. Install it in your active environment with: "
            "pip install xgboost"
        ) from exc
    return XGBRegressor


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


def clip_raw_frames(
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


def build_xgb_fe10(raw: pd.DataFrame, *, ratio_eps: float) -> pd.DataFrame:
    raw = raw[RAW_COLUMNS].astype(np.float32)
    values = raw.to_numpy(dtype=np.float32)
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1)

    engineered = {
        "log_M12": np.log1p(np.clip(raw["M12"].to_numpy(dtype=np.float32), 0.0, None)),
        "log_M15": np.log1p(np.clip(raw["M15"].to_numpy(dtype=np.float32), 0.0, None)),
        "log_M4": np.log1p(np.clip(raw["M4"].to_numpy(dtype=np.float32), 0.0, None)),
        "log_M7": np.log1p(np.clip(raw["M7"].to_numpy(dtype=np.float32), 0.0, None)),
        "log_R": np.log1p(np.clip(raw["R"].to_numpy(dtype=np.float32), 0.0, None)),
        "row_mean": row_mean,
        "row_std": row_std,
        "M12_sq": np.square(raw["M12"].to_numpy(dtype=np.float32)),
        "M4_sq": np.square(raw["M4"].to_numpy(dtype=np.float32)),
        "R_sq": np.square(raw["R"].to_numpy(dtype=np.float32)),
    }
    frame = pd.concat([raw, pd.DataFrame(engineered, index=raw.index, dtype=np.float32)], axis=1)
    frame["M12_x_M4"] = raw["M12"].to_numpy(dtype=np.float32) * raw["M4"].to_numpy(dtype=np.float32)
    frame["M15_x_M7"] = raw["M15"].to_numpy(dtype=np.float32) * raw["M7"].to_numpy(dtype=np.float32)
    frame["R_over_row_mean"] = raw["R"].to_numpy(dtype=np.float32) / (row_mean + ratio_eps)
    return frame.astype(np.float32)


def select_max_features(
    x_fit: pd.DataFrame,
    y_fit: pd.DataFrame,
    *,
    max_features: int,
) -> list[str]:
    if x_fit.shape[1] <= max_features:
        return list(x_fit.columns)
    scores: dict[str, float] = {}
    y_mean = y_fit.mean(axis=1).to_numpy(dtype=np.float32)
    y_std = float(np.std(y_mean))
    for column in x_fit.columns:
        values = x_fit[column].to_numpy(dtype=np.float32)
        x_std = float(np.std(values))
        if x_std <= 1e-12 or y_std <= 1e-12:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(values, y_mean)[0, 1]))
            if not np.isfinite(score):
                score = 0.0
        scores[column] = score
    return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_features]]


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


def get_target_multiplicities(schema, modeled_targets: list[str]) -> np.ndarray:
    return np.asarray(
        [
            sum(1 for original in schema.original_targets if schema.representative_for_target[original] == target)
            for target in modeled_targets
        ],
        dtype=np.float32,
    )


def make_xgb_model(XGBRegressor, params: dict[str, float | int], *, random_state: int, n_jobs: int):
    base = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def make_oof_predictions(
    XGBRegressor,
    x_train_nohum: pd.DataFrame,
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    *,
    params: dict[str, float | int],
    cv_folds: int,
    random_state: int,
    tail_quantile: float,
    ratio_eps: float,
    max_selected_features: int,
    n_jobs: int,
    verbose: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, int]]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof = pd.DataFrame(index=y_train.index, columns=y_train.columns, dtype=np.float32)
    reports: list[dict[str, object]] = []
    feature_counter: dict[str, int] = {}

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train_nohum), start=1):
        fit_index = x_train_nohum.index[fit_idx]
        valid_index = x_train_nohum.index[valid_idx]
        raw_fit, raw_valid, _ = clip_raw_frames(
            x_train_nohum.loc[fit_index],
            x_train_nohum.loc[valid_index],
            tail_quantile=tail_quantile,
        )
        fe_fit = build_xgb_fe10(raw_fit, ratio_eps=ratio_eps)
        fe_valid = build_xgb_fe10(raw_valid, ratio_eps=ratio_eps)
        selected = select_max_features(fe_fit, y_train.loc[fit_index], max_features=max_selected_features)
        for feature in selected:
            feature_counter[feature] = feature_counter.get(feature, 0) + 1
        model = make_xgb_model(XGBRegressor, params, random_state=random_state + fold, n_jobs=n_jobs)
        model.fit(
            fe_fit[selected],
            y_train.loc[fit_index],
            sample_weight=row_weights.loc[fit_index].to_numpy(dtype=np.float32),
        )
        pred = np.clip(model.predict(fe_valid[selected]), 0.0, 1.0)
        oof.loc[valid_index] = pred.astype(np.float32)
        reports.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_index)),
                "valid_rows": int(len(valid_index)),
                "selected_features": selected,
            }
        )
        if verbose:
            log(f"XGB OOF fold {fold}/{cv_folds} ready; selected={selected}")

    if oof.isna().any().any():
        raise RuntimeError("Missing XGB OOF predictions.")
    return oof.astype(np.float32), reports, feature_counter


def fit_full_and_predict_test(
    XGBRegressor,
    x_train_nohum: pd.DataFrame,
    y_train: pd.DataFrame,
    row_weights: pd.Series,
    x_test_nohum: pd.DataFrame,
    *,
    params: dict[str, float | int],
    random_state: int,
    tail_quantile: float,
    ratio_eps: float,
    max_selected_features: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, list[str], dict[str, float | int]]:
    raw_train, raw_test, profile = clip_raw_frames(x_train_nohum, x_test_nohum, tail_quantile=tail_quantile)
    fe_train = build_xgb_fe10(raw_train, ratio_eps=ratio_eps)
    fe_test = build_xgb_fe10(raw_test, ratio_eps=ratio_eps)
    selected = select_max_features(fe_train, y_train, max_features=max_selected_features)
    model = make_xgb_model(XGBRegressor, params, random_state=random_state, n_jobs=n_jobs)
    model.fit(fe_train[selected], y_train, sample_weight=row_weights.to_numpy(dtype=np.float32))
    pred = np.clip(model.predict(fe_test[selected]), 0.0, 1.0)
    return pd.DataFrame(pred, index=x_test_nohum.index, columns=y_train.columns, dtype=np.float32), selected, profile


def find_model50_oof(model50_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = model50_dir / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(model50_dir.glob("*_oof_blend_modelspace.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *_oof_blend_modelspace.csv found in {model50_dir}")
    return candidates[-1]


def load_model50_oof(path: Path, y_train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "row_index"})
    if "row_index" in frame.columns:
        frame = frame.set_index("row_index")
    else:
        frame.index = y_train.index
    frame.index = frame.index.astype(y_train.index.dtype, copy=False)
    return frame[y_train.columns].reindex(y_train.index).astype(np.float32)


def targetwise_correlation(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in left.columns:
        corr = np.corrcoef(left[target].to_numpy(dtype=np.float64), right[target].to_numpy(dtype=np.float64))[0, 1]
        rows.append({"target": target, "pearson_corr": float(corr)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    XGBRegressor = require_xgboost()
    data_dir = resolve_path(args.data_dir)
    model50_dir = resolve_path(args.model50_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_stem(args.prefix)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bundle = maybe_subsample_bundle(
        load_modeling_data(data_dir),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    x_train_nohum = bundle.data.x_train.drop(columns=["ID", "Humidity"], errors="ignore").copy()
    x_test_nohum = bundle.data.x_test.drop(columns=["ID", "Humidity"], errors="ignore").copy()
    row_weights = compute_model50_weights(bundle.data.x_train["Humidity"])
    target_multiplicities = get_target_multiplicities(bundle.schema, list(bundle.y_train_model.columns))
    full_target_count = len(bundle.schema.original_targets)

    base_params = {
        "n_estimators": int(args.xgb_n_estimators),
        "max_depth": 5,
        "learning_rate": 0.045,
        "subsample": 0.88,
        "colsample_bytree": 0.85,
        "min_child_weight": 5.0,
        "reg_lambda": 3.0,
        "reg_alpha": 0.05,
    }

    trial_rows: list[dict[str, float | int]] = []

    def objective(trial: optuna.Trial) -> float:
        start = time.time()
        params = {
            "n_estimators": int(args.xgb_n_estimators),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.5, log=True),
        }
        oof, _, _ = make_oof_predictions(
            XGBRegressor,
            x_train_nohum,
            bundle.y_train_model,
            row_weights,
            params=params,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            tail_quantile=float(args.tail_quantile),
            ratio_eps=float(args.ratio_eps),
            max_selected_features=int(args.max_selected_features),
            n_jobs=int(args.n_jobs),
            verbose=False,
        )
        score = weighted_wrmse(
            bundle.y_train_model,
            oof,
            row_weights=row_weights,
            target_multiplicities=target_multiplicities,
            full_target_count=full_target_count,
        )
        elapsed = time.time() - start
        trial_rows.append({"trial": int(trial.number), "score": float(score), "elapsed_sec": float(elapsed), **params})
        log(f"Optuna trial {trial.number}: WRMSE={score:.6f}, elapsed={elapsed:.1f}s, params={params}")
        return float(score)

    log(
        f"Starting XGB no-Humidity Optuna: trials={int(args.optuna_trials)}, "
        f"soft max per trial={int(args.optuna_timeout_per_trial_sec)}s"
    )
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(args.random_state)))
    study.optimize(objective, n_trials=int(args.optuna_trials), show_progress_bar=False)
    best_params = dict(base_params)
    best_params.update(study.best_params)

    log(f"Best XGB params: {best_params}")
    xgb_oof, fold_reports, feature_counter = make_oof_predictions(
        XGBRegressor,
        x_train_nohum,
        bundle.y_train_model,
        row_weights,
        params=best_params,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        max_selected_features=int(args.max_selected_features),
        n_jobs=int(args.n_jobs),
        verbose=bool(args.verbose),
    )
    xgb_score = weighted_wrmse(
        bundle.y_train_model,
        xgb_oof,
        row_weights=row_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )
    xgb_test, full_selected_features, clipping_profile = fit_full_and_predict_test(
        XGBRegressor,
        x_train_nohum,
        bundle.y_train_model,
        row_weights,
        x_test_nohum,
        params=best_params,
        random_state=int(args.random_state),
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
        max_selected_features=int(args.max_selected_features),
        n_jobs=int(args.n_jobs),
    )

    model50_oof_path = find_model50_oof(model50_dir, args.model50_oof_file)
    model50_oof = load_model50_oof(model50_oof_path, bundle.y_train_model)
    corr_by_target = targetwise_correlation(xgb_oof, model50_oof)
    corr_summary = {
        "mean": float(corr_by_target["pearson_corr"].mean()),
        "std": float(corr_by_target["pearson_corr"].std(ddof=1)),
        "min": float(corr_by_target["pearson_corr"].min()),
        "max": float(corr_by_target["pearson_corr"].max()),
    }
    model50_score = weighted_wrmse(
        bundle.y_train_model,
        model50_oof,
        row_weights=row_weights,
        target_multiplicities=target_multiplicities,
        full_target_count=full_target_count,
    )

    xgb_full = bundle.schema.expand_predictions(xgb_test)
    submission = build_submission_frame(bundle.data.x_test["ID"], xgb_full)

    paths = {
        "summary": output_dir / f"{prefix}_{timestamp}.json",
        "oof": output_dir / f"{prefix}_{timestamp}_oof_modelspace.csv",
        "test": output_dir / f"{prefix}_{timestamp}_test_modelspace.csv",
        "submission": output_dir / f"{prefix}_{timestamp}.csv",
        "correlation": output_dir / f"{prefix}_{timestamp}_correlation_vs_model50.csv",
        "optuna_trials": output_dir / f"{prefix}_{timestamp}_optuna_trials.csv",
        "feature_counts": output_dir / f"{prefix}_{timestamp}_selected_feature_counts.csv",
        "sample_weights": output_dir / f"{prefix}_{timestamp}_sample_weights.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    xgb_oof.to_csv(paths["oof"], index=True)
    xgb_test.to_csv(paths["test"], index=True)
    submission.to_csv(paths["submission"], index=False)
    corr_by_target.to_csv(paths["correlation"], index=False)
    pd.DataFrame(trial_rows).to_csv(paths["optuna_trials"], index=False)
    pd.DataFrame(
        [{"feature": feature, "selected_folds": count} for feature, count in sorted(feature_counter.items())]
    ).to_csv(paths["feature_counts"], index=False)
    pd.DataFrame({"Humidity": bundle.data.x_train["Humidity"], "sample_weight": row_weights}).to_csv(paths["sample_weights"], index=True)

    summary = {
        "generated_at_utc": timestamp,
        "model": "XGBoost no-Humidity FE10 logs/poly2 max15 compared with model50 blend OOF",
        "training": {
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "sample_weights": "model50 piecewise Humidity weights",
            "metric": "weighted WRMSE in compressed model target space with target multiplicity expansion",
        },
        "features": {
            "humidity_removed": True,
            "raw_columns": RAW_COLUMNS,
            "feature_engineering": "raw 12 + FE10 core logs/poly2/stat features + few interactions/ratios; fold selection keeps max 15",
            "max_selected_features": int(args.max_selected_features),
            "full_selected_features": full_selected_features,
            "clipping": clipping_profile,
        },
        "xgboost": {
            "optuna_trials": int(args.optuna_trials),
            "soft_timeout_per_trial_sec": int(args.optuna_timeout_per_trial_sec),
            "best_params": {key: float(value) if isinstance(value, float) else int(value) for key, value in best_params.items()},
            "oof_weighted_wrmse": float(xgb_score),
        },
        "model50_reference": {
            "oof_path": display_path(model50_oof_path),
            "oof_weighted_wrmse_recomputed": float(model50_score),
        },
        "correlation_vs_model50": corr_summary,
        "fold_reports": fold_reports,
        "artifacts": {key: display_path(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"XGB OOF WRMSE={xgb_score:.6f}; model50 OOF WRMSE={model50_score:.6f}")
    log(f"Mean targetwise correlation vs model50={corr_summary['mean']:.6f}")
    log(f"Submission written to {paths['submission']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
