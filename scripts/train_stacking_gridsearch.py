from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    engineer_features,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast stacking blender via basic grid search.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, and y_train.csv.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts_stacking_grid")
    parser.add_argument("--submission-prefix", default="stacking_grid")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Use reduced search space and sampled train rows.")
    parser.add_argument("--pilot-sample-size", type=int, default=70000)
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional hard cap on tested candidates (0 = all).")
    args = parser.parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be between 0 and 1.")
    return args


def prepare_targets(y_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target_frame = y_frame.drop(columns=["ID"]) if "ID" in y_frame.columns else y_frame.copy()
    train_targets = [column for column in target_frame.columns if column != "d15"]
    return target_frame[train_targets].copy(), train_targets


def enforce_known_constraints(predictions: pd.DataFrame) -> pd.DataFrame:
    constrained = predictions.copy()
    duplicate_groups = [
        ["d02", "d07", "d20"],
        ["d03", "d04", "d22"],
        ["d05", "d06", "d21"],
    ]
    for group in duplicate_groups:
        group_mean = constrained[group].mean(axis=1)
        for column in group:
            constrained[column] = group_mean
    constrained["d15"] = 0.0
    return constrained[[f"d{i:02d}" for i in range(1, 24)]]


def make_feature_sets(x_train: pd.DataFrame, x_test: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    raw_train = raw_features(x_train)
    raw_test = raw_features(x_test)

    eng_train = engineer_features(x_train)
    eng_test = engineer_features(x_test)

    return {
        "raw": (raw_train, raw_test),
        "raw_no_env": (raw_train.drop(columns=["Env"]), raw_test.drop(columns=["Env"])),
        "engineered": (eng_train, eng_test),
        "engineered_no_env": (eng_train.drop(columns=["Env"]), eng_test.drop(columns=["Env"])),
    }


def build_models(rf_params: dict, et_params: dict, gb_params: dict, xgb_params: dict | None) -> dict[str, object]:
    models: dict[str, object] = {
        "rf": RandomForestRegressor(**rf_params),
        "et": ExtraTreesRegressor(**et_params),
        "gb": MultiOutputRegressor(GradientBoostingRegressor(**gb_params)),
    }
    if xgb_params is not None and XGBOOST_AVAILABLE:
        models["xgb"] = MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=-1)
    return models


def normalize_weights(weights: dict[str, float], model_names: list[str]) -> dict[str, float]:
    filtered = {name: weights[name] for name in model_names if name in weights}
    total = float(sum(filtered.values()))
    return {name: value / total for name, value in filtered.items()}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    feature_sets = make_feature_sets(data.x_train, data.x_test)

    # Basic grid (deliberately small for pilot reliability).
    rf_grid = [
        {"n_estimators": 220, "max_depth": 18, "min_samples_split": 0.005, "min_samples_leaf": 20, "max_features": 0.7, "random_state": args.random_state, "n_jobs": -1},
        {"n_estimators": 320, "max_depth": 22, "min_samples_split": 0.003, "min_samples_leaf": 15, "max_features": 0.7, "random_state": args.random_state, "n_jobs": -1},
    ]
    et_grid = [
        {"n_estimators": 260, "max_depth": 20, "min_samples_split": 0.004, "min_samples_leaf": 12, "max_features": 0.8, "random_state": args.random_state, "n_jobs": -1},
        {"n_estimators": 360, "max_depth": 24, "min_samples_split": 0.003, "min_samples_leaf": 10, "max_features": 0.7, "random_state": args.random_state, "n_jobs": -1},
    ]
    gb_grid = [
        {"n_estimators": 120, "learning_rate": 0.06, "max_depth": 3, "subsample": 0.9, "random_state": args.random_state},
        {"n_estimators": 180, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.8, "random_state": args.random_state},
    ]

    xgb_grid = [
        {
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1e-4,
            "reg_lambda": 1.0,
            "random_state": args.random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
    ]

    weight_grid = [
        {"rf": 0.40, "et": 0.35, "gb": 0.25, "xgb": 0.0},
        {"rf": 0.45, "et": 0.30, "gb": 0.15, "xgb": 0.10},
        {"rf": 0.35, "et": 0.30, "gb": 0.15, "xgb": 0.20},
    ]

    feature_modes = ["raw", "raw_no_env", "engineered", "engineered_no_env"]

    if args.pilot:
        rf_grid = rf_grid[:1]
        et_grid = et_grid[:1]
        gb_grid = gb_grid[:1]
        weight_grid = weight_grid[:2]
        feature_modes = ["raw", "raw_no_env"]

    candidate_specs: list[dict[str, object]] = []
    for feature_mode in feature_modes:
        for rf_params in rf_grid:
            for et_params in et_grid:
                for gb_params in gb_grid:
                    for weights in weight_grid:
                        if XGBOOST_AVAILABLE:
                            for xgb_params in xgb_grid:
                                candidate_specs.append(
                                    {
                                        "feature_mode": feature_mode,
                                        "rf": rf_params,
                                        "et": et_params,
                                        "gb": gb_params,
                                        "xgb": xgb_params,
                                        "weights": weights,
                                    }
                                )
                        else:
                            candidate_specs.append(
                                {
                                    "feature_mode": feature_mode,
                                    "rf": rf_params,
                                    "et": et_params,
                                    "gb": gb_params,
                                    "xgb": None,
                                    "weights": weights,
                                }
                            )

    if args.max_candidates > 0:
        candidate_specs = candidate_specs[: args.max_candidates]

    best_score = float("inf")
    best_spec: dict[str, object] | None = None
    last_heartbeat = time.time()
    t0 = time.time()

    total_candidates = len(candidate_specs)
    print(f"Grid search started with {total_candidates} candidates. Pilot={args.pilot}")

    for idx, spec in enumerate(candidate_specs, start=1):
        feature_mode = str(spec["feature_mode"])
        X_train_all, X_test_all = feature_sets[feature_mode]
        y_all = y_train_full

        if args.pilot and len(X_train_all) > args.pilot_sample_size:
            sampled = X_train_all.sample(args.pilot_sample_size, random_state=args.random_state).index
            X_train_all = X_train_all.loc[sampled]
            y_all = y_all.loc[sampled]

        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train_all,
            y_all,
            test_size=args.holdout_fraction,
            random_state=args.random_state,
        )

        models = build_models(
            rf_params=dict(spec["rf"]),
            et_params=dict(spec["et"]),
            gb_params=dict(spec["gb"]),
            xgb_params=spec["xgb"],
        )

        weights = normalize_weights(dict(spec["weights"]), list(models.keys()))

        pred_valid = None
        for name, model in models.items():
            model.fit(X_fit, y_fit)
            model_pred = np.asarray(model.predict(X_valid), dtype=float)
            weighted = weights[name] * model_pred
            pred_valid = weighted if pred_valid is None else pred_valid + weighted

        valid_df = pd.DataFrame(pred_valid, columns=train_target_columns, index=X_valid.index)
        valid_full = enforce_known_constraints(valid_df)
        y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))
        score = competition_rmse(y_valid_full, valid_full)

        elapsed = time.time() - t0
        print(f"[{idx}/{total_candidates}] mode={feature_mode} score={score:.6f} elapsed={elapsed/60:.1f}m")

        if score < best_score:
            best_score = score
            best_spec = spec

        now = time.time()
        if now - last_heartbeat >= 60:
            progress = 100.0 * idx / total_candidates
            print(f"HEARTBEAT: progress={progress:.1f}% best_score={best_score:.6f} elapsed={elapsed/60:.1f}m")
            last_heartbeat = now

    if best_spec is None:
        raise RuntimeError("No candidate evaluated.")

    best_mode = str(best_spec["feature_mode"])
    X_train, X_test = feature_sets[best_mode]
    y_train = y_train_full

    if args.pilot and len(X_train) > args.pilot_sample_size:
        sampled = X_train.sample(args.pilot_sample_size, random_state=args.random_state).index
        X_train = X_train.loc[sampled]
        y_train = y_train.loc[sampled]

    final_models = build_models(
        rf_params=dict(best_spec["rf"]),
        et_params=dict(best_spec["et"]),
        gb_params=dict(best_spec["gb"]),
        xgb_params=best_spec["xgb"],
    )
    final_weights = normalize_weights(dict(best_spec["weights"]), list(final_models.keys()))

    pred_train = None
    pred_test = None
    for name, model in final_models.items():
        model.fit(X_train, y_train)
        train_part = np.asarray(model.predict(X_train), dtype=float)
        test_part = np.asarray(model.predict(X_test), dtype=float)
        pred_train = final_weights[name] * train_part if pred_train is None else pred_train + final_weights[name] * train_part
        pred_test = final_weights[name] * test_part if pred_test is None else pred_test + final_weights[name] * test_part

    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train.assign(d15=0.0)),
        enforce_known_constraints(pd.DataFrame(pred_train, columns=train_target_columns, index=X_train.index)),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        final_predictions = enforce_known_constraints(pd.DataFrame(pred_test, columns=train_target_columns, index=X_test.index))
        submission = build_submission_frame(data.x_test["ID"], final_predictions)
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "stacking_gridsearch_multioutput_regression",
        "data_dir": str(data_dir),
        "pilot_mode": bool(args.pilot),
        "xgboost_available": bool(XGBOOST_AVAILABLE),
        "search_space_size": total_candidates,
        "best_score_validation": float(best_score),
        "best_spec": {
            "feature_mode": best_mode,
            "rf": best_spec["rf"],
            "et": best_spec["et"],
            "gb": best_spec["gb"],
            "xgb": best_spec["xgb"],
            "weights": final_weights,
        },
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
        "constant_targets": schema.constant_targets,
        "target_strategy": {
            "trained_targets": train_target_columns,
            "d15_handling": "removed_from_training_and_reinserted_as_zero",
            "duplicate_groups_handling": "predictions_averaged_within_known_duplicate_groups_before_export",
        },
        "diagnostic_scores": {
            "full_train_rmse": float(full_train_rmse),
        },
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
