from __future__ import annotations

import argparse
import json
import random
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
    engineer_env_focus_features,
    engineer_features,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


FEATURE_MODES = ["raw", "raw_no_env", "engineered", "engineered_no_env", "env_focus", "env_focus_no_env"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autopilot search for a strong blended model.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv and y_train.csv.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts_master_autopilot")
    parser.add_argument("--submission-prefix", default="master_autopilot")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument(
        "--autopilot-level",
        choices=["quick", "standard", "aggressive"],
        default="standard",
        help="Search budget profile.",
    )
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional hard cap (0 means profile default).")
    parser.add_argument("--heartbeat-seconds", type=int, default=60, help="Progress print interval.")
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


def build_feature_sets(x_train: pd.DataFrame, x_test: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    raw_train = raw_features(x_train)
    raw_test = raw_features(x_test)

    eng_train = engineer_features(x_train)
    eng_test = engineer_features(x_test)

    env_train = engineer_env_focus_features(x_train)
    env_test = engineer_env_focus_features(x_test)

    return {
        "raw": (raw_train, raw_test),
        "raw_no_env": (raw_train.drop(columns=["Env"]), raw_test.drop(columns=["Env"])),
        "engineered": (eng_train, eng_test),
        "engineered_no_env": (eng_train.drop(columns=["Env"]), eng_test.drop(columns=["Env"])),
        "env_focus": (env_train, env_test),
        "env_focus_no_env": (env_train.drop(columns=["Env"]), env_test.drop(columns=["Env"])),
    }


def normalize_weights(raw_weights: dict[str, float], active_names: list[str]) -> dict[str, float]:
    filt = {name: raw_weights[name] for name in active_names if name in raw_weights}
    total = float(sum(filt.values()))
    return {name: value / total for name, value in filt.items()}


def sample_candidate(rng: random.Random) -> dict[str, object]:
    feature_mode = rng.choice(FEATURE_MODES)

    rf = {
        "n_estimators": rng.choice([220, 300, 420, 520]),
        "max_depth": rng.choice([16, 20, 22, 26]),
        "min_samples_split": rng.choice([0.002, 0.003, 0.005, 0.008]),
        "min_samples_leaf": rng.choice([10, 15, 20, 30, 40]),
        "max_features": rng.choice([0.6, 0.7, 0.8]),
    }
    et = {
        "n_estimators": rng.choice([240, 320, 480, 640]),
        "max_depth": rng.choice([18, 22, 24, 28]),
        "min_samples_split": rng.choice([0.002, 0.003, 0.004, 0.006]),
        "min_samples_leaf": rng.choice([8, 12, 16, 24]),
        "max_features": rng.choice([0.6, 0.7, 0.8, 0.9]),
    }
    gb = {
        "n_estimators": rng.choice([100, 140, 180, 220]),
        "learning_rate": rng.choice([0.03, 0.04, 0.06, 0.08]),
        "max_depth": rng.choice([2, 3, 4]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
    }

    xgb = None
    if XGBOOST_AVAILABLE:
        xgb = {
            "n_estimators": rng.choice([180, 240, 320, 420]),
            "max_depth": rng.choice([4, 5, 6, 7]),
            "learning_rate": rng.choice([0.03, 0.05, 0.08]),
            "subsample": rng.choice([0.7, 0.8, 0.9]),
            "colsample_bytree": rng.choice([0.7, 0.8, 0.9]),
            "reg_alpha": rng.choice([1e-5, 1e-4, 1e-3]),
            "reg_lambda": rng.choice([0.5, 1.0, 2.0]),
        }

    weights = {
        "rf": rng.uniform(0.2, 0.6),
        "et": rng.uniform(0.2, 0.6),
        "gb": rng.uniform(0.05, 0.35),
        "xgb": rng.uniform(0.05, 0.35) if XGBOOST_AVAILABLE else 0.0,
    }

    return {
        "feature_mode": feature_mode,
        "rf": rf,
        "et": et,
        "gb": gb,
        "xgb": xgb,
        "weights": weights,
    }


def build_models(spec: dict[str, object], random_state: int) -> dict[str, object]:
    rf_params = {
        **spec["rf"],
        "random_state": random_state,
        "n_jobs": -1,
    }
    et_params = {
        **spec["et"],
        "random_state": random_state,
        "n_jobs": -1,
    }
    gb_params = {
        **spec["gb"],
        "random_state": random_state,
    }

    models: dict[str, object] = {
        "rf": RandomForestRegressor(**rf_params),
        "et": ExtraTreesRegressor(**et_params),
        "gb": MultiOutputRegressor(GradientBoostingRegressor(**gb_params)),
    }

    if spec["xgb"] is not None and XGBOOST_AVAILABLE:
        xgb_params = {
            **spec["xgb"],
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
        models["xgb"] = MultiOutputRegressor(XGBRegressor(**xgb_params), n_jobs=-1)

    return models


def profile_default_candidates(level: str) -> int:
    if level == "quick":
        return 8
    if level == "aggressive":
        return 48
    return 24


def evaluate_candidate(
    spec: dict[str, object],
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    holdout_fraction: float,
    random_state: int,
) -> tuple[float, float]:
    mode = str(spec["feature_mode"])
    X_train, _ = feature_sets[mode]

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=holdout_fraction,
        random_state=random_state,
    )

    models = build_models(spec, random_state=random_state)
    weights = normalize_weights(spec["weights"], list(models.keys()))

    pred_fit = None
    pred_valid = None
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        fit_part = np.asarray(model.predict(X_fit), dtype=float)
        valid_part = np.asarray(model.predict(X_valid), dtype=float)
        pred_fit = weights[name] * fit_part if pred_fit is None else pred_fit + weights[name] * fit_part
        pred_valid = weights[name] * valid_part if pred_valid is None else pred_valid + weights[name] * valid_part

    fit_full = enforce_known_constraints(pd.DataFrame(pred_fit, columns=train_target_columns, index=X_fit.index))
    valid_full = enforce_known_constraints(pd.DataFrame(pred_valid, columns=train_target_columns, index=X_valid.index))
    y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
    y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))

    train_rmse = competition_rmse(y_fit_full, fit_full)
    valid_rmse = competition_rmse(y_valid_full, valid_full)
    return float(train_rmse), float(valid_rmse)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    rng = random.Random(args.random_state)

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)
    y_train_full, train_target_columns = prepare_targets(data.y_train)

    feature_sets = build_feature_sets(data.x_train, data.x_test)

    n_candidates = args.max_candidates if args.max_candidates > 0 else profile_default_candidates(args.autopilot_level)
    specs = [sample_candidate(rng) for _ in range(n_candidates)]

    best_spec = None
    best_val = float("inf")
    best_train = float("inf")

    t0 = time.time()
    last_heartbeat = t0
    print(f"Autopilot search started: candidates={n_candidates}, level={args.autopilot_level}, xgboost={XGBOOST_AVAILABLE}")

    for i, spec in enumerate(specs, start=1):
        train_rmse, valid_rmse = evaluate_candidate(
            spec,
            feature_sets,
            y_train_full,
            train_target_columns,
            args.holdout_fraction,
            args.random_state,
        )

        elapsed = time.time() - t0
        print(f"[{i}/{n_candidates}] mode={spec['feature_mode']} train={train_rmse:.6f} val={valid_rmse:.6f} elapsed={elapsed/60:.1f}m")

        if valid_rmse < best_val:
            best_val = valid_rmse
            best_train = train_rmse
            best_spec = spec

        now = time.time()
        if now - last_heartbeat >= args.heartbeat_seconds:
            progress = 100.0 * i / n_candidates
            print(f"HEARTBEAT: progress={progress:.1f}% best_val={best_val:.6f} elapsed={elapsed/60:.1f}m")
            last_heartbeat = now

    if best_spec is None:
        raise RuntimeError("No candidate evaluated.")

    best_mode = str(best_spec["feature_mode"])
    X_train, X_test = feature_sets[best_mode]
    final_models = build_models(best_spec, random_state=args.random_state)
    final_weights = normalize_weights(best_spec["weights"], list(final_models.keys()))

    pred_train = None
    pred_test = None
    for name, model in final_models.items():
        model.fit(X_train, y_train_full)
        train_part = np.asarray(model.predict(X_train), dtype=float)
        test_part = np.asarray(model.predict(X_test), dtype=float)
        pred_train = final_weights[name] * train_part if pred_train is None else pred_train + final_weights[name] * train_part
        pred_test = final_weights[name] * test_part if pred_test is None else pred_test + final_weights[name] * test_part

    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train_full.assign(d15=0.0)),
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
        "task": "master_autopilot_multioutput_regression",
        "data_dir": str(data_dir),
        "xgboost_available": bool(XGBOOST_AVAILABLE),
        "autopilot_level": args.autopilot_level,
        "search_space_evaluated": n_candidates,
        "best_feature_mode": best_mode,
        "best_train_rmse": float(best_train),
        "best_validation_rmse": float(best_val),
        "full_train_rmse": float(full_train_rmse),
        "best_spec": {
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
        "submission_path": submission_path,
        "skip_submission": bool(args.skip_submission),
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
