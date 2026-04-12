from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, engineer_features, infer_target_schema, load_competition_data  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


# Trial 4 baseline params (proven model).
TRIAL4_PARAMS = {
    "rf_n_estimators": 300,
    "rf_max_depth": 19,
    "rf_min_samples_leaf": 40,
    "rf_max_features": 0.8437832058402787,
    "xgb_n_estimators": 300,
    "xgb_max_depth": 8,
    "xgb_learning_rate": 0.09772699724544108,
    "xgb_subsample": 0.976562270506935,
    "xgb_colsample_bytree": 0.7265477506155759,
    "xgb_reg_alpha": 0.0002465844721448739,
    "xgb_reg_lambda": 0.6130682222763452,
    "meta_alpha": 0.09462175356461491,
}

# More conservative preset to reduce overfitting while keeping strong signal.
ANTIOVERFIT_PARAMS = {
    "rf_max_depth": 15,
    "rf_min_samples_leaf": 20,
    "rf_max_features": 0.75,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.04,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.8,
    "xgb_reg_alpha": 0.01,
    "xgb_reg_lambda": 2.0,
    "xgb_min_child_weight": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trial4 baseline + targeted interactions + multi-seed ensemble.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, y_train.csv.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--add-interactions", action="store_true", help="Add targeted interaction features.")
    parser.add_argument("--seed-ensemble", default="42,52,62", help="Comma-separated seeds for averaging.")
    parser.add_argument("--max-train-rows", type=int, default=0, help="If >0, train on a reproducible subset of rows.")
    parser.add_argument("--rf-n-estimators", type=int, default=TRIAL4_PARAMS["rf_n_estimators"])
    parser.add_argument("--xgb-n-estimators", type=int, default=TRIAL4_PARAMS["xgb_n_estimators"])
    parser.add_argument("--rf-jobs", type=int, default=4, help="Parallel workers for RandomForest.")
    parser.add_argument("--anti-overfit", action="store_true", help="Use stronger regularization preset.")
    parser.add_argument("--output-dir", default="artifacts_rf_xgb_stacking_improved")
    parser.add_argument("--submission-prefix", default="submission_rf_xgb_stacking_improved")
    parser.add_argument("--skip-submission", action="store_true")
    return parser.parse_args()


def parse_seed_ensemble(seed_text: str) -> list[int]:
    seeds = [int(chunk.strip()) for chunk in seed_text.split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("--seed-ensemble must contain at least one integer seed.")
    return seeds


def log_step(message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{now} UTC] {message}", flush=True)


def prepare_targets(y_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    target_frame = y_frame.drop(columns=["ID"]) if "ID" in y_frame.columns else y_frame.copy()
    train_targets = [column for column in target_frame.columns if column != "d15"]
    return target_frame[train_targets].copy(), train_targets


def enforce_known_constraints(predictions: pd.DataFrame) -> pd.DataFrame:
    constrained = predictions.copy()
    duplicate_groups = [["d02", "d07", "d20"], ["d03", "d04", "d22"], ["d05", "d06", "d21"]]
    for group in duplicate_groups:
        group_mean = constrained[group].mean(axis=1)
        for column in group:
            constrained[column] = group_mean
    constrained["d15"] = 0.0
    return constrained[[f"d{i:02d}" for i in range(1, 24)]]


def top_feature_candidates(X: pd.DataFrame, y: pd.DataFrame, top_k: int) -> list[str]:
    y_signal = y.mean(axis=1)
    scores = X.corrwith(y_signal).abs().fillna(0.0)
    ranked = scores.sort_values(ascending=False).index.tolist()
    return ranked[: min(top_k, len(ranked))]


def add_targeted_interactions(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                               top_k: int = 12, max_pairs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = top_feature_candidates(X_train, y_train, top_k)
    if len(candidates) < 2:
        return X_train.copy(), X_test.copy()

    out_train = X_train.copy()
    out_test = X_test.copy()

    pair_count = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            left, right = candidates[i], candidates[j]

            # Multiplicative interaction
            col_mul = f"int_mul__{left}__{right}"
            out_train[col_mul] = X_train[left] * X_train[right]
            out_test[col_mul] = X_test[left] * X_test[right]

            pair_count += 1
            if pair_count >= max_pairs:
                return out_train, out_test

    return out_train, out_test


def generate_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_splits: int,
    random_state: int,
    rf_n_estimators: int,
    xgb_n_estimators: int,
    rf_jobs: int,
    rf_params: dict[str, float],
    xgb_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    n_targets = y_train.shape[1]
    rf_oof = np.zeros((X_train.shape[0], n_targets), dtype=float)
    xgb_oof = np.zeros((X_train.shape[0], n_targets), dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_id, (train_idx, valid_idx) in enumerate(kf.split(X_train), start=1):
        log_step(f"OOF fold {fold_id}/{n_splits} start")
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]

        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=int(rf_params["max_depth"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            max_features=float(rf_params["max_features"]),
            random_state=random_state,
            n_jobs=rf_jobs,
        )
        rf.fit(X_fold_train, y_fold_train)
        rf_oof[valid_idx] = rf.predict(X_fold_valid)

        X_fold_train_np = np.asarray(X_fold_train, dtype=np.float32)
        y_fold_train_np = np.asarray(y_fold_train, dtype=np.float32)
        X_fold_valid_np = np.asarray(X_fold_valid, dtype=np.float32)

        xgb = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=xgb_n_estimators,
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
                reg_alpha=float(xgb_params["reg_alpha"]),
                reg_lambda=float(xgb_params["reg_lambda"]),
                min_child_weight=float(xgb_params["min_child_weight"]),
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
                objective="reg:squarederror",
                verbosity=0,
            ),
            n_jobs=1,
        )
        xgb.fit(X_fold_train_np, y_fold_train_np)
        xgb_oof[valid_idx] = xgb.predict(X_fold_valid_np)
        log_step(f"OOF fold {fold_id}/{n_splits} done")

    return rf_oof, xgb_oof


def select_blend_weight(
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    rf_oof: np.ndarray,
    xgb_oof: np.ndarray,
) -> tuple[float, float, float, float]:
    y_true_full = enforce_known_constraints(y_train.assign(d15=0.0))
    best_weight = 0.5
    best_score = float("inf")
    best_gap = 0.0

    for weight_rf in np.linspace(0.2, 0.8, 25):
        weight_xgb = 1.0 - weight_rf
        blend_oof = weight_rf * rf_oof + weight_xgb * xgb_oof
        blend_df = pd.DataFrame(blend_oof, columns=train_target_columns, index=y_train.index)
        score = competition_rmse(y_true_full, enforce_known_constraints(blend_df))
        if score < best_score:
            best_score = float(score)
            best_weight = float(weight_rf)

    blend_oof = best_weight * rf_oof + (1.0 - best_weight) * xgb_oof
    blend_df = pd.DataFrame(blend_oof, columns=train_target_columns, index=y_train.index)
    train_rmse = competition_rmse(y_true_full, enforce_known_constraints(blend_df))

    rf_df = pd.DataFrame(rf_oof, columns=train_target_columns, index=y_train.index)
    xgb_df = pd.DataFrame(xgb_oof, columns=train_target_columns, index=y_train.index)
    rf_rmse = competition_rmse(y_true_full, enforce_known_constraints(rf_df))
    xgb_rmse = competition_rmse(y_true_full, enforce_known_constraints(xgb_df))
    best_gap = float(abs(rf_rmse - xgb_rmse))

    return best_weight, float(train_rmse), float(rf_rmse), float(xgb_rmse)


def main() -> None:
    args = parse_args()
    seed_ensemble = parse_seed_ensemble(args.seed_ensemble)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)
    y_train, train_target_columns = prepare_targets(data.y_train)

    X_train = engineer_features(data.x_train).drop(columns=["Env"])
    X_test = engineer_features(data.x_test).drop(columns=["Env"])

    if args.max_train_rows > 0 and args.max_train_rows < len(X_train):
        sampled_idx = X_train.sample(n=args.max_train_rows, random_state=args.random_state).index
        X_train = X_train.loc[sampled_idx].copy()
        y_train = y_train.loc[sampled_idx].copy()
        log_step(f"Training on sampled subset: {len(X_train)} rows")

    log_step(f"Prepared features train={X_train.shape}, test={X_test.shape}")

    if args.add_interactions:
        X_train, X_test = add_targeted_interactions(X_train, X_test, y_train, top_k=10, max_pairs=8)
        log_step(f"Interactions added, train={X_train.shape}, test={X_test.shape}")

    if args.anti_overfit:
        rf_params = {
            "max_depth": ANTIOVERFIT_PARAMS["rf_max_depth"],
            "min_samples_leaf": ANTIOVERFIT_PARAMS["rf_min_samples_leaf"],
            "max_features": ANTIOVERFIT_PARAMS["rf_max_features"],
        }
        xgb_params = {
            "max_depth": ANTIOVERFIT_PARAMS["xgb_max_depth"],
            "learning_rate": ANTIOVERFIT_PARAMS["xgb_learning_rate"],
            "subsample": ANTIOVERFIT_PARAMS["xgb_subsample"],
            "colsample_bytree": ANTIOVERFIT_PARAMS["xgb_colsample_bytree"],
            "reg_alpha": ANTIOVERFIT_PARAMS["xgb_reg_alpha"],
            "reg_lambda": ANTIOVERFIT_PARAMS["xgb_reg_lambda"],
            "min_child_weight": ANTIOVERFIT_PARAMS["xgb_min_child_weight"],
        }
        log_step("Using anti-overfit preset")
    else:
        rf_params = {
            "max_depth": TRIAL4_PARAMS["rf_max_depth"],
            "min_samples_leaf": TRIAL4_PARAMS["rf_min_samples_leaf"],
            "max_features": TRIAL4_PARAMS["rf_max_features"],
        }
        xgb_params = {
            "max_depth": TRIAL4_PARAMS["xgb_max_depth"],
            "learning_rate": TRIAL4_PARAMS["xgb_learning_rate"],
            "subsample": TRIAL4_PARAMS["xgb_subsample"],
            "colsample_bytree": TRIAL4_PARAMS["xgb_colsample_bytree"],
            "reg_alpha": TRIAL4_PARAMS["xgb_reg_alpha"],
            "reg_lambda": TRIAL4_PARAMS["xgb_reg_lambda"],
            "min_child_weight": 1.0,
        }

    # OOF predictions used to tune blend weight for better generalization.
    rf_oof, xgb_oof = generate_oof_predictions(
        X_train,
        y_train,
        args.n_splits,
        args.random_state,
        args.rf_n_estimators,
        args.xgb_n_estimators,
        args.rf_jobs,
        rf_params,
        xgb_params,
    )

    weight_rf, oof_blend_rmse, rf_oof_rmse, xgb_oof_rmse = select_blend_weight(
        y_train,
        train_target_columns,
        rf_oof,
        xgb_oof,
    )
    weight_xgb = 1.0 - weight_rf
    log_step(f"Selected OOF blend weights rf={weight_rf:.3f}, xgb={weight_xgb:.3f}")

    # Train final models on full data.
    rf_full = RandomForestRegressor(
        n_estimators=args.rf_n_estimators,
        max_depth=int(rf_params["max_depth"]),
        min_samples_leaf=int(rf_params["min_samples_leaf"]),
        max_features=float(rf_params["max_features"]),
        random_state=args.random_state,
        n_jobs=args.rf_jobs,
    )
    rf_full.fit(X_train, y_train)
    log_step("Full RF trained")

    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    X_test_np = np.asarray(X_test, dtype=np.float32)

    xgb_full = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
            min_child_weight=float(xgb_params["min_child_weight"]),
            random_state=args.random_state,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        ),
        n_jobs=1,
    )
    xgb_full.fit(X_train_np, y_train_np)
    log_step("Full XGB trained")

    train_pred_raw = weight_rf * rf_oof + weight_xgb * xgb_oof
    train_pred_df = pd.DataFrame(train_pred_raw, columns=train_target_columns, index=y_train.index)
    train_rmse = competition_rmse(
        enforce_known_constraints(y_train.assign(d15=0.0)),
        enforce_known_constraints(train_pred_df),
    )

    # Multi-seed ensemble for test predictions.
    test_preds_ensemble = []
    for seed in seed_ensemble:
        log_step(f"Seed ensemble model start: {seed}")
        rf_seed = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=int(rf_params["max_depth"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            max_features=float(rf_params["max_features"]),
            random_state=seed,
            n_jobs=args.rf_jobs,
        )
        rf_seed.fit(X_train, y_train)
        rf_test_pred = rf_seed.predict(X_test)

        xgb_seed = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=args.xgb_n_estimators,
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
                reg_alpha=float(xgb_params["reg_alpha"]),
                reg_lambda=float(xgb_params["reg_lambda"]),
                min_child_weight=float(xgb_params["min_child_weight"]),
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                objective="reg:squarederror",
                verbosity=0,
            ),
            n_jobs=1,
        )
        xgb_seed.fit(X_train_np, y_train_np)
        xgb_test_pred = xgb_seed.predict(X_test_np)

        ens_pred = weight_rf * rf_test_pred + weight_xgb * xgb_test_pred
        test_preds_ensemble.append(ens_pred)
        log_step(f"Seed ensemble model done: {seed}")

    test_pred_raw = np.mean(np.stack(test_preds_ensemble, axis=0), axis=0)
    test_pred_df = pd.DataFrame(test_pred_raw, columns=train_target_columns, index=X_test.index)
    final_test = enforce_known_constraints(test_pred_df)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        submission = build_submission_frame(data.x_test["ID"], final_test)
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "rf_xgb_stacking_improved",
        "feature_mode": "engineered_no_env",
        "add_interactions": args.add_interactions,
        "n_cv_splits": args.n_splits,
        "seed_ensemble": seed_ensemble,
        "max_train_rows": int(args.max_train_rows),
        "rf_n_estimators": int(args.rf_n_estimators),
        "xgb_n_estimators": int(args.xgb_n_estimators),
        "rf_jobs": int(args.rf_jobs),
        "anti_overfit": bool(args.anti_overfit),
        "blend_weights": {"rf": float(weight_rf), "xgb": float(weight_xgb)},
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "diagnostic_scores": {
            "oof_blend_rmse": float(oof_blend_rmse),
            "oof_rf_rmse": float(rf_oof_rmse),
            "oof_xgb_rmse": float(xgb_oof_rmse),
            "train_blend_rmse": float(train_rmse),
        },
        "params": {
            "rf": rf_params,
            "xgb": xgb_params,
        },
        "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
        "constant_targets": schema.constant_targets,
        "submission_path": submission_path,
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
