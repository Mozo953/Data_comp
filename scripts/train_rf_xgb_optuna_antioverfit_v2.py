from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, engineer_features, load_competition_data, raw_features  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF+XGB Optuna with anti-overfit CV and targeted interactions.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, y_train.csv.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--overfit-penalty", type=float, default=0.6, help="Penalty multiplier on fold overfit gap.")
    parser.add_argument("--variance-penalty", type=float, default=0.6, help="Penalty multiplier on fold RMSE std.")
    parser.add_argument("--seed-ensemble", default="42,52,62", help="Comma-separated seeds for final prediction averaging.")
    parser.add_argument("--output-dir", default="artifacts_rf_xgb_optuna_antioverfit_v2")
    parser.add_argument("--submission-prefix", default="rf_xgb_optuna_antioverfit_v2")
    parser.add_argument("--skip-submission", action="store_true")
    return parser.parse_args()


def parse_seed_ensemble(seed_text: str) -> list[int]:
    seeds = [int(chunk.strip()) for chunk in seed_text.split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("--seed-ensemble must contain at least one integer seed.")
    return seeds


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

    return {
        "raw_no_env": (raw_train.drop(columns=["Env"]), raw_test.drop(columns=["Env"])),
        "engineered_no_env": (eng_train.drop(columns=["Env"]), eng_test.drop(columns=["Env"])),
        "engineered_with_env": (eng_train, eng_test),
    }


def top_feature_candidates(X: pd.DataFrame, y: pd.DataFrame, top_k: int) -> list[str]:
    y_signal = y.mean(axis=1)
    scores = X.corrwith(y_signal).abs().fillna(0.0)
    ranked = scores.sort_values(ascending=False).index.tolist()
    return ranked[: min(top_k, len(ranked))]


def add_targeted_interactions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    top_k: int,
    max_pairs: int,
    interaction_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if top_k < 2 or max_pairs <= 0 or interaction_mode == "none":
        return X_train.copy(), X_test.copy(), []

    candidates = top_feature_candidates(X_train, y_train, top_k)
    if len(candidates) < 2:
        return X_train.copy(), X_test.copy(), []

    out_train = X_train.copy()
    out_test = X_test.copy()
    created: list[str] = []

    pair_count = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            left = candidates[i]
            right = candidates[j]

            if interaction_mode in {"mul", "both"}:
                col = f"int_mul__{left}__{right}"
                out_train[col] = X_train[left] * X_train[right]
                out_test[col] = X_test[left] * X_test[right]
                created.append(col)

            if interaction_mode in {"diff", "both"}:
                col = f"int_diff__{left}__{right}"
                out_train[col] = X_train[left] - X_train[right]
                out_test[col] = X_test[left] - X_test[right]
                created.append(col)

            pair_count += 1
            if pair_count >= max_pairs:
                return out_train, out_test, created

    return out_train, out_test, created


def fit_models_and_blend(
    X_fit: pd.DataFrame,
    y_fit: pd.DataFrame,
    X_eval: pd.DataFrame,
    rf_params: dict,
    xgb_params: dict,
    w_rf: float,
    random_state: int,
) -> np.ndarray:
    rf = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)
    rf.fit(X_fit, y_fit)
    pred_rf = np.asarray(rf.predict(X_eval), dtype=float)

    # Keep XGBoost execution stable on Windows by avoiding nested parallel workers.
    X_fit_np = np.asarray(X_fit, dtype=np.float32)
    y_fit_np = np.asarray(y_fit, dtype=np.float32)
    X_eval_np = np.asarray(X_eval, dtype=np.float32)

    xgb = MultiOutputRegressor(
        XGBRegressor(
            **xgb_params,
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        ),
        n_jobs=1,
    )
    xgb.fit(X_fit_np, y_fit_np)
    pred_xgb = np.asarray(xgb.predict(X_eval_np), dtype=float)

    return w_rf * pred_rf + (1.0 - w_rf) * pred_xgb


def evaluate_params_cv(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_target_columns: list[str],
    rf_params: dict,
    xgb_params: dict,
    w_rf: float,
    *,
    cv_splits: int,
    cv_repeats: int,
    random_state: int,
    overfit_penalty: float,
    variance_penalty: float,
) -> dict[str, float]:
    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)

    fold_valid_scores: list[float] = []
    fold_train_scores: list[float] = []
    fold_gaps: list[float] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(rkf.split(X_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train.iloc[fit_idx]
        y_valid = y_train.iloc[valid_idx]

        # Fit once per fold, then score train/valid to avoid doubling runtime.
        rf = RandomForestRegressor(**rf_params, random_state=random_state + fold_idx, n_jobs=-1)
        rf.fit(X_fit, y_fit)
        pred_fit_rf = np.asarray(rf.predict(X_fit), dtype=float)
        pred_valid_rf = np.asarray(rf.predict(X_valid), dtype=float)

        X_fit_np = np.asarray(X_fit, dtype=np.float32)
        y_fit_np = np.asarray(y_fit, dtype=np.float32)
        X_valid_np = np.asarray(X_valid, dtype=np.float32)
        xgb = MultiOutputRegressor(
            XGBRegressor(
                **xgb_params,
                random_state=random_state + fold_idx,
                n_jobs=1,
                tree_method="hist",
                objective="reg:squarederror",
                verbosity=0,
            ),
            n_jobs=1,
        )
        xgb.fit(X_fit_np, y_fit_np)
        pred_fit_xgb = np.asarray(xgb.predict(X_fit_np), dtype=float)
        pred_valid_xgb = np.asarray(xgb.predict(X_valid_np), dtype=float)

        pred_fit = w_rf * pred_fit_rf + (1.0 - w_rf) * pred_fit_xgb
        pred_valid = w_rf * pred_valid_rf + (1.0 - w_rf) * pred_valid_xgb

        fit_df = pd.DataFrame(pred_fit, columns=train_target_columns, index=X_fit.index)
        valid_df = pd.DataFrame(pred_valid, columns=train_target_columns, index=X_valid.index)

        y_fit_full = enforce_known_constraints(y_fit.assign(d15=0.0))
        y_valid_full = enforce_known_constraints(y_valid.assign(d15=0.0))
        fit_full = enforce_known_constraints(fit_df)
        valid_full = enforce_known_constraints(valid_df)

        train_rmse = competition_rmse(y_fit_full, fit_full)
        valid_rmse = competition_rmse(y_valid_full, valid_full)
        gap = max(0.0, valid_rmse - train_rmse)

        fold_train_scores.append(train_rmse)
        fold_valid_scores.append(valid_rmse)
        fold_gaps.append(gap)

    mean_valid = float(np.mean(fold_valid_scores))
    std_valid = float(np.std(fold_valid_scores))
    mean_gap = float(np.mean(fold_gaps))
    mean_train = float(np.mean(fold_train_scores))

    objective = mean_valid + overfit_penalty * mean_gap + variance_penalty * std_valid
    return {
        "objective": float(objective),
        "mean_valid_rmse": mean_valid,
        "std_valid_rmse": std_valid,
        "mean_train_rmse": mean_train,
        "mean_overfit_gap": mean_gap,
    }


def main() -> None:
    args = parse_args()
    seed_ensemble = parse_seed_ensemble(args.seed_ensemble)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    y_train_full, train_target_columns = prepare_targets(data.y_train)
    feature_sets = build_feature_sets(data.x_train, data.x_test)

    def objective(trial: optuna.Trial) -> float:
        feature_mode = trial.suggest_categorical("feature_mode", list(feature_sets.keys()))
        X_train_base, X_test_base = feature_sets[feature_mode]

        interaction_mode = trial.suggest_categorical("interaction_mode", ["none", "mul", "diff", "both"])
        interaction_top_k = trial.suggest_int("interaction_top_k", 6, 14, step=2)
        interaction_max_pairs = trial.suggest_int("interaction_max_pairs", 4, 18, step=2)

        X_train, _, created_interactions = add_targeted_interactions(
            X_train_base,
            X_test_base,
            y_train_full,
            top_k=interaction_top_k,
            max_pairs=interaction_max_pairs,
            interaction_mode=interaction_mode,
        )

        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 220, 520, step=60),
            "max_depth": trial.suggest_int("rf_max_depth", 12, 24),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 8, 40),
            "min_samples_split": trial.suggest_float("rf_min_samples_split", 0.002, 0.015),
            "max_features": trial.suggest_float("rf_max_features", 0.55, 0.95),
        }

        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 180, 480, step=60),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.02, 0.12),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.55, 0.95),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 0.5, log=True),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.4, 5.0),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 1.5),
            "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1.0, 12.0),
        }

        w_rf = trial.suggest_float("weight_rf", 0.15, 0.85)

        cv_result = evaluate_params_cv(
            X_train,
            y_train_full,
            train_target_columns,
            rf_params,
            xgb_params,
            w_rf,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            random_state=args.random_state,
            overfit_penalty=args.overfit_penalty,
            variance_penalty=args.variance_penalty,
        )

        trial.set_user_attr("mean_valid_rmse", float(cv_result["mean_valid_rmse"]))
        trial.set_user_attr("std_valid_rmse", float(cv_result["std_valid_rmse"]))
        trial.set_user_attr("mean_train_rmse", float(cv_result["mean_train_rmse"]))
        trial.set_user_attr("mean_overfit_gap", float(cv_result["mean_overfit_gap"]))
        trial.set_user_attr("n_interactions", int(len(created_interactions)))

        return float(cv_result["objective"])

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=0),
        study_name="rf_xgb_antioverfit_v2",
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_feature_mode = best.params["feature_mode"]
    X_train_base, X_test_base = feature_sets[best_feature_mode]

    X_train, X_test, created_interactions = add_targeted_interactions(
        X_train_base,
        X_test_base,
        y_train_full,
        top_k=int(best.params["interaction_top_k"]),
        max_pairs=int(best.params["interaction_max_pairs"]),
        interaction_mode=str(best.params["interaction_mode"]),
    )

    rf_params = {
        "n_estimators": int(best.params["rf_n_estimators"]),
        "max_depth": int(best.params["rf_max_depth"]),
        "min_samples_leaf": int(best.params["rf_min_samples_leaf"]),
        "min_samples_split": float(best.params["rf_min_samples_split"]),
        "max_features": float(best.params["rf_max_features"]),
    }
    xgb_params = {
        "n_estimators": int(best.params["xgb_n_estimators"]),
        "max_depth": int(best.params["xgb_max_depth"]),
        "learning_rate": float(best.params["xgb_learning_rate"]),
        "subsample": float(best.params["xgb_subsample"]),
        "colsample_bytree": float(best.params["xgb_colsample_bytree"]),
        "reg_alpha": float(best.params["xgb_reg_alpha"]),
        "reg_lambda": float(best.params["xgb_reg_lambda"]),
        "gamma": float(best.params["xgb_gamma"]),
        "min_child_weight": float(best.params["xgb_min_child_weight"]),
    }
    w_rf = float(best.params["weight_rf"])

    cv_result_best = evaluate_params_cv(
        X_train,
        y_train_full,
        train_target_columns,
        rf_params,
        xgb_params,
        w_rf,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        random_state=args.random_state,
        overfit_penalty=args.overfit_penalty,
        variance_penalty=args.variance_penalty,
    )

    train_preds_ensemble = []
    test_preds_ensemble = []
    for seed in seed_ensemble:
        train_pred = fit_models_and_blend(X_train, y_train_full, X_train, rf_params, xgb_params, w_rf, seed)
        test_pred = fit_models_and_blend(X_train, y_train_full, X_test, rf_params, xgb_params, w_rf, seed)
        train_preds_ensemble.append(train_pred)
        test_preds_ensemble.append(test_pred)

    pred_train_full = np.mean(np.stack(train_preds_ensemble, axis=0), axis=0)
    pred_test_full = np.mean(np.stack(test_preds_ensemble, axis=0), axis=0)

    full_train_rmse = competition_rmse(
        enforce_known_constraints(y_train_full.assign(d15=0.0)),
        enforce_known_constraints(pd.DataFrame(pred_train_full, columns=train_target_columns, index=X_train.index)),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = None
    if not args.skip_submission:
        test_pred_df = pd.DataFrame(pred_test_full, columns=train_target_columns, index=X_test.index)
        submission = build_submission_frame(data.x_test["ID"], enforce_known_constraints(test_pred_df))
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "task": "rf_xgb_optuna_antioverfit_v2",
        "data_dir": str(data_dir),
        "n_trials": args.n_trials,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "overfit_penalty": args.overfit_penalty,
        "variance_penalty": args.variance_penalty,
        "seed_ensemble": seed_ensemble,
        "best_feature_mode": best_feature_mode,
        "best_objective": float(best.value),
        "best_trial": int(best.number),
        "best_trial_metrics": {
            "mean_train_rmse": float(best.user_attrs.get("mean_train_rmse", np.nan)),
            "mean_valid_rmse": float(best.user_attrs.get("mean_valid_rmse", np.nan)),
            "std_valid_rmse": float(best.user_attrs.get("std_valid_rmse", np.nan)),
            "mean_overfit_gap": float(best.user_attrs.get("mean_overfit_gap", np.nan)),
            "n_interactions": int(best.user_attrs.get("n_interactions", 0)),
        },
        "selected_params": {
            "rf": rf_params,
            "xgb": xgb_params,
            "weight_rf": w_rf,
            "weight_xgb": 1.0 - w_rf,
            "interaction_mode": best.params["interaction_mode"],
            "interaction_top_k": int(best.params["interaction_top_k"]),
            "interaction_max_pairs": int(best.params["interaction_max_pairs"]),
        },
        "created_interaction_features": created_interactions,
        "feature_shape_final": list(X_train.shape),
        "diagnostic_scores": {
            "cv_mean_valid_rmse": float(cv_result_best["mean_valid_rmse"]),
            "cv_std_valid_rmse": float(cv_result_best["std_valid_rmse"]),
            "cv_mean_overfit_gap": float(cv_result_best["mean_overfit_gap"]),
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
