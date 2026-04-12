from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, engineer_features, infer_target_schema, load_competition_data  # noqa: E402
from odor_competition.metrics import competition_rmse  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF+XGB stacking with fixed Trial 4 parameters and export CSV.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_train.csv, X_test.csv, y_train.csv.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts_rf_xgb_stacking")
    parser.add_argument("--submission-prefix", default="submission_rf_xgb_stacking_trial4")
    parser.add_argument("--skip-submission", action="store_true")
    return parser.parse_args()


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


def generate_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_targets = y_train.shape[1]
    rf_oof = np.zeros((X_train.shape[0], n_targets), dtype=float)
    xgb_oof = np.zeros((X_train.shape[0], n_targets), dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]

        rf = RandomForestRegressor(
            n_estimators=TRIAL4_PARAMS["rf_n_estimators"],
            max_depth=TRIAL4_PARAMS["rf_max_depth"],
            min_samples_leaf=TRIAL4_PARAMS["rf_min_samples_leaf"],
            max_features=TRIAL4_PARAMS["rf_max_features"],
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_fold_train, y_fold_train)
        rf_oof[valid_idx] = rf.predict(X_fold_valid)

        xgb = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=TRIAL4_PARAMS["xgb_n_estimators"],
                max_depth=TRIAL4_PARAMS["xgb_max_depth"],
                learning_rate=TRIAL4_PARAMS["xgb_learning_rate"],
                subsample=TRIAL4_PARAMS["xgb_subsample"],
                colsample_bytree=TRIAL4_PARAMS["xgb_colsample_bytree"],
                reg_alpha=TRIAL4_PARAMS["xgb_reg_alpha"],
                reg_lambda=TRIAL4_PARAMS["xgb_reg_lambda"],
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                objective="reg:squarederror",
                verbosity=0,
            ),
            n_jobs=-1,
        )
        xgb.fit(X_fold_train, y_fold_train)
        xgb_oof[valid_idx] = xgb.predict(X_fold_valid)

    return rf_oof, xgb_oof


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)
    y_train, train_target_columns = prepare_targets(data.y_train)

    # Use engineered_no_env because Trial 4 came from this feature mode.
    X_train = engineer_features(data.x_train).drop(columns=["Env"])
    X_test = engineer_features(data.x_test).drop(columns=["Env"])

    rf_oof, xgb_oof = generate_oof_predictions(X_train, y_train, args.n_splits, args.random_state)
    meta_features = np.hstack([rf_oof, xgb_oof])

    meta_model = Ridge(alpha=TRIAL4_PARAMS["meta_alpha"])
    meta_model.fit(meta_features, y_train.values)

    # Fit base models on full data for test inference.
    rf_full = RandomForestRegressor(
        n_estimators=TRIAL4_PARAMS["rf_n_estimators"],
        max_depth=TRIAL4_PARAMS["rf_max_depth"],
        min_samples_leaf=TRIAL4_PARAMS["rf_min_samples_leaf"],
        max_features=TRIAL4_PARAMS["rf_max_features"],
        random_state=args.random_state,
        n_jobs=-1,
    )
    rf_full.fit(X_train, y_train)

    xgb_full = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=TRIAL4_PARAMS["xgb_n_estimators"],
            max_depth=TRIAL4_PARAMS["xgb_max_depth"],
            learning_rate=TRIAL4_PARAMS["xgb_learning_rate"],
            subsample=TRIAL4_PARAMS["xgb_subsample"],
            colsample_bytree=TRIAL4_PARAMS["xgb_colsample_bytree"],
            reg_alpha=TRIAL4_PARAMS["xgb_reg_alpha"],
            reg_lambda=TRIAL4_PARAMS["xgb_reg_lambda"],
            random_state=args.random_state,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        ),
        n_jobs=-1,
    )
    xgb_full.fit(X_train, y_train)

    train_pred_raw = meta_model.predict(meta_features)
    train_pred_df = pd.DataFrame(train_pred_raw, columns=train_target_columns, index=y_train.index)
    train_rmse = competition_rmse(
        enforce_known_constraints(y_train.assign(d15=0.0)),
        enforce_known_constraints(train_pred_df),
    )

    rf_test_pred = rf_full.predict(X_test)
    xgb_test_pred = xgb_full.predict(X_test)
    test_meta = np.hstack([rf_test_pred, xgb_test_pred])
    test_pred_raw = meta_model.predict(test_meta)
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
        "task": "rf_xgb_stacking_fixed_trial4",
        "feature_mode": "engineered_no_env",
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "diagnostic_scores": {"train_rmse": float(train_rmse)},
        "params": TRIAL4_PARAMS,
        "duplicate_target_groups": [group for group in schema.duplicate_groups if len(group) > 1],
        "constant_targets": schema.constant_targets,
        "submission_path": submission_path,
    }

    summary_file = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
