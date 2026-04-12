from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, load_modeling_data  # noqa: E402
from train_fixed_et3_xgbraw_blender import (  # noqa: E402
    FIXED_ET_PARAMS,
    build_global_feature_pool,
    make_et_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit fixed et_allpool_3 on full train and predict X_test.")
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/18_fixed_et3_solo")
    parser.add_argument("--submission-prefix", default="fixed_et_allpool_3_solo")
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    return parser.parse_args()


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def fit_feature_matrices(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    *,
    ratio_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], int]:
    train_expanded = build_global_feature_pool(x_train_raw, ratio_eps=ratio_eps)
    total_feature_count = int(train_expanded.shape[1])
    constant_columns = [
        column for column in train_expanded.columns if train_expanded[column].nunique(dropna=False) <= 1
    ]
    selected_columns = [column for column in train_expanded.columns if column not in constant_columns]

    x_train_model = train_expanded[selected_columns].copy()
    x_test_model = build_global_feature_pool(x_test_raw, ratio_eps=ratio_eps)[selected_columns].copy()
    return x_train_model, x_test_model, selected_columns, constant_columns, total_feature_count


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_modeling_data(data_dir)
    data = bundle.data
    schema = bundle.schema
    x_train_raw = bundle.x_train_raw
    x_test_raw = bundle.x_test_raw
    y_train_model = bundle.y_train_model

    if args.max_train_rows is not None:
        x_train_raw = x_train_raw.iloc[: args.max_train_rows].copy()
        y_train_model = y_train_model.iloc[: args.max_train_rows].copy()

    if args.max_test_rows is not None:
        x_test_raw = x_test_raw.iloc[: args.max_test_rows].copy()
        x_test_ids = data.x_test["ID"].iloc[: args.max_test_rows].copy()
    else:
        x_test_ids = data.x_test["ID"].copy()

    log_progress("ET3 solo: building full feature pool")
    x_train_model, x_test_model, selected_columns, constant_columns, total_feature_count = fit_feature_matrices(
        x_train_raw,
        x_test_raw,
        ratio_eps=args.ratio_eps,
    )

    params = dict(FIXED_ET_PARAMS["et_allpool_3"])
    params["n_jobs"] = args.n_jobs

    log_progress(
        f"ET3 solo: fitting fixed et_allpool_3 on {len(x_train_model)} train rows with {x_train_model.shape[1]} features"
    )
    model = make_et_model(params)
    model.fit(x_train_model, y_train_model)

    log_progress(f"ET3 solo: predicting {len(x_test_model)} test rows")
    test_predictions_model = pd.DataFrame(
        model.predict(x_test_model),
        columns=y_train_model.columns,
        index=x_test_raw.index,
    ).astype("float32")
    test_predictions_full = schema.expand_predictions(test_predictions_model)
    submission = build_submission_frame(x_test_ids, test_predictions_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_path = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    summary = {
        "generated_at_utc": timestamp,
        "model": "fixed et_allpool_3 solo",
        "data_dir": str(data_dir),
        "params": params,
        "feature_pool": {
            "strategy": "all generated features, drop constants only",
            "total_feature_count": total_feature_count,
            "selected_feature_count": int(len(selected_columns)),
            "dropped_constant_feature_count": int(len(constant_columns)),
            "selected_columns": selected_columns,
        },
        "target_handling": {
            "modeled_targets": schema.model_targets,
            "constant_targets": schema.constant_targets,
        },
        "submission_path": str(submission_path.relative_to(ROOT)),
        "rows_predicted": int(len(submission)),
    }
    summary_path = output_dir / f"{args.submission_prefix}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log_progress(f"ET3 solo: submission written to {submission_path}")


if __name__ == "__main__":
    main()
