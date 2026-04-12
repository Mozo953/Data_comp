from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import build_submission_frame, load_competition_data, raw_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a test submission from a saved Trial 11 RandomForest model.")
    parser.add_argument("--data-dir", default=".", help="Directory containing X_test.csv and related data.")
    parser.add_argument("--model-path", required=True, help="Path to the saved .joblib model artifact.")
    parser.add_argument("--output-dir", default="artifacts_final_rf500_trial11", help="Directory for the CSV output.")
    parser.add_argument("--submission-prefix", default="final_rf500_trial11_infer", help="Prefix for the output file.")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    artifact = joblib.load(Path(args.model_path))
    model = artifact["model"]

    data = load_competition_data(data_dir)
    X_test = raw_features(data.x_test)
    test_pred = pd.DataFrame(model.predict(X_test), columns=artifact["train_target_columns"], index=X_test.index)
    final_predictions = enforce_known_constraints(test_pred)
    submission = build_submission_frame(data.x_test["ID"], final_predictions)

    timestamp = artifact.get("generated_at_utc", "unknown")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
    submission.to_csv(output_file, index=False)

    print(json.dumps({"output_file": str(output_file.relative_to(ROOT)), "rows": len(submission)}, indent=2))


if __name__ == "__main__":
    main()