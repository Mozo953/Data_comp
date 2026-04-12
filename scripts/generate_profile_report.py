from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import engineer_features, load_competition_data  # noqa: E402
from odor_competition.reporting import build_ydata_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ydata-profiling HTML reports for the competition.")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing X_train.csv, X_test.csv, and y_train.csv.",
    )
    parser.add_argument(
        "--dataset",
        choices=["raw_train_targets", "engineered_train_features"],
        default="raw_train_targets",
        help="Profile the raw training features joined with targets, or the engineered training features.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50_000,
        help="Number of training rows to sample for the report.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for row sampling.",
    )
    parser.add_argument(
        "--with-interactions",
        action="store_true",
        help="Include the heavy scatter-matrix interaction section.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    data = load_competition_data(data_dir)

    sample_size = min(args.sample_size, len(data.x_train))
    sample_index = data.x_train.sample(sample_size, random_state=args.random_state).index

    if args.dataset == "raw_train_targets":
        profile_frame = (
            data.x_train.drop(columns=["ID"]).loc[sample_index].reset_index(drop=True).join(
                data.y_train.drop(columns=["ID"]).loc[sample_index].reset_index(drop=True)
            )
        )
        destination = ROOT / "reports" / "profile" / "train_correlation_profile.html"
        title = "Competition 1 Train Correlation Profile"
    else:
        profile_frame = engineer_features(data.x_train).loc[sample_index].reset_index(drop=True)
        destination = ROOT / "reports" / "profile" / "engineered_train_profile.html"
        title = "Competition 1 Engineered Train Profile"

    build_ydata_profile(
        profile_frame,
        destination,
        title,
        minimal=False,
        correlation_focus=True,
        explorative=True,
        with_interactions=args.with_interactions,
    )
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
