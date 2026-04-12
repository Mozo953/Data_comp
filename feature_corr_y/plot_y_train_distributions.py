#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import load_competition_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the distribution of each Y_train target with seaborn histplot."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT,
        help="Directory containing X_train.csv, X_test.csv and y_train.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "feature_corr_y" / "plots_y_train",
        help="Directory where histograms are saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for the distribution histogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_competition_data(args.data_dir)

    y_train = data.y_train.copy()
    if "ID" in y_train.columns:
        y_train = y_train.drop(columns=["ID"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for target in y_train.columns:
        values = y_train[target].dropna()
        target_dir = args.output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            values,
            bins=args.bins,
            kde=False,
            ax=ax,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.6,
        )
        ax.set_title(f"Histogram of {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        fig.tight_layout()
        out_hist = target_dir / f"hist_{target}.png"
        fig.savefig(out_hist, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(
            values,
            ax=ax,
            color="#d62728",
            fill=True,
            alpha=0.2,
            linewidth=2.0,
        )
        ax.set_title(f"KDE Curve of {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Density")
        fig.tight_layout()
        out_kde = target_dir / f"kde_{target}.png"
        fig.savefig(out_kde, dpi=150)
        plt.close(fig)

    print(f"Saved {len(y_train.columns)} histogram plots and {len(y_train.columns)} KDE plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
