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
        description="Generate a correlation heatmap for the 23 Y_train targets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "src" / "odor_competition" / "data",
        help="Directory containing X_train.csv, X_test.csv and y_train.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "feature_corr_y" / "correlation",
        help="Directory where correlation outputs are saved.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "kendall"],
        help="Correlation method.",
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="Display correlation values in heatmap cells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_competition_data(args.data_dir)
    y_train = data.y_train.copy()
    if "ID" in y_train.columns:
        y_train = y_train.drop(columns=["ID"])

    corr = y_train.corr(method=args.method)
    corr_csv = args.output_dir / f"y_train_corr_{args.method}.csv"
    corr.to_csv(corr_csv)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cmap="coolwarm",
        square=True,
        annot=args.annot,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )
    plt.tight_layout()

    heatmap_png = args.output_dir / f"y_train_corr_heatmap_{args.method}.png"
    plt.savefig(heatmap_png, dpi=180)
    plt.close()

    print(f"Saved correlation matrix: {corr_csv}")
    print(f"Saved heatmap: {heatmap_png}")


if __name__ == "__main__":
    main()
