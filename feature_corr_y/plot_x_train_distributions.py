#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaz_competition.data import load_competition_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram and KDE distribution for each X_train feature."
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
        default=ROOT / "feature_corr_y" / "resultats_x_train",
        help="Directory where plots are saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for histograms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_competition_data(args.data_dir)

    x_train = data.x_train.copy()
    if "ID" in x_train.columns:
        x_train = x_train.drop(columns=["ID"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for feature in x_train.columns:
        values = x_train[feature].dropna()
        feature_dir = args.output_dir / feature
        feature_dir.mkdir(parents=True, exist_ok=True)

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
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        fig.tight_layout()
        out_hist = feature_dir / f"hist_{feature}.png"
        fig.savefig(out_hist, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        if values.nunique(dropna=False) > 1:
            sns.kdeplot(
                values,
                ax=ax,
                color="#d62728",
                fill=True,
                alpha=0.2,
                linewidth=2.0,
                warn_singular=False,
            )
        else:
            ax.text(0.5, 0.5, "Variance nulle: KDE indisponible", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        fig.tight_layout()
        out_kde = feature_dir / f"kde_{feature}.png"
        fig.savefig(out_kde, dpi=150)
        plt.close(fig)

    print(f"Saved {len(x_train.columns)} histogram plots and {len(x_train.columns)} KDE plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
