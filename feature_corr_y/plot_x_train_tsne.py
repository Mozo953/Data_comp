#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import load_competition_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot t-SNE for X_train Xi features."
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
        default=ROOT / "feature_corr_y" / "tsne_x",
        help="Directory where t-SNE outputs are saved.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of rows used for t-SNE. Use 0 to disable sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_competition_data(args.data_dir)
    x_train = data.x_train.copy()

    ids = x_train["ID"].copy() if "ID" in x_train.columns else pd.RangeIndex(len(x_train))
    if "ID" in x_train.columns:
        x_train = x_train.drop(columns=["ID"])

    if args.max_samples > 0 and len(x_train) > args.max_samples:
        sampled_idx = x_train.sample(n=args.max_samples, random_state=args.random_state).index
        x_train = x_train.loc[sampled_idx]
        ids = pd.Series(ids, index=data.x_train.index).loc[sampled_idx].to_numpy()
        env_values = data.x_train.loc[sampled_idx, "Env"].to_numpy() if "Env" in data.x_train.columns else None
    else:
        ids = ids.to_numpy() if hasattr(ids, "to_numpy") else ids
        env_values = data.x_train["Env"].to_numpy() if "Env" in data.x_train.columns else None

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        init="pca",
        random_state=args.random_state,
        max_iter=1000,
    )
    embedding = tsne.fit_transform(x_scaled)

    out_df = pd.DataFrame(
        {
            "ID": ids,
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
        }
    )

    if env_values is not None:
        out_df["Env"] = env_values

    csv_path = args.output_dir / "x_train_tsne.csv"
    out_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 8))
    if "Env" in out_df.columns:
        sns.scatterplot(
            data=out_df,
            x="tsne_1",
            y="tsne_2",
            hue="Env",
            palette="viridis",
            s=22,
            alpha=0.85,
            linewidth=0,
        )
    else:
        sns.scatterplot(data=out_df, x="tsne_1", y="tsne_2", s=22, alpha=0.85, linewidth=0)

    plt.title("t-SNE projection of X_train Xi features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    png_path = args.output_dir / "x_train_tsne.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    print(f"Saved t-SNE table: {csv_path}")
    print(f"Saved t-SNE plot: {png_path}")


if __name__ == "__main__":
    main()
