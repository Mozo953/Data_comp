from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaz_competition.data import load_modeling_data  # noqa: E402
from gaz_competition.metrics import competition_rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check: train on a shuffled target.")
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--target", default="c01")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_modeling_data(ROOT / args.data_dir)
    x_train = bundle.x_train_raw
    y_train = bundle.y_train_full

    if args.max_rows is not None:
        x_train = x_train.iloc[: args.max_rows].copy()
        y_train = y_train.iloc[: args.max_rows].copy()

    if args.target not in y_train.columns:
        raise ValueError(f"Unknown target {args.target!r}. Available targets: {list(y_train.columns)}")

    rng = np.random.default_rng(args.random_state)
    y_original = y_train[args.target].to_numpy(dtype=float)
    y_shuffled = rng.permutation(y_original)

    model_params = {
        "n_estimators": 120,
        "max_features": 0.6,
        "bootstrap": True,
        "random_state": args.random_state,
        "n_jobs": -1,
    }
    folds = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    fold_scores: list[float] = []

    for fold_number, (fit_idx, valid_idx) in enumerate(folds.split(x_train), start=1):
        model = ExtraTreesRegressor(**model_params)
        model.fit(x_train.iloc[fit_idx], y_shuffled[fit_idx])
        predictions = model.predict(x_train.iloc[valid_idx])
        score = competition_rmse(y_shuffled[valid_idx], predictions)
        fold_scores.append(score)
        print(f"fold={fold_number} shuffled_rmse={score:.6f}")

    print(f"mean_shuffled_rmse={np.mean(fold_scores):.6f}")
    print(f"std_shuffled_rmse={np.std(fold_scores):.6f}")


if __name__ == "__main__":
    main()
