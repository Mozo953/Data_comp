from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import load_modeling_data  # noqa: E402


MODEL_COLUMNS = ["et_rowagg_mf06_bs", "et_allpool_3", "rf_local_045_080"]
COLORS = {
    "et_rowagg_mf06_bs": "#4C78A8",
    "et_allpool_3": "#F58518",
    "rf_local_045_080": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot each model's effective contribution in a conditional blend by humidity bin."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--rowagg-oof-file", required=True)
    parser.add_argument("--allpool-oof-file", required=True)
    parser.add_argument("--rf-oof-file", required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--ada-low", type=float, default=0.45)
    parser.add_argument("--ada-high", type=float, default=0.80)
    parser.add_argument("--report-prefix", default="conditional_model_implication_by_humidity_bin")
    parser.add_argument("--title", default="Model implication by Humidity bin")
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle", color: str = "#2F3437") -> str:
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" fill="{color}">{escaped}</text>'
    )


def load_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(int)
    return frame.astype(np.float32)


def zone_for_humidity(humidity: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where((humidity >= low) & (humidity <= high), "inside_045_080", "outside_045_080")


def compute_implication(
    humidity: pd.Series,
    predictions: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    *,
    bins: int,
    low: float,
    high: float,
) -> pd.DataFrame:
    targets = list(predictions[MODEL_COLUMNS[0]].columns)
    for model_name in MODEL_COLUMNS:
        if list(predictions[model_name].columns) != targets:
            raise ValueError(f"OOF target columns mismatch for {model_name}.")

    n_rows = len(humidity)
    for model_name, frame in predictions.items():
        if len(frame) != n_rows:
            raise ValueError(f"OOF row count mismatch for {model_name}: {len(frame)} vs humidity={n_rows}.")

    weight_lookup = {
        (str(row["target"]), str(row["zone"])): np.asarray([row[model] for model in MODEL_COLUMNS], dtype=np.float32)
        for _, row in weights.iterrows()
    }
    humidity_values = humidity.to_numpy(dtype=np.float32)
    zones = zone_for_humidity(humidity_values, low, high)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []

    pred_arrays = {model: predictions[model][targets].to_numpy(dtype=np.float32) for model in MODEL_COLUMNS}

    for bin_id in range(bins):
        left, right = float(edges[bin_id]), float(edges[bin_id + 1])
        if bin_id == bins - 1:
            mask = (humidity_values >= left) & (humidity_values <= right)
        else:
            mask = (humidity_values >= left) & (humidity_values < right)
        count = int(mask.sum())
        row: dict[str, float | int] = {
            "bin_id": int(bin_id),
            "humidity_left": left,
            "humidity_right": right,
            "humidity_mid": (left + right) / 2.0,
            "train_count": count,
        }
        if count == 0:
            for model in MODEL_COLUMNS:
                row[f"{model}_weight_mean"] = 0.0
                row[f"{model}_prediction_share"] = 0.0
            rows.append(row)
            continue

        contribution_sum = {model: 0.0 for model in MODEL_COLUMNS}
        contribution_total = 0.0
        weight_sum = {model: 0.0 for model in MODEL_COLUMNS}
        weight_count = 0

        row_indices = np.flatnonzero(mask)
        for target_idx, target in enumerate(targets):
            inside_weights = weight_lookup[(target, "inside_045_080")]
            outside_weights = weight_lookup[(target, "outside_045_080")]
            target_weights_by_row = np.where(zones[row_indices, None] == "inside_045_080", inside_weights, outside_weights)
            for model_idx, model in enumerate(MODEL_COLUMNS):
                pred_values = pred_arrays[model][row_indices, target_idx]
                contributions = target_weights_by_row[:, model_idx] * pred_values
                contribution_sum[model] += float(contributions.sum(dtype=np.float64))
                weight_sum[model] += float(target_weights_by_row[:, model_idx].sum(dtype=np.float64))
            contribution_total += float(
                sum(
                    (target_weights_by_row[:, model_idx] * pred_arrays[model][row_indices, target_idx]).sum(dtype=np.float64)
                    for model_idx, model in enumerate(MODEL_COLUMNS)
                )
            )
            weight_count += len(row_indices)

        for model in MODEL_COLUMNS:
            row[f"{model}_weight_mean"] = weight_sum[model] / max(1, weight_count)
            row[f"{model}_prediction_share"] = contribution_sum[model] / max(1e-12, contribution_total)
        rows.append(row)

    return pd.DataFrame(rows)


def write_svg(curve: pd.DataFrame, figure_path: Path, *, title: str) -> None:
    width, height = 1500, 760
    left, right, top, bottom = 95, 50, 82, 115
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 4
    n = len(curve)
    bar_w = max(4.0, (plot_w - bar_gap * (n - 1)) / max(1, n))
    max_count = max(1, int(curve["train_count"].max()))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(width / 2, 38, title, size=24),
        svg_text(width / 2, 62, "Barres empilÃ©es = contribution effective dans la prÃ©diction finale; ligne grise = nombre de lignes train", size=12, color="#687076"),
    ]

    # Grid for share axis.
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e1db" stroke-width="1"/>')
        lines.append(svg_text(left - 12, y + 4, f"{tick:.1f}", size=11, anchor="end", color="#687076"))
    lines.append(svg_text(22, top + plot_h / 2, "Contribution share", size=13, color="#687076"))

    # Stacked contribution bars.
    for _, row in curve.iterrows():
        x = left + int(row["bin_id"]) * (bar_w + bar_gap)
        running = 0.0
        for model in MODEL_COLUMNS:
            value = float(row[f"{model}_prediction_share"])
            h = value * plot_h
            y = top + plot_h - running * plot_h - h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(0.0, h):.1f}" '
                f'fill="{COLORS[model]}" opacity="0.82"/>'
            )
            running += value

    # Count line, scaled on same panel for context.
    count_points = []
    for _, row in curve.iterrows():
        x = left + int(row["bin_id"]) * (bar_w + bar_gap) + bar_w / 2
        y = top + plot_h - (float(row["train_count"]) / max_count) * plot_h
        count_points.append(f"{x:.1f},{y:.1f}")
    lines.append(f'<polyline points="{" ".join(count_points)}" fill="none" stroke="#42484D" stroke-width="2.8" opacity="0.58"/>')

    # X labels.
    for _, row in curve.iterrows():
        bin_id = int(row["bin_id"])
        if bin_id % 2 == 0:
            x = left + bin_id * (bar_w + bar_gap) + bar_w / 2
            label = f"{float(row['humidity_left']):.2f}-{float(row['humidity_right']):.2f}"
            lines.append(svg_text(x, top + plot_h + 28, label, size=10, color="#687076"))
    lines.append(svg_text(left + plot_w / 2, height - 34, "Humidity bin", size=14))

    # Legend.
    legend_x, legend_y = left, height - 78
    for i, model in enumerate(MODEL_COLUMNS):
        x = legend_x + i * 260
        lines.append(f'<rect x="{x:.1f}" y="{legend_y - 12:.1f}" width="16" height="16" fill="{COLORS[model]}" opacity="0.85"/>')
        lines.append(svg_text(x + 24, legend_y + 1, model, size=12, anchor="start"))
    lines.append(f'<line x1="{legend_x + 790}" y1="{legend_y - 4}" x2="{legend_x + 840}" y2="{legend_y - 4}" stroke="#42484D" stroke-width="2.8" opacity="0.58"/>')
    lines.append(svg_text(legend_x + 850, legend_y + 1, "train_count scaled", size=12, anchor="start"))
    lines.append("</svg>")
    figure_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    weights_path = source_dir / args.weights_file
    weights = pd.read_csv(weights_path)
    predictions = {
        "et_rowagg_mf06_bs": load_oof(source_dir / args.rowagg_oof_file),
        "et_allpool_3": load_oof(source_dir / args.allpool_oof_file),
        "rf_local_045_080": load_oof(source_dir / args.rf_oof_file),
    }
    humidity = load_modeling_data(resolve_path(args.data_dir)).data.x_train["Humidity"]
    humidity = humidity.iloc[: len(predictions["et_rowagg_mf06_bs"])].reset_index(drop=True)

    curve = compute_implication(
        humidity,
        predictions,
        weights,
        bins=int(args.bins),
        low=float(args.ada_low),
        high=float(args.ada_high),
    )
    csv_path = source_dir / f"{args.report_prefix}_{timestamp}.csv"
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"
    curve.to_csv(csv_path, index=False)
    write_svg(curve, figure_path, title=args.title)

    summary = {
        "generated_at_utc": timestamp,
        "figure_path": display_path(figure_path),
        "curve_csv_path": display_path(csv_path),
        "weights_file": display_path(weights_path),
        "bins": int(args.bins),
        "model_columns": MODEL_COLUMNS,
        "note": "prediction_share is computed from target-wise blend weights multiplied by each model OOF prediction, aggregated per humidity bin.",
        "max_rf_share_outside_interval": float(
            curve.loc[
                (curve["humidity_right"] <= float(args.ada_low)) | (curve["humidity_left"] > float(args.ada_high)),
                "rf_local_045_080_prediction_share",
            ].max()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

