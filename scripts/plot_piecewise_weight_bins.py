from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import standardize_feature_columns, standardize_target_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed piecewise sample weights by Humidity bin.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--bins-file", required=True)
    parser.add_argument("--report-prefix", default="piecewise_weight_bins")
    parser.add_argument("--title", default="Piecewise sample weights by Humidity bin")
    parser.add_argument("--weight-axis-max", type=float, default=1.5)
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--oof-file", default=None)
    parser.add_argument("--sample-weights-file", default=None)
    parser.add_argument("--loss-axis-max", type=float, default=0.09)
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
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" fill="{color}">{text}</text>'
    )


def parse_interval(interval: str) -> tuple[float, float]:
    values = re.findall(r"\d+\.\d+|\d+", str(interval))
    if len(values) < 2:
        raise ValueError(f"Unable to parse interval: {interval}")
    return float(values[0]), float(values[1])


def load_oof_predictions(oof_path: Path, y_train: pd.DataFrame) -> pd.DataFrame:
    oof = standardize_target_columns(pd.read_csv(oof_path))
    if "Unnamed: 0" in oof.columns:
        oof = oof.rename(columns={"Unnamed: 0": "row_index"})
    if "row_index" in oof.columns:
        oof = oof.set_index("row_index")
    else:
        oof.index = np.arange(len(oof))
    oof.index = oof.index.astype(int)
    columns = [column for column in oof.columns if column in y_train.columns]
    if not columns:
        raise ValueError(f"No target columns found in OOF file: {oof_path}")
    return oof[columns].reindex(y_train.index).astype(np.float64)


def load_sample_weights(weights_path: Path, expected_len: int) -> np.ndarray:
    frame = pd.read_csv(weights_path)
    value_columns = [column for column in frame.columns if column.lower() not in {"unnamed: 0", "index", "row_index"}]
    if not value_columns:
        raise ValueError(f"No sample-weight column found in {weights_path}")
    weights = frame[value_columns[-1]].to_numpy(dtype=np.float64)
    if len(weights) != expected_len:
        raise ValueError(f"Sample weight length mismatch: {len(weights)} != {expected_len}")
    return weights


def add_loss_per_bin(
    frame: pd.DataFrame,
    *,
    data_dir: Path,
    oof_path: Path,
    sample_weights_path: Path | None,
) -> pd.DataFrame:
    x_train = standardize_feature_columns(pd.read_csv(data_dir / "X_train.csv"))[["Humidity"]]
    y_train_raw = standardize_target_columns(pd.read_csv(data_dir / "y_train.csv"))
    y_train = y_train_raw.drop(columns=["ID"]) if "ID" in y_train_raw.columns else y_train_raw
    oof = load_oof_predictions(oof_path, y_train)
    y_model = y_train[oof.columns].astype(np.float64)
    humidity = x_train["Humidity"].to_numpy(dtype=np.float64)
    if sample_weights_path is None:
        row_weights = np.ones(len(humidity), dtype=np.float64)
        for _, row in frame.iterrows():
            left, right = float(row["left"]), float(row["right"])
            mask = (humidity >= left) & (humidity <= right if right >= 1.0 else humidity < right)
            row_weights[mask] = float(row["fixed_weight"])
    else:
        row_weights = load_sample_weights(sample_weights_path, len(humidity))

    sq_error = np.square(np.clip(oof.to_numpy(dtype=np.float64), 0.0, 1.0) - y_model.to_numpy(dtype=np.float64))
    losses = []
    for _, row in frame.iterrows():
        left, right = float(row["left"]), float(row["right"])
        if right >= 1.0:
            mask = (humidity >= left) & (humidity <= right)
        else:
            mask = (humidity >= left) & (humidity < right)
        if int(mask.sum()) == 0:
            losses.append(np.nan)
        else:
            losses.append(float(np.sqrt(np.mean(row_weights[mask, None] * sq_error[mask]))))
    frame = frame.copy()
    frame["weighted_loss"] = losses
    return frame


def write_svg(frame: pd.DataFrame, figure_path: Path, *, title: str, weight_axis_max: float, loss_axis_max: float) -> None:
    width, height = 1320, 720
    left, right, top, bottom = 92, 68, 76, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = max(1, int(max(frame["train_count"].max(), frame["test_count"].max())))
    count_h = plot_h * 0.35

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(width / 2, 36, title, size=24),
        svg_text(width / 2, 60, "Rouge = sample weight; violet = loss OOF; bleu/orange = volumes train/test par bin", size=12, color="#687076"),
    ]

    for tick in np.linspace(0.0, weight_axis_max, 6):
        y = top + plot_h - (tick / weight_axis_max) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e4e1dc" stroke-width="1"/>')
        lines.append(svg_text(left - 12, y + 4, f"{tick:.2f}", size=11, anchor="end", color="#687076"))

    bar_w = plot_w / len(frame) * 0.72
    step_points = []
    for _, row in frame.iterrows():
        left_edge = float(row["left"])
        right_edge = float(row["right"])
        x_mid = left + ((left_edge + right_edge) / 2.0) * plot_w
        train_h = (float(row["train_count"]) / max_count) * count_h
        test_h = (float(row["test_count"]) / max_count) * count_h
        lines.append(f'<rect x="{x_mid - bar_w / 2:.1f}" y="{top + plot_h - train_h:.1f}" width="{bar_w:.1f}" height="{train_h:.1f}" fill="#4C78A8" opacity="0.20"/>')
        lines.append(f'<rect x="{x_mid - bar_w / 2:.1f}" y="{top + plot_h - test_h:.1f}" width="{bar_w:.1f}" height="{test_h:.1f}" fill="#F58518" opacity="0.26"/>')
        y_weight = top + plot_h - (float(row["fixed_weight"]) / weight_axis_max) * plot_h
        x_left = left + left_edge * plot_w
        x_right = left + right_edge * plot_w
        step_points.extend([(x_left, y_weight), (x_right, y_weight)])
        lines.append(svg_text(x_mid, y_weight - 10, f"{float(row['fixed_weight']):.2f}", size=12, color="#B42335"))
        lines.append(svg_text(x_mid, top + plot_h + 28, str(row["humidity_interval"]), size=10, color="#687076"))

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in step_points)
    lines.append(f'<polyline points="{points}" fill="none" stroke="#C73E4A" stroke-width="4.0"/>')

    if "weighted_loss" in frame.columns:
        loss_points = []
        for _, row in frame.iterrows():
            left_edge = float(row["left"])
            right_edge = float(row["right"])
            x_mid = left + ((left_edge + right_edge) / 2.0) * plot_w
            loss = float(row["weighted_loss"])
            y_loss = top + plot_h - (min(max(loss, 0.0), loss_axis_max) / loss_axis_max) * plot_h
            loss_points.append((x_mid, y_loss))
            lines.append(f'<circle cx="{x_mid:.1f}" cy="{y_loss:.1f}" r="5.2" fill="#7C3AED" opacity="0.96"/>')
            lines.append(svg_text(x_mid, y_loss - 11, f"{loss:.3f}", size=10, color="#5B21B6"))
        loss_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in loss_points)
        lines.append(f'<polyline points="{loss_poly}" fill="none" stroke="#7C3AED" stroke-width="3.4"/>')

    for x_tick in np.linspace(0.0, 1.0, 6):
        x = left + x_tick * plot_w
        lines.append(f'<line x1="{x:.1f}" y1="{top + plot_h:.1f}" x2="{x:.1f}" y2="{top + plot_h + 7:.1f}" stroke="#777" stroke-width="1"/>')

    lines.append(svg_text(28, top + plot_h / 2, "Sample weight", size=13, color="#C73E4A"))
    lines.append(svg_text(width - 26, top + plot_h / 2, "Weighted loss", size=13, color="#7C3AED"))
    lines.append(svg_text(left + plot_w / 2, height - 24, "Humidity bin", size=14))

    for tick in np.linspace(0.0, loss_axis_max, 6):
        y = top + plot_h - (tick / loss_axis_max) * plot_h
        lines.append(svg_text(left + plot_w + 12, y + 4, f"{tick:.3f}", size=10, anchor="start", color="#7C3AED"))

    legend_x, legend_y = left + plot_w - 430, top + 32
    lines.append(f'<rect x="{legend_x - 18}" y="{legend_y - 24}" width="410" height="112" rx="8" fill="white" opacity="0.93" stroke="#ddd"/>')
    lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}" stroke="#C73E4A" stroke-width="4"/>')
    lines.append(svg_text(legend_x + 52, legend_y + 4, "sample weight", size=12, anchor="start"))
    lines.append(f'<line x1="{legend_x + 180}" y1="{legend_y}" x2="{legend_x + 222}" y2="{legend_y}" stroke="#7C3AED" stroke-width="3.4"/>')
    lines.append(svg_text(legend_x + 232, legend_y + 4, "weighted loss", size=12, anchor="start"))
    lines.append(f'<rect x="{legend_x}" y="{legend_y + 24}" width="26" height="14" fill="#4C78A8" opacity="0.20"/>')
    lines.append(svg_text(legend_x + 38, legend_y + 36, "train_count", size=12, anchor="start"))
    lines.append(f'<rect x="{legend_x + 180}" y="{legend_y + 24}" width="26" height="14" fill="#F58518" opacity="0.26"/>')
    lines.append(svg_text(legend_x + 218, legend_y + 36, "test_count", size=12, anchor="start"))
    lines.append("</svg>")
    figure_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    data_dir = resolve_path(args.data_dir)
    bins_path = source_dir / args.bins_file
    frame = pd.read_csv(bins_path)
    parsed = frame["humidity_interval"].map(parse_interval)
    frame["left"] = [left for left, _ in parsed]
    frame["right"] = [right for _, right in parsed]
    oof_path = None if args.oof_file is None else source_dir / args.oof_file
    sample_weights_path = None if args.sample_weights_file is None else source_dir / args.sample_weights_file
    if oof_path is not None:
        frame = add_loss_per_bin(
            frame,
            data_dir=data_dir,
            oof_path=oof_path,
            sample_weights_path=sample_weights_path,
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"
    write_svg(
        frame,
        figure_path,
        title=args.title,
        weight_axis_max=float(args.weight_axis_max),
        loss_axis_max=float(args.loss_axis_max),
    )
    curve_path = source_dir / f"{args.report_prefix}_{timestamp}.csv"
    frame.to_csv(curve_path, index=False)
    summary = {
        "generated_at_utc": timestamp,
        "source_bins_file": display_path(bins_path),
        "figure_path": display_path(figure_path),
        "curve_csv_path": display_path(curve_path),
        "oof_file": None if oof_path is None else display_path(oof_path),
        "sample_weights_file": None if sample_weights_path is None else display_path(sample_weights_path),
        "weight_min": float(frame["fixed_weight"].min()),
        "weight_max": float(frame["fixed_weight"].max()),
        "loss_min": None if "weighted_loss" not in frame.columns else float(frame["weighted_loss"].min()),
        "loss_max": None if "weighted_loss" not in frame.columns else float(frame["weighted_loss"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
