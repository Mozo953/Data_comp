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

from gaz_competition.data import standardize_feature_columns, standardize_target_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original-style piecewise weight plot with green loss curve.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--bins-file", required=True)
    parser.add_argument("--oof-file", required=True)
    parser.add_argument("--sample-weights-file", required=True)
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--report-prefix", default="piecewise_weight_bins_with_loss_original_style")
    parser.add_argument("--title", default="Picked model sample weight by Humidity bin")
    parser.add_argument("--weight-y-min", type=float, default=0.95)
    parser.add_argument("--weight-y-max", type=float, default=1.42)
    parser.add_argument("--row-axis-max", type=float, default=162000.0)
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


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle", color: str = "#1F2937", rotate: str | None = None) -> str:
    transform = "" if rotate is None else f' transform="{rotate}"'
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" fill="{color}"{transform}>{escaped}</text>'
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
        raise ValueError(f"No target columns found in {oof_path}")
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


def add_weighted_loss(
    bins: pd.DataFrame,
    *,
    data_dir: Path,
    oof_path: Path,
    sample_weights_path: Path,
) -> pd.DataFrame:
    x_train = standardize_feature_columns(pd.read_csv(data_dir / "X_train.csv"))[["Humidity"]]
    y_train_raw = standardize_target_columns(pd.read_csv(data_dir / "y_train.csv"))
    y_train = y_train_raw.drop(columns=["ID"]) if "ID" in y_train_raw.columns else y_train_raw
    oof = load_oof_predictions(oof_path, y_train)
    y_model = y_train[oof.columns].astype(np.float64)
    humidity = x_train["Humidity"].to_numpy(dtype=np.float64)
    row_weights = load_sample_weights(sample_weights_path, len(humidity))
    sq_error = np.square(np.clip(oof.to_numpy(dtype=np.float64), 0.0, 1.0) - y_model.to_numpy(dtype=np.float64))

    losses = []
    for _, row in bins.iterrows():
        left, right = float(row["left"]), float(row["right"])
        if right >= 1.0:
            mask = (humidity >= left) & (humidity <= right)
        else:
            mask = (humidity >= left) & (humidity < right)
        losses.append(float(np.sqrt(np.mean(row_weights[mask, None] * sq_error[mask]))) if int(mask.sum()) else np.nan)
    bins = bins.copy()
    bins["weighted_loss"] = losses
    return bins


def write_svg(
    frame: pd.DataFrame,
    figure_path: Path,
    *,
    title: str,
    weight_y_min: float,
    weight_y_max: float,
    row_axis_max: float,
    loss_axis_max: float,
) -> None:
    width, height = 1280, 720
    left, right, top, bottom = 92, 96, 76, 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(frame)
    step = plot_w / n
    bar_w = step * 0.28

    def x_mid(i: int) -> float:
        return left + step * (i + 0.5)

    def y_weight(value: float) -> float:
        return top + plot_h - ((value - weight_y_min) / (weight_y_max - weight_y_min)) * plot_h

    def y_rows(value: float) -> float:
        return top + plot_h - (value / row_axis_max) * plot_h

    def y_loss(value: float) -> float:
        # Loss is drawn on its own green scale but kept in the same panel.
        return top + plot_h - (min(max(value, 0.0), loss_axis_max) / loss_axis_max) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 34, title, size=26, color="#111827"),
        svg_text(width / 2, 60, "Fixed piecewise weights with train/test row counts and OOF loss context", size=13, color="#6B7280"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="#d1d5db" stroke-width="1.2"/>',
    ]

    for tick in [1.00, 1.10, 1.25, 1.35, 1.40]:
        y = y_weight(tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(svg_text(left - 14, y + 4, f"{tick:.2f}", size=12, anchor="end", color="#374151"))

    for tick in np.linspace(0.0, row_axis_max, 5):
        y = y_rows(float(tick))
        label = "0" if tick == 0 else f"{int(tick / 1000)}k"
        lines.append(svg_text(left + plot_w + 14, y + 4, label, size=11, anchor="start", color="#6B7280"))

    loss_tick_x = left + plot_w + 56
    for tick in np.linspace(0.0, loss_axis_max, 4):
        y = y_loss(float(tick))
        lines.append(svg_text(loss_tick_x, y + 4, f"{tick:.2f}", size=10, anchor="start", color="#15803D"))

    for i, row in frame.reset_index(drop=True).iterrows():
        xm = x_mid(i)
        train_y = y_rows(float(row["train_count"]))
        test_y = y_rows(float(row["test_count"]))
        lines.append(
            f'<rect x="{xm - bar_w - 2:.1f}" y="{train_y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - train_y:.1f}" '
            f'fill="#AFC6F3" opacity="0.65"/>'
        )
        lines.append(
            f'<rect x="{xm + 2:.1f}" y="{test_y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - test_y:.1f}" '
            f'fill="#F6D39B" opacity="0.72"/>'
        )

    weight_points = []
    loss_points = []
    for i, row in frame.reset_index(drop=True).iterrows():
        xm = x_mid(i)
        yw = y_weight(float(row["fixed_weight"]))
        yl = y_loss(float(row["weighted_loss"]))
        weight_points.append((xm, yw))
        loss_points.append((xm, yl))
        lines.append(f'<line x1="{xm:.1f}" y1="{top + plot_h:.1f}" x2="{xm:.1f}" y2="{yw:.1f}" stroke="#EF4444" stroke-width="1" opacity="0.42"/>')
        lines.append(f'<circle cx="{xm:.1f}" cy="{yw:.1f}" r="6" fill="#DC2626"/>')
        lines.append(svg_text(xm, yw - 13, f"{float(row['fixed_weight']):.2f}", size=12, color="#B91C1C"))
        lines.append(f'<circle cx="{xm:.1f}" cy="{yl:.1f}" r="5.2" fill="#16A34A"/>')
        lines.append(svg_text(xm, yl - 12, f"{float(row['weighted_loss']):.3f}", size=10, color="#15803D"))
        label = str(row["humidity_interval"])
        lines.append(svg_text(xm, top + plot_h + 40, label, size=10, color="#374151", rotate=f"rotate(-35 {xm:.1f} {top + plot_h + 40:.1f})"))

    weight_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in weight_points)
    loss_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in loss_points)
    lines.append(f'<polyline points="{weight_poly}" fill="none" stroke="#DC2626" stroke-width="3.5"/>')
    lines.append(f'<polyline points="{loss_poly}" fill="none" stroke="#16A34A" stroke-width="3.2"/>')

    lines.append(svg_text(24, top + plot_h / 2, "Sample weight", size=14, color="#111827", rotate=f"rotate(-90 24 {top + plot_h / 2:.1f})"))
    lines.append(svg_text(width - 22, top + plot_h / 2, "Rows / Loss scale", size=13, color="#4B5563", rotate=f"rotate(90 {width - 22} {top + plot_h / 2:.1f})"))
    lines.append(svg_text(left + plot_w / 2, height - 24, "Humidity bins", size=15, color="#111827"))

    legend_x, legend_y = left + plot_w - 310, top + 30
    lines.append(f'<rect x="{legend_x - 18}" y="{legend_y - 24}" width="290" height="122" rx="8" fill="white" opacity="0.94" stroke="#E5E7EB"/>')
    lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="28" height="15" fill="#AFC6F3" opacity="0.65"/>')
    lines.append(svg_text(legend_x + 40, legend_y + 3, "train count", size=12, anchor="start"))
    lines.append(f'<rect x="{legend_x}" y="{legend_y + 16}" width="28" height="15" fill="#F6D39B" opacity="0.72"/>')
    lines.append(svg_text(legend_x + 40, legend_y + 29, "test count", size=12, anchor="start"))
    lines.append(f'<line x1="{legend_x}" y1="{legend_y + 53}" x2="{legend_x + 34}" y2="{legend_y + 53}" stroke="#DC2626" stroke-width="3.5"/>')
    lines.append(svg_text(legend_x + 46, legend_y + 57, "fixed sample weight", size=12, anchor="start"))
    lines.append(f'<line x1="{legend_x}" y1="{legend_y + 79}" x2="{legend_x + 34}" y2="{legend_y + 79}" stroke="#16A34A" stroke-width="3.2"/>')
    lines.append(svg_text(legend_x + 46, legend_y + 83, "weighted OOF loss", size=12, anchor="start"))

    lines.append("</svg>")
    figure_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    data_dir = resolve_path(args.data_dir)
    bins_path = source_dir / args.bins_file
    oof_path = source_dir / args.oof_file
    sample_weights_path = source_dir / args.sample_weights_file
    frame = pd.read_csv(bins_path)
    parsed = frame["humidity_interval"].map(parse_interval)
    frame["left"] = [left for left, _ in parsed]
    frame["right"] = [right for _, right in parsed]
    frame = add_weighted_loss(frame, data_dir=data_dir, oof_path=oof_path, sample_weights_path=sample_weights_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    csv_path = source_dir / f"{args.report_prefix}_{timestamp}.csv"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"
    frame.to_csv(csv_path, index=False)
    write_svg(
        frame,
        figure_path,
        title=args.title,
        weight_y_min=float(args.weight_y_min),
        weight_y_max=float(args.weight_y_max),
        row_axis_max=float(args.row_axis_max),
        loss_axis_max=float(args.loss_axis_max),
    )
    summary = {
        "generated_at_utc": timestamp,
        "figure_path": display_path(figure_path),
        "curve_csv_path": display_path(csv_path),
        "bins_file": display_path(bins_path),
        "oof_file": display_path(oof_path),
        "sample_weights_file": display_path(sample_weights_path),
        "loss_min": float(frame["weighted_loss"].min()),
        "loss_max": float(frame["weighted_loss"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
