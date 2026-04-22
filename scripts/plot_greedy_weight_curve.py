from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot greedy blender continuous weight curve.")
    parser.add_argument(
        "--source-dir",
        default="artifacts_extratrees_corr_optuna/greedy_Blender_ET3_allpool_+_rowaggbs_error_then_tradeof_loss_density_totest",
    )
    parser.add_argument("--weight-file", default="weight_20260419T232424Z.csv")
    parser.add_argument("--report-prefix", default="greedy_weight_curve")
    parser.add_argument("--weight-axis-cap", type=float, default=3.0)
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    pts: list[str] = []
    for x_val, y_val in zip(x_values, y_values):
        x = left + ((float(x_val) - x_min) / max(x_max - x_min, 1e-9)) * width
        y = top + height - ((float(y_val) - y_min) / max(y_max - y_min, 1e-9)) * height
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    weight_path = source_dir / args.weight_file
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)

    frame = pd.read_csv(weight_path)
    frame = frame.sort_values("humidity_mean").reset_index(drop=True)
    x = frame["humidity_mean"].to_numpy(dtype=np.float64)
    weight_smooth = frame["weight_smooth"].to_numpy(dtype=np.float64)
    weight_raw = frame["weight_raw"].to_numpy(dtype=np.float64)
    loss_norm = frame["loss_norm"].to_numpy(dtype=np.float64)
    ratio_norm = frame["ratio_norm"].to_numpy(dtype=np.float64)

    width = 1280
    height = 760
    margin_left = 92
    margin_right = 86
    margin_top = 72
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = -0.02
    x_max = 1.02
    weight_min = 0.0
    weight_max = float(args.weight_axis_cap)
    norm_min = 0.0
    norm_max = 1.0

    def x_pos(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_left(value: float) -> float:
        return margin_top + plot_height - ((value - weight_min) / (weight_max - weight_min)) * plot_height

    def y_right(value: float) -> float:
        return margin_top + plot_height - ((value - norm_min) / (norm_max - norm_min)) * plot_height

    weight_smooth_line = points(
        x,
        np.clip(weight_smooth, weight_min, weight_max),
        x_min=x_min,
        x_max=x_max,
        y_min=weight_min,
        y_max=weight_max,
        left=margin_left,
        top=margin_top,
        width=plot_width,
        height=plot_height,
    )
    weight_raw_line = points(
        x,
        np.clip(weight_raw, weight_min, weight_max),
        x_min=x_min,
        x_max=x_max,
        y_min=weight_min,
        y_max=weight_max,
        left=margin_left,
        top=margin_top,
        width=plot_width,
        height=plot_height,
    )
    loss_line = points(
        x,
        loss_norm,
        x_min=x_min,
        x_max=x_max,
        y_min=norm_min,
        y_max=norm_max,
        left=margin_left,
        top=margin_top,
        width=plot_width,
        height=plot_height,
    )
    ratio_line = points(
        x,
        ratio_norm,
        x_min=x_min,
        x_max=x_max,
        y_min=norm_min,
        y_max=norm_max,
        left=margin_left,
        top=margin_top,
        width=plot_width,
        height=plot_height,
    )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8f8f8"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">Greedy continuous weight curve vs humidity</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fcfcfc" stroke="#c7c7c7" stroke-width="1.2"/>',
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        x_tick = x_pos(float(tick))
        elements.append(f'<line x1="{x_tick:.1f}" y1="{margin_top}" x2="{x_tick:.1f}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{x_tick:.1f}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-size="13" font-family="Arial">{tick:.1f}</text>')
    for tick in np.linspace(weight_min, weight_max, 7):
        y_tick = y_left(float(tick))
        elements.append(f'<line x1="{margin_left}" y1="{y_tick:.1f}" x2="{margin_left + plot_width}" y2="{y_tick:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y_tick + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{tick:.1f}</text>')
    for tick in np.linspace(0.0, 1.0, 6):
        y_tick = y_right(float(tick))
        elements.append(f'<text x="{margin_left + plot_width + 12}" y="{y_tick + 4:.1f}" text-anchor="start" font-size="12" font-family="Arial" fill="#2563eb">{tick:.1f}</text>')

    for boundary in [0.2, 0.6, 1.0]:
        x_boundary = x_pos(boundary)
        elements.append(f'<line x1="{x_boundary:.1f}" y1="{margin_top}" x2="{x_boundary:.1f}" y2="{margin_top + plot_height}" stroke="#b9b9b9" stroke-width="1.2" stroke-dasharray="6,6"/>')

    elements.extend(
        [
            f'<polyline points="{weight_raw_line}" fill="none" stroke="#fca5a5" stroke-width="2.0" stroke-dasharray="5,5"/>',
            f'<polyline points="{weight_smooth_line}" fill="none" stroke="#dc2626" stroke-width="3.5"/>',
            f'<polyline points="{loss_line}" fill="none" stroke="#2563eb" stroke-width="2.4" stroke-dasharray="8,6"/>',
            f'<polyline points="{ratio_line}" fill="none" stroke="#16a34a" stroke-width="2.4" stroke-dasharray="8,6"/>',
        f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="Arial">Humidity</text>',
            f'<text x="28" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" fill="#dc2626" transform="rotate(-90 28 {margin_top + plot_height / 2:.1f})">Sample weight</text>',
            f'<text x="{width - 28}" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" fill="#2563eb" transform="rotate(90 {width - 28} {margin_top + plot_height / 2:.1f})">Normalized loss / ratio</text>',
        ]
    )

    legend_x = margin_left + plot_width - 330
    legend_y = margin_top + 28
    elements.append(f'<rect x="{legend_x - 18}" y="{legend_y - 22}" width="310" height="120" rx="8" fill="white" fill-opacity="0.95" stroke="#d0d0d0"/>')
    legend_items = [
        ("weight smooth", "#dc2626", ""),
        ("weight raw", "#fca5a5", ' stroke-dasharray="5,5"'),
        ("loss_norm", "#2563eb", ' stroke-dasharray="8,6"'),
        ("ratio_norm", "#16a34a", ' stroke-dasharray="8,6"'),
    ]
    for i, (label, color, dash) in enumerate(legend_items):
        y = legend_y + i * 24
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>')
        elements.append(f'<text x="{legend_x + 42}" y="{y + 4}" font-size="13" font-family="Arial" fill="#1f2937">{label}</text>')

    elements.append("</svg>")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"
    figure_path.write_text("\n".join(elements), encoding="utf-8")
    summary = {
        "generated_at_utc": timestamp,
        "source_weight_file": str(weight_path.relative_to(ROOT)),
        "figure_path": str(figure_path.relative_to(ROOT)),
        "weight_min": float(weight_smooth.min()),
        "weight_max": float(weight_smooth.max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

