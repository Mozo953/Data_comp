from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple plot: greedy sample weight vs humidity.")
    parser.add_argument(
        "--source-dir",
        default="artifacts_extratrees_corr_optuna/greedy_Blender_ET3_allpool_+_rowaggbs_error_then_tradeof_loss_density_totest",
    )
    parser.add_argument("--weight-file", default="weight_20260419T232424Z.csv")
    parser.add_argument("--report-prefix", default="simple_greedy_weight_vs_humidity")
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def line_points(
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

    frame = pd.read_csv(weight_path).sort_values("env_mean").reset_index(drop=True)
    env = frame["env_mean"].to_numpy(dtype=np.float64)
    weight = frame["weight_smooth"].to_numpy(dtype=np.float64)

    width = 1120
    height = 620
    margin_left = 84
    margin_right = 36
    margin_top = 66
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min = 0.0
    x_max = 1.0
    y_min = 1.0
    y_max = max(3.0, float(weight.max()) * 1.08)
    curve = line_points(
        env,
        weight,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=margin_left,
        top=margin_top,
        width=plot_width,
        height=plot_height,
    )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">Greedy weight vs humidity</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fcfcfc" stroke="#d1d5db" stroke-width="1.2"/>',
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        x = margin_left + ((tick - x_min) / (x_max - x_min)) * plot_width
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{x:.1f}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-size="13" font-family="Arial" fill="#374151">{tick:.1f}</text>')

    for tick in np.linspace(y_min, y_max, 5):
        y = margin_top + plot_height - ((tick - y_min) / (y_max - y_min)) * plot_height
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#f3f4f6" stroke-width="1"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{tick:.2f}</text>')

    for boundary in [0.2, 0.6]:
        x = margin_left + ((boundary - x_min) / (x_max - x_min)) * plot_width
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="6,6"/>')

    elements.extend(
        [
            f'<polyline points="{curve}" fill="none" stroke="#dc2626" stroke-width="4"/>',
            f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="16" font-family="Arial" fill="#111827">Humidity / Env</text>',
            f'<text x="28" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" fill="#111827" transform="rotate(-90 28 {margin_top + plot_height / 2:.1f})">Sample weight</text>',
            f'<text x="{margin_left + plot_width - 150}" y="{margin_top + 34}" font-size="14" font-family="Arial" fill="#dc2626">weight_smooth</text>',
        ]
    )

    elements.append("</svg>")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"
    figure_path.write_text("\n".join(elements), encoding="utf-8")
    summary = {
        "generated_at_utc": timestamp,
        "source_weight_file": str(weight_path.relative_to(ROOT)),
        "figure_path": str(figure_path.relative_to(ROOT)),
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
