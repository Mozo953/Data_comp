from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


COLORS = {
    "et_rowagg_mf06_bs": "#4C78A8",
    "et_allpool_3": "#F58518",
    "rf_local_045_080": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot conditional blender weights by zone and target.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--report-prefix", default="conditional_blend_weights")
    parser.add_argument("--title", default="Conditional blender weights")
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


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    weights_path = source_dir / args.weights_file
    weights = pd.read_csv(weights_path)
    model_cols = [column for column in ["et_rowagg_mf06_bs", "et_allpool_3", "rf_local_045_080"] if column in weights.columns]
    if "zone" not in weights.columns or "target" not in weights.columns or not model_cols:
        raise ValueError("Weights CSV must contain target, zone, and model weight columns.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"
    mean_path = source_dir / f"{args.report_prefix}_{timestamp}_zone_means.csv"

    zone_order = [zone for zone in ["inside_045_080", "outside_045_080"] if zone in set(weights["zone"])]
    zone_means = weights.groupby("zone")[model_cols].mean().reindex(zone_order)
    zone_means.to_csv(mean_path)

    width, height = 1300, 720
    margin_left, margin_right = 90, 40
    top, bottom = 80, 115
    plot_w = width - margin_left - margin_right
    plot_h = height - top - bottom
    panel_gap = 60
    panel_w = (plot_w - panel_gap) / 2
    max_targets = max(1, weights["target"].nunique())

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(width / 2, 38, args.title, size=24),
        svg_text(width / 2, 62, "Chaque barre = poids target-wise moyen ou poids par target; RF forcé à 0 hors zone", size=12, color="#687076"),
    ]

    # Left panel: stacked mean weights by zone.
    x0, y0 = margin_left, top
    lines.append(svg_text(x0 + panel_w / 2, y0 - 18, "Moyenne des poids par zone", size=16))
    bar_w = 170
    gap = 120
    for i, zone in enumerate(zone_order):
        x = x0 + 95 + i * (bar_w + gap)
        y_base = y0 + plot_h
        running = 0.0
        for model in model_cols:
            value = float(zone_means.loc[zone, model]) if zone in zone_means.index else 0.0
            h = value * (plot_h - 45)
            y = y_base - running * (plot_h - 45) - h
            color = COLORS.get(model, "#999999")
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.82"/>')
            if h > 20:
                lines.append(svg_text(x + bar_w / 2, y + h / 2 + 4, f"{value:.2f}", size=12, color="#ffffff"))
            running += value
        label = "inside [0.45,0.80]" if zone.startswith("inside") else "outside"
        lines.append(svg_text(x + bar_w / 2, y_base + 28, label, size=12))
    for tick in np.linspace(0, 1, 6):
        y = y0 + plot_h - tick * (plot_h - 45)
        lines.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_w - 20:.1f}" y2="{y:.1f}" stroke="#e4e1dc" stroke-width="1"/>')
        lines.append(svg_text(x0 - 12, y + 4, f"{tick:.1f}", size=11, anchor="end", color="#687076"))

    # Right panel: per-target lines/points by zone.
    x1 = margin_left + panel_w + panel_gap
    lines.append(svg_text(x1 + panel_w / 2, y0 - 18, "Poids par target", size=16))
    target_order = sorted(weights["target"].unique())
    x_positions = {target: x1 + 15 + idx * ((panel_w - 60) / max(1, max_targets - 1)) for idx, target in enumerate(target_order)}
    for zone_idx, zone in enumerate(zone_order):
        zone_df = weights[weights["zone"] == zone].set_index("target").reindex(target_order)
        dash = "" if zone.startswith("inside") else ' stroke-dasharray="6 5"'
        opacity = "0.92" if zone.startswith("inside") else "0.56"
        for model in model_cols:
            points = []
            for target in target_order:
                value = float(zone_df.loc[target, model])
                x = x_positions[target]
                y = y0 + plot_h - value * (plot_h - 45)
                points.append(f"{x:.1f},{y:.1f}")
            color = COLORS.get(model, "#999999")
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4" opacity="{opacity}"{dash}/>')
    for tick in np.linspace(0, 1, 6):
        y = y0 + plot_h - tick * (plot_h - 45)
        lines.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x1 + panel_w:.1f}" y2="{y:.1f}" stroke="#e4e1dc" stroke-width="1"/>')
        lines.append(svg_text(x1 - 12, y + 4, f"{tick:.1f}", size=11, anchor="end", color="#687076"))
    for idx, target in enumerate(target_order):
        if idx % 2 == 0:
            lines.append(svg_text(x_positions[target], y0 + plot_h + 28, target, size=10, color="#687076"))

    legend_x, legend_y = margin_left, height - 42
    for i, model in enumerate(model_cols):
        x = legend_x + i * 240
        lines.append(f'<rect x="{x:.1f}" y="{legend_y - 12:.1f}" width="16" height="16" fill="{COLORS.get(model, "#999999")}" opacity="0.85"/>')
        lines.append(svg_text(x + 24, legend_y + 1, model, size=12, anchor="start"))
    lines.append(svg_text(width - 260, height - 42, "plein = inside, pointillé = outside", size=12, anchor="start", color="#687076"))

    lines.append("</svg>")
    figure_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "generated_at_utc": timestamp,
        "weights_file": display_path(weights_path),
        "figure_path": display_path(figure_path),
        "zone_mean_weights_path": display_path(mean_path),
        "zone_mean_weights": zone_means.to_dict(orient="index"),
        "outside_max_rf_weight": float(weights.loc[weights["zone"] == "outside_045_080", "rf_local_045_080"].max())
        if "rf_local_045_080" in weights.columns and "outside_045_080" in set(weights["zone"])
        else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
