from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "et_rowagg_mf06_bs": "#4C78A8",
    "et_allpool_3": "#F58518",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot target-wise Dirichlet simplex weights.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--report-prefix", default="target_simplex_usage")
    parser.add_argument("--title", default="Target-wise Dirichlet simplex usage")
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    anchor: str = "middle",
    color: str = "#1F2937",
    rotate: str | None = None,
) -> str:
    transform = "" if rotate is None else f' transform="{rotate}"'
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" fill="{color}"{transform}>{escaped}</text>'
    )


def load_weights(path: Path) -> pd.DataFrame:
    weights = pd.read_csv(path, index_col=0)
    weights.index.name = "target"
    weights = weights.reset_index()
    model_cols = [column for column in weights.columns if column != "target"]
    if not model_cols:
        raise ValueError(f"No model weight columns found in {path}")
    row_sum = weights[model_cols].sum(axis=1)
    if (row_sum <= 0).any():
        raise ValueError("Some simplex rows have non-positive total weight.")
    weights[model_cols] = weights[model_cols].div(row_sum, axis=0)
    return weights


def write_svg(weights: pd.DataFrame, figure_path: Path, *, title: str) -> None:
    model_cols = [column for column in weights.columns if column != "target"]
    width, height = 1380, 760
    left, right, top, bottom = 92, 52, 86, 132
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(weights)
    gap = 8
    bar_w = max(18, (plot_w - gap * (n - 1)) / max(1, n))

    mean_weights = weights[model_cols].mean().sort_values(ascending=False)
    winners = weights[model_cols].idxmax(axis=1).value_counts().reindex(model_cols).fillna(0).astype(int)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(width / 2, 38, title, size=25, color="#111827"),
        svg_text(
            width / 2,
            63,
            "Chaque barre = poids du simplex Dirichlet par target; ce sont des poids de modèles, pas des features d'entrée",
            size=12,
            color="#687076",
        ),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d8dce0" stroke-width="1.2"/>',
    ]

    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y = top + plot_h - tick * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(svg_text(left - 12, y + 4, f"{tick:.2f}", size=11, anchor="end", color="#6B7280"))

    for i, row in weights.reset_index(drop=True).iterrows():
        x = left + i * (bar_w + gap)
        y_base = top + plot_h
        running = 0.0
        for model in model_cols:
            value = float(row[model])
            h = value * plot_h
            y = y_base - running * plot_h - h
            color = COLORS.get(model, "#999999")
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.86"/>')
            if h >= 32:
                lines.append(svg_text(x + bar_w / 2, y + h / 2 + 4, f"{value:.2f}", size=10, color="#ffffff"))
            running += value
        label_x = x + bar_w / 2
        lines.append(svg_text(label_x, top + plot_h + 30, str(row["target"]), size=11, color="#374151"))

    lines.append(svg_text(28, top + plot_h / 2, "Simplex weight", size=14, color="#111827", rotate=f"rotate(-90 28 {top + plot_h / 2:.1f})"))
    lines.append(svg_text(left + plot_w / 2, height - 26, "Modeled targets", size=14, color="#111827"))

    legend_x, legend_y = left + plot_w - 430, top + 30
    lines.append(f'<rect x="{legend_x - 18}" y="{legend_y - 24}" width="390" height="124" rx="8" fill="white" opacity="0.95" stroke="#E5E7EB"/>')
    for i, model in enumerate(model_cols):
        y = legend_y + i * 28
        lines.append(f'<rect x="{legend_x}" y="{y - 12}" width="20" height="16" fill="{COLORS.get(model, "#999999")}" opacity="0.86"/>')
        lines.append(svg_text(legend_x + 30, y + 1, f"{model}", size=12, anchor="start"))
        lines.append(svg_text(legend_x + 225, y + 1, f"mean={float(mean_weights[model]):.3f}", size=12, anchor="start", color="#4B5563"))
        lines.append(svg_text(legend_x + 315, y + 1, f"wins={int(winners[model])}", size=12, anchor="start", color="#4B5563"))

    lines.append("</svg>")
    figure_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    weights_path = source_dir / args.weights_file
    weights = load_weights(weights_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = source_dir / f"{args.report_prefix}_{timestamp}.svg"
    csv_path = source_dir / f"{args.report_prefix}_{timestamp}.csv"
    summary_path = source_dir / f"{args.report_prefix}_{timestamp}.json"

    weights.to_csv(csv_path, index=False)
    write_svg(weights, figure_path, title=args.title)

    model_cols = [column for column in weights.columns if column != "target"]
    winners = weights[model_cols].idxmax(axis=1).value_counts().reindex(model_cols).fillna(0).astype(int)
    summary = {
        "generated_at_utc": timestamp,
        "weights_file": display_path(weights_path),
        "figure_path": display_path(figure_path),
        "curve_csv_path": display_path(csv_path),
        "mean_weights": {model: float(weights[model].mean()) for model in model_cols},
        "winner_counts": {model: int(winners[model]) for model in model_cols},
        "note": "Dirichlet simplex weights are target-wise model weights, not input-feature importances.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
