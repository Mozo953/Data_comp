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

from gaz_competition.data import standardize_feature_columns, standardize_target_columns  # noqa: E402


COLORS = {
    "model_a": "#C73E4A",
    "model_b": "#2F6DB3",
    "diff": "#33383D",
    "count": "#8B8F92",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two OOF blend losses by Humidity bins.")
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--model-a-dir", required=True)
    parser.add_argument("--model-a-oof", required=True)
    parser.add_argument("--model-a-label", default="best_0.1391")
    parser.add_argument("--model-b-dir", required=True)
    parser.add_argument("--model-b-oof", required=True)
    parser.add_argument("--model-b-label", default="model_53_rf")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-prefix", default="compare_humidity_bin_loss")
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--loss-weighting", choices=["binary-humidity", "none"], default="binary-humidity")
    parser.add_argument("--loss-axis-cap", type=float, default=0.09)
    parser.add_argument("--title", default="Loss by Humidity bin: best vs RF blender")
    parser.add_argument("--histogram-density", action="store_true", help="Plot train/test as density histograms instead of row counts.")
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


def load_oof_predictions(oof_path: Path, y_train: pd.DataFrame) -> pd.DataFrame:
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    oof = standardize_target_columns(pd.read_csv(oof_path))
    if "Unnamed: 0" in oof.columns:
        oof = oof.rename(columns={"Unnamed: 0": "row_index"})
    if "row_index" in oof.columns:
        oof = oof.set_index("row_index")
    else:
        oof.index = np.arange(len(oof))
    oof.index = oof.index.astype(int)
    model_cols = [column for column in oof.columns if column in y_train.columns]
    if not model_cols:
        raise ValueError(f"No target columns from y_train found in {oof_path}")
    return oof[model_cols].reindex(y_train.index).astype(np.float64)


def row_weights_for_humidity(humidity: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(humidity), dtype=np.float64)
    return np.where(humidity >= 0.6, 1.2, 1.0).astype(np.float64)


def row_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> np.ndarray:
    pred = np.clip(y_pred.to_numpy(dtype=np.float64), 0.0, 1.0)
    true = y_true[y_pred.columns].to_numpy(dtype=np.float64)
    return np.sqrt(np.mean(np.square(pred - true), axis=1))


def bin_loss_curve(
    humidity: np.ndarray,
    test_humidity: np.ndarray,
    y_true: pd.DataFrame,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    row_weights: np.ndarray,
    bin_width: float,
) -> pd.DataFrame:
    if not np.isclose((1.0 / bin_width), round(1.0 / bin_width), atol=1e-8):
        raise ValueError("--bin-width should divide 1.0 cleanly, e.g. 0.1 or 0.05.")
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    edges[-1] = 1.0
    rmse_a = row_rmse(y_true, pred_a)
    rmse_b = row_rmse(y_true, pred_b)
    sq_a = np.square(np.clip(pred_a.to_numpy(dtype=np.float64), 0.0, 1.0) - y_true[pred_a.columns].to_numpy(dtype=np.float64))
    sq_b = np.square(np.clip(pred_b.to_numpy(dtype=np.float64), 0.0, 1.0) - y_true[pred_b.columns].to_numpy(dtype=np.float64))
    rows: list[dict[str, float | int]] = []

    for bin_id in range(len(edges) - 1):
        left, right = float(edges[bin_id]), float(edges[bin_id + 1])
        if bin_id == len(edges) - 2:
            mask = (humidity >= left) & (humidity <= right)
            test_mask = (test_humidity >= left) & (test_humidity <= right)
        else:
            mask = (humidity >= left) & (humidity < right)
            test_mask = (test_humidity >= left) & (test_humidity < right)
        count = int(mask.sum())
        test_count = int(test_mask.sum())
        if count == 0:
            loss_a = np.nan
            loss_b = np.nan
            mean_rmse_a = np.nan
            mean_rmse_b = np.nan
        else:
            loss_a = float(np.sqrt(np.mean(row_weights[mask, None] * sq_a[mask])))
            loss_b = float(np.sqrt(np.mean(row_weights[mask, None] * sq_b[mask])))
            mean_rmse_a = float(rmse_a[mask].mean())
            mean_rmse_b = float(rmse_b[mask].mean())
        rows.append(
            {
                "bin_id": int(bin_id),
                "humidity_left": left,
                "humidity_right": right,
                "humidity_mid": (left + right) / 2.0,
                "train_count": count,
                "test_count": test_count,
                "model_a_wrmse": loss_a,
                "model_b_wrmse": loss_b,
                "model_b_minus_model_a": loss_b - loss_a if count else np.nan,
                "model_a_row_rmse_mean": mean_rmse_a,
                "model_b_row_rmse_mean": mean_rmse_b,
            }
        )
    return pd.DataFrame(rows)


def map_x(value: float, *, left: float, width: float) -> float:
    return left + value * width


def map_y(value: float, *, top: float, height: float, cap: float) -> float:
    value = min(max(value, 0.0), cap)
    return top + height - (value / cap) * height


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_svg(
    curve: pd.DataFrame,
    figure_path: Path,
    *,
    title: str,
    model_a_label: str,
    model_b_label: str,
    loss_axis_cap: float,
    histogram_density: bool,
) -> None:
    width, height = 1500, 780
    left, right, top, bottom = 95, 75, 86, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    bin_width = float((curve["humidity_right"] - curve["humidity_left"]).iloc[0])
    train_total = max(1.0, float(curve["train_count"].sum()))
    test_total = max(1.0, float(curve["test_count"].sum()))
    curve = curve.copy()
    curve["train_density"] = curve["train_count"] / train_total / bin_width
    curve["test_density"] = curve["test_count"] / test_total / bin_width
    max_count = max(1, int(max(curve["train_count"].max(), curve["test_count"].max())))
    max_density = max(1e-12, float(max(curve["train_density"].max(), curve["test_density"].max())))
    dist_h = plot_h * 0.42

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        svg_text(width / 2, 38, title, size=24),
        svg_text(width / 2, 62, "Courbes = WRMSE par bin; histogrammes train/test en fond", size=12, color="#687076"),
    ]

    for tick in np.linspace(0, loss_axis_cap, 7):
        y = map_y(float(tick), top=top, height=plot_h, cap=loss_axis_cap)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e1db" stroke-width="1"/>')
        lines.append(svg_text(left - 12, y + 4, f"{tick:.3f}", size=11, anchor="end", color="#687076"))
    lines.append(svg_text(30, top + plot_h / 2, "WRMSE", size=13, color="#687076"))

    bar_w = plot_w / len(curve) * 0.92
    for _, row in curve.iterrows():
        x = map_x(float(row["humidity_mid"]), left=left, width=plot_w)
        if histogram_density:
            train_h = (float(row["train_density"]) / max_density) * dist_h
            test_h = (float(row["test_density"]) / max_density) * dist_h
        else:
            train_h = (float(row["train_count"]) / max_count) * dist_h
            test_h = (float(row["test_count"]) / max_count) * dist_h
        y_train = top + plot_h - train_h
        y_test = top + plot_h - test_h
        lines.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y_train:.1f}" width="{bar_w:.1f}" height="{train_h:.1f}" fill="#4C78A8" opacity="0.22"/>')
        lines.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y_test:.1f}" width="{bar_w:.1f}" height="{test_h:.1f}" fill="#F58518" opacity="0.28"/>')
        if (not histogram_density) and int(row["bin_id"]) % 2 == 0:
            label_y = min(y_train, y_test) - 7
            lines.append(svg_text(x, label_y, f"Tn {int(row['train_count'])//1000}k / Te {int(row['test_count'])//1000}k", size=9, color="#5E666C"))

    points_a = []
    points_b = []
    for _, row in curve.iterrows():
        if np.isfinite(row["model_a_wrmse"]):
            x = map_x(float(row["humidity_mid"]), left=left, width=plot_w)
            points_a.append((x, map_y(float(row["model_a_wrmse"]), top=top, height=plot_h, cap=loss_axis_cap)))
            points_b.append((x, map_y(float(row["model_b_wrmse"]), top=top, height=plot_h, cap=loss_axis_cap)))
    lines.append(f'<polyline points="{polyline(points_a)}" fill="none" stroke="{COLORS["model_a"]}" stroke-width="3.4"/>')
    lines.append(f'<polyline points="{polyline(points_b)}" fill="none" stroke="{COLORS["model_b"]}" stroke-width="3.4"/>')

    for x, y in points_a:
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="{COLORS["model_a"]}"/>')
    for x, y in points_b:
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="{COLORS["model_b"]}"/>')

    for x_tick in np.linspace(0, 1, 11):
        x = map_x(float(x_tick), left=left, width=plot_w)
        lines.append(f'<line x1="{x:.1f}" y1="{top + plot_h:.1f}" x2="{x:.1f}" y2="{top + plot_h + 7:.1f}" stroke="#777" stroke-width="1"/>')
        lines.append(svg_text(x, top + plot_h + 28, f"{x_tick:.1f}", size=11, color="#687076"))
    lines.append(svg_text(left + plot_w / 2, height - 34, "Humidity", size=14))

    legend_x, legend_y = left, height - 82
    lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 50}" y2="{legend_y}" stroke="{COLORS["model_a"]}" stroke-width="3.4"/>')
    lines.append(svg_text(legend_x + 62, legend_y + 4, model_a_label, size=12, anchor="start"))
    lines.append(f'<line x1="{legend_x + 280}" y1="{legend_y}" x2="{legend_x + 330}" y2="{legend_y}" stroke="{COLORS["model_b"]}" stroke-width="3.4"/>')
    lines.append(svg_text(legend_x + 342, legend_y + 4, model_b_label, size=12, anchor="start"))
    lines.append(f'<rect x="{legend_x + 565}" y="{legend_y - 12}" width="32" height="20" fill="#4C78A8" opacity="0.22"/>')
    lines.append(svg_text(legend_x + 607, legend_y + 4, "train density" if histogram_density else "train distribution", size=12, anchor="start"))
    lines.append(f'<rect x="{legend_x + 765}" y="{legend_y - 12}" width="32" height="20" fill="#F58518" opacity="0.28"/>')
    lines.append(svg_text(legend_x + 807, legend_y + 4, "test density" if histogram_density else "test distribution", size=12, anchor="start"))

    # Right-side distribution scale.
    if histogram_density:
        for tick in np.linspace(0, max_density, 5):
            y = top + plot_h - (float(tick) / max_density) * dist_h
            lines.append(svg_text(left + plot_w + 12, y + 4, f"{tick:.1f}", size=10, anchor="start", color="#8B8F92"))
        lines.append(svg_text(width - 22, top + plot_h - dist_h / 2, "Density", size=12, color="#8B8F92"))
    else:
        for tick in np.linspace(0, max_count, 5):
            y = top + plot_h - (float(tick) / max_count) * dist_h
            lines.append(svg_text(left + plot_w + 12, y + 4, f"{int(tick / 1000)}k", size=10, anchor="start", color="#8B8F92"))
        lines.append(svg_text(width - 22, top + plot_h - dist_h / 2, "Rows", size=12, color="#8B8F92"))

    best_rows = curve.dropna(subset=["model_b_minus_model_a"])
    if len(best_rows):
        mean_delta = float(best_rows["model_b_minus_model_a"].mean())
        lines.append(svg_text(width - 330, 98, f"Delta moyen RF - best = {mean_delta:+.5f}", size=13, anchor="start", color="#33383D"))

    lines.append("</svg>")
    figure_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    model_a_dir = resolve_path(args.model_a_dir)
    model_b_dir = resolve_path(args.model_b_dir)
    output_dir = model_b_dir if args.output_dir is None else resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = standardize_feature_columns(pd.read_csv(data_dir / "X_train.csv"))[["Humidity"]]
    x_test = standardize_feature_columns(pd.read_csv(data_dir / "X_test.csv"))[["Humidity"]]
    y_train_raw = standardize_target_columns(pd.read_csv(data_dir / "y_train.csv"))
    y_train = y_train_raw.drop(columns=["ID"]) if "ID" in y_train_raw.columns else y_train_raw
    pred_a = load_oof_predictions(model_a_dir / args.model_a_oof, y_train)
    pred_b = load_oof_predictions(model_b_dir / args.model_b_oof, y_train)
    common_targets = [target for target in pred_a.columns if target in pred_b.columns]
    if not common_targets:
        raise ValueError("No common target columns between the two OOF files.")
    pred_a = pred_a[common_targets]
    pred_b = pred_b[common_targets]
    y_model = y_train[common_targets]
    humidity = x_train["Humidity"].to_numpy(dtype=np.float64)
    row_weights = row_weights_for_humidity(humidity, args.loss_weighting)

    curve = bin_loss_curve(
        humidity,
        x_test["Humidity"].to_numpy(dtype=np.float64),
        y_model,
        pred_a,
        pred_b,
        row_weights=row_weights,
        bin_width=float(args.bin_width),
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"{args.report_prefix}_{timestamp}.csv"
    figure_path = output_dir / f"{args.report_prefix}_{timestamp}.svg"
    summary_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
    curve.to_csv(csv_path, index=False)
    write_svg(
        curve,
        figure_path,
        title=args.title,
        model_a_label=args.model_a_label,
        model_b_label=args.model_b_label,
        loss_axis_cap=float(args.loss_axis_cap),
        histogram_density=bool(args.histogram_density),
    )
    summary = {
        "generated_at_utc": timestamp,
        "figure_path": display_path(figure_path),
        "curve_csv_path": display_path(csv_path),
        "model_a": {
            "label": args.model_a_label,
            "oof_path": display_path(model_a_dir / args.model_a_oof),
        },
        "model_b": {
            "label": args.model_b_label,
            "oof_path": display_path(model_b_dir / args.model_b_oof),
        },
        "bin_width": float(args.bin_width),
        "loss_weighting": args.loss_weighting,
        "common_targets": common_targets,
        "mean_delta_model_b_minus_model_a": float(curve["model_b_minus_model_a"].dropna().mean()),
        "bins_where_model_b_better": int((curve["model_b_minus_model_a"] < 0).sum()),
        "bins_where_model_a_better": int((curve["model_b_minus_model_a"] > 0).sum()),
        "total_train_rows": int(curve["train_count"].sum()),
        "total_test_rows": int(curve["test_count"].sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
