from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/test humidity distributions and OOF blend loss by humidity."
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--oof-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--title", default="Humidity : train vs test + loss")
    parser.add_argument("--count-axis-cap", type=float, default=21000.0)
    parser.add_argument("--loss-axis-cap", type=float, default=0.09)
    parser.add_argument("--report-prefix", default="humidity_loss")
    parser.add_argument(
        "--loss-weighting",
        choices=["binary-env", "model46", "sample-file", "none"],
        default="binary-env",
        help="binary-env: 1.2 if Env>=0.6; model46: 1.4 if Env>=0.4; sample-file: read sample_weights_*.csv; none: unweighted.",
    )
    parser.add_argument("--sample-weights-file", default=None)
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if len(values) <= 1:
        return values.astype(np.float64)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / max(sigma, 1e-6)))
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def polygon_points(
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
    close_baseline: bool,
) -> str:
    if y_max <= y_min:
        y_max = y_min + 1e-6
    pts: list[str] = []
    if close_baseline:
        x0 = left + ((float(x_values[0]) - x_min) / (x_max - x_min)) * width
        pts.append(f"{x0:.1f},{top + height:.1f}")
    for x_val, y_val in zip(x_values, y_values):
        x = left + ((float(x_val) - x_min) / (x_max - x_min)) * width
        y = top + height - ((float(y_val) - y_min) / (y_max - y_min)) * height
        pts.append(f"{x:.1f},{y:.1f}")
    if close_baseline:
        x1 = left + ((float(x_values[-1]) - x_min) / (x_max - x_min)) * width
        pts.append(f"{x1:.1f},{top + height:.1f}")
    return " ".join(pts)


def format_count_tick(value: float) -> str:
    if value >= 1000:
        if abs(value % 1000) < 1e-9:
            return f"{int(round(value / 1000.0))}k"
        return f"{value / 1000.0:.1f}k"
    return str(int(round(value)))


def find_oof_file(source_dir: Path, explicit_name: str | None) -> Path:
    if explicit_name:
        path = source_dir / explicit_name
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    patterns = [
        "*_oof_blend_modelspace.csv",
        "blend_model_*.csv",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(source_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No OOF blend file found in {source_dir}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def load_oof_predictions(oof_path: Path, y_train: pd.DataFrame) -> pd.DataFrame:
    oof = pd.read_csv(oof_path)
    if "Unnamed: 0" in oof.columns:
        oof = oof.rename(columns={"Unnamed: 0": "row_index"})
    if "row_index" in oof.columns:
        oof = oof.set_index("row_index")
    else:
        oof.index = np.arange(len(oof))
    model_cols = [column for column in oof.columns if column in y_train.columns]
    if not model_cols:
        raise ValueError(f"No target columns from y_train found in {oof_path}")
    return oof[model_cols].reindex(y_train.index).astype(np.float64)


def row_weights_for_env(env_values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(env_values, dtype=np.float64)
    if mode == "model46":
        return np.where(env_values >= 0.4, 1.4, 1.0).astype(np.float64)
    return np.where(env_values >= 0.6, 1.2, 1.0).astype(np.float64)


def load_sample_weights(source_dir: Path, explicit_name: str | None, expected_len: int) -> tuple[np.ndarray, str]:
    if explicit_name:
        path = source_dir / explicit_name
        if not path.exists():
            raise FileNotFoundError(path)
    else:
        candidates = sorted(source_dir.glob("sample_weights_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No sample_weights_*.csv found in {source_dir}")
        path = sorted(candidates, key=lambda item: item.stat().st_mtime)[-1]

    frame = pd.read_csv(path)
    value_columns = [column for column in frame.columns if column.lower() not in {"unnamed: 0", "index", "row_index"}]
    if not value_columns:
        raise ValueError(f"No sample-weight value column found in {path}")
    weights = frame[value_columns[-1]].to_numpy(dtype=np.float64)
    if len(weights) != expected_len:
        raise ValueError(f"Sample weight length mismatch: {len(weights)} != {expected_len}")
    return weights, path.name


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    source_dir = resolve_path(args.source_dir)
    output_dir = source_dir if args.output_dir is None else resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_path = find_oof_file(source_dir, args.oof_file)
    x_train = pd.read_csv(data_dir / "X_train.csv", usecols=["Env"])
    x_test = pd.read_csv(data_dir / "X_test.csv", usecols=["Env"])
    y_train_raw = pd.read_csv(data_dir / "y_train.csv")
    y_train = y_train_raw.drop(columns=["ID"]) if "ID" in y_train_raw.columns else y_train_raw
    oof = load_oof_predictions(oof_path, y_train)
    y_model = y_train[oof.columns].copy()

    train_env = x_train["Env"].to_numpy(dtype=np.float64)
    test_env = x_test["Env"].to_numpy(dtype=np.float64)
    sample_weight_source = None
    if args.loss_weighting == "sample-file":
        row_weights, sample_weight_source = load_sample_weights(source_dir, args.sample_weights_file, len(train_env))
    else:
        row_weights = row_weights_for_env(train_env, args.loss_weighting)
    row_rmse = np.sqrt(
        np.mean(
            np.square(np.clip(oof.to_numpy(dtype=np.float64), 0.0, 1.0) - y_model.to_numpy(dtype=np.float64)),
            axis=1,
        )
    )
    weighted_row_rmse = row_rmse * row_weights

    edges = np.linspace(0.0, 1.0, 51)
    mids = (edges[:-1] + edges[1:]) / 2.0
    train_counts, _ = np.histogram(train_env, bins=edges)
    test_counts, _ = np.histogram(test_env, bins=edges)
    train_curve = gaussian_smooth(train_counts, sigma=2.0)
    test_curve = gaussian_smooth(test_counts, sigma=2.0)

    loss_sum, _ = np.histogram(train_env, bins=edges, weights=weighted_row_rmse)
    loss_count, _ = np.histogram(train_env, bins=edges)
    loss_per_bin = np.divide(
        loss_sum,
        np.clip(loss_count, 1, None),
        out=np.zeros_like(loss_sum, dtype=np.float64),
        where=loss_count > 0,
    )
    loss_curve = gaussian_smooth(loss_per_bin, sigma=1.8)

    width = 1360
    height = 820
    margin_left = 100
    margin_right = 100
    margin_top = 72
    margin_bottom = 96
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = -0.05
    x_max = 1.05
    count_max = float(args.count_axis_cap)
    loss_max = float(args.loss_axis_cap)

    def x_pos(x: float) -> float:
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_width

    def y_left(v: float) -> float:
        return margin_top + plot_height - (v / max(count_max, 1e-6)) * plot_height

    def y_right(v: float) -> float:
        return margin_top + plot_height - (v / max(loss_max, 1e-6)) * plot_height

    curve_x = np.concatenate(([0.0], mids, [1.0]))
    train_plot = np.concatenate(([0.0], np.clip(train_curve, 0.0, count_max), [0.0]))
    test_plot = np.concatenate(([0.0], np.clip(test_curve, 0.0, count_max), [0.0]))
    loss_plot = np.clip(loss_curve, 0.0, loss_max)

    train_poly = polygon_points(curve_x, train_plot, x_min=x_min, x_max=x_max, y_min=0.0, y_max=count_max, left=margin_left, top=margin_top, width=plot_width, height=plot_height, close_baseline=True)
    test_poly = polygon_points(curve_x, test_plot, x_min=x_min, x_max=x_max, y_min=0.0, y_max=count_max, left=margin_left, top=margin_top, width=plot_width, height=plot_height, close_baseline=True)
    train_line = polygon_points(curve_x, train_plot, x_min=x_min, x_max=x_max, y_min=0.0, y_max=count_max, left=margin_left, top=margin_top, width=plot_width, height=plot_height, close_baseline=False)
    test_line = polygon_points(curve_x, test_plot, x_min=x_min, x_max=x_max, y_min=0.0, y_max=count_max, left=margin_left, top=margin_top, width=plot_width, height=plot_height, close_baseline=False)
    loss_line = polygon_points(mids, loss_plot, x_min=x_min, x_max=x_max, y_min=0.0, y_max=loss_max, left=margin_left, top=margin_top, width=plot_width, height=plot_height, close_baseline=False)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8f8f8"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#2d3748">{args.title}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fcfcfc" stroke="#c7c7c7" stroke-width="1.2"/>',
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        x = x_pos(float(tick))
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#d6d6d6" stroke-width="1"/>')
        elements.append(f'<text x="{x:.1f}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-size="13" font-family="Arial">{tick:.1f}</text>')
    for tick in np.linspace(0.0, count_max, 6):
        y = y_left(float(tick))
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" font-weight="600" fill="#374151">{format_count_tick(float(tick))}</text>')
    for tick in np.linspace(0.0, loss_max, 6):
        y = y_right(float(tick))
        elements.append(f'<text x="{margin_left + plot_width + 12}" y="{y + 4:.1f}" text-anchor="start" font-size="12" font-family="Arial" fill="#dc2626">{tick:.3f}</text>')

    elements.extend([
        f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="Arial">Humidity</text>',
        f'<text x="28" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" fill="#2d3748" transform="rotate(-90 28 {margin_top + plot_height / 2:.1f})">Number of rows</text>',
        f'<text x="{width - 28}" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" fill="#dc2626" transform="rotate(90 {width - 28} {margin_top + plot_height / 2:.1f})">Loss</text>',
    ])

    for boundary in [0.2, 0.6, 1.0]:
        x = x_pos(boundary)
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#b9b9b9" stroke-width="1.2" stroke-dasharray="6,6"/>')

    elements.extend([
        f'<polygon points="{train_poly}" fill="#4f83cc" fill-opacity="0.28" stroke="none"/>',
        f'<polyline points="{train_line}" fill="none" stroke="#ffffff" stroke-width="6"/>',
        f'<polyline points="{train_line}" fill="none" stroke="#3b6fb6" stroke-width="2.8"/>',
        f'<polygon points="{test_poly}" fill="#f4a261" fill-opacity="0.26" stroke="none"/>',
        f'<polyline points="{test_line}" fill="none" stroke="#ffffff" stroke-width="6"/>',
        f'<polyline points="{test_line}" fill="none" stroke="#ee8e3a" stroke-width="2.8"/>',
        f'<polyline points="{loss_line}" fill="none" stroke="#ffffff" stroke-width="7"/>',
        f'<polyline points="{loss_line}" fill="none" stroke="#dc2626" stroke-width="3.2"/>',
    ])

    legend_x = margin_left + plot_width - 250
    legend_y = margin_top + 24
    elements.append(f'<rect x="{legend_x - 18}" y="{legend_y - 18}" width="230" height="92" rx="8" fill="white" fill-opacity="0.94" stroke="#d0d0d0"/>')
    for i, (label, color) in enumerate([("train", "#3b6fb6"), ("test", "#ee8e3a"), ("loss", "#dc2626")]):
        y = legend_y + i * 24
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 40}" y="{y + 4}" font-size="13" font-family="Arial" fill="#1f2937">{label}</text>')

    elements.append("</svg>")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figure_path = output_dir / f"{args.report_prefix}_{timestamp}.svg"
    csv_path = output_dir / f"{args.report_prefix}_{timestamp}.csv"
    summary_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
    figure_path.write_text("\n".join(elements), encoding="utf-8")
    pd.DataFrame(
        {
            "humidity_mid": mids,
            "train_count": train_counts,
            "test_count": test_counts,
            "train_curve_smooth": train_curve,
            "test_curve_smooth": test_curve,
            "loss_curve_smooth": loss_curve,
        }
    ).to_csv(csv_path, index=False)
    summary = {
        "generated_at_utc": timestamp,
        "figure_path": str(figure_path.relative_to(ROOT)),
        "curve_csv_path": str(csv_path.relative_to(ROOT)),
        "count_axis_cap": int(args.count_axis_cap),
        "loss_axis_cap": float(args.loss_axis_cap),
        "total_train_rows": int(len(train_env)),
        "total_test_rows": int(len(test_env)),
        "loss_source_file": oof_path.name,
        "loss_weighting": args.loss_weighting,
        "sample_weight_source_file": sample_weight_source,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
