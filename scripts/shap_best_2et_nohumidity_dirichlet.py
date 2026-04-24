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

import best_2et_nohumidity_core as clean  # noqa: E402
import train_best_2et_nohumidity_dirichlet as best_pipeline  # noqa: E402
from gaz_competition.data import load_modeling_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SHAP importances for the 2 ExtraTrees used in the best "
            "nohumidity Dirichlet gas-detection pipeline."
        )
    )
    parser.add_argument("--data-dir", default="src/gaz_competition/data")
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/Best_models__2ET_nohumidty_dirichlet_0.1391/shap",
    )
    parser.add_argument("--artifact-prefix", default="best_2et_nohumidity_dirichlet_shap")
    parser.add_argument("--model", choices=["rowagg", "allpool", "both"], default="both")
    parser.add_argument("--target", default=None, help="Optional modeled target name, for example c01.")
    parser.add_argument("--weight-preset", choices=["model49", "model50", "model50_low02_115"], default="model50")
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-shap-rows", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_shap_rows <= 0:
        raise ValueError("--max-shap-rows must be > 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0.")
    return args


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_shap():
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'shap'. Install it with `pip install -r requirements.txt`."
        ) from exc
    return shap


def load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def normalize_shap_values(
    shap_values,
    *,
    sample_count: int,
    feature_count: int,
    target_count: int,
) -> np.ndarray:
    if isinstance(shap_values, list):
        stacked = [np.asarray(values, dtype=np.float32) for values in shap_values]
        if any(values.shape != (sample_count, feature_count) for values in stacked):
            raise ValueError("Unexpected SHAP list element shape.")
        return np.stack(stacked, axis=2)

    values = np.asarray(shap_values, dtype=np.float32)
    if values.ndim == 2:
        if values.shape != (sample_count, feature_count):
            raise ValueError(f"Unexpected SHAP shape: {values.shape}")
        return values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP shape: {values.shape}")

    if values.shape == (sample_count, feature_count, target_count):
        return values
    if values.shape == (target_count, sample_count, feature_count):
        return np.moveaxis(values, 0, 2)
    if values.shape == (sample_count, target_count, feature_count):
        return np.moveaxis(values, 1, 2)
    raise ValueError(f"Unsupported SHAP 3D shape: {values.shape}")


def select_target_name(bundle, raw_target: str | None) -> str | None:
    if raw_target is None:
        return None
    if raw_target in bundle.schema.constant_targets:
        raise ValueError(
            f"Target {raw_target} is constant in training data and is not explained by the tree models."
        )
    representative = bundle.schema.representative_for_target.get(raw_target, raw_target)
    if representative not in bundle.y_train_model.columns:
        raise ValueError(
            f"Unknown target {raw_target}. Available modeled targets: {', '.join(bundle.y_train_model.columns)}"
        )
    return representative


def sample_frame(frame: pd.DataFrame, *, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=random_state).sort_index()


def mean_abs_shap_importance(
    shap_array: np.ndarray,
    feature_names: list[str],
    *,
    target_names: list[str],
    selected_target: str | None,
) -> pd.DataFrame:
    if selected_target is None:
        mean_abs = np.abs(shap_array).mean(axis=(0, 2))
        target_label = "all_modeled_targets"
    else:
        target_idx = target_names.index(selected_target)
        mean_abs = np.abs(shap_array[:, :, target_idx]).mean(axis=0)
        target_label = selected_target

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs.astype(np.float64),
            "target_scope": target_label,
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)


def save_importance_plot(importance: pd.DataFrame, *, title: str, top_k: int, output_path: Path) -> None:
    plt = load_pyplot()
    if plt is None:
        raise ModuleNotFoundError("matplotlib.pyplot is not available.")
    top_frame = importance.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(top_frame) + 1.4)))
    ax.barh(top_frame["feature"], top_frame["mean_abs_shap"], color="#2F6B7C")
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_ylabel("feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_models_for_shap(args: argparse.Namespace):
    bundle = clean.maybe_subsample_bundle(
        load_modeling_data(resolve_path(args.data_dir)),
        max_train_rows=args.max_train_rows,
        max_test_rows=None,
    )
    humidity_train = bundle.data.x_train["Humidity"].copy()
    row_weights = best_pipeline.compute_piecewise_humidity_weights(
        humidity_train,
        preset=str(args.weight_preset),
    )

    x_train_nohumidity = clean.drop_humidity_columns(bundle.data.x_train)
    clean.validate_no_humidity_columns(x_train_nohumidity, "x_train_nohumidity")
    views = clean.build_feature_views(
        x_train_nohumidity,
        x_train_nohumidity,
        tail_quantile=float(args.tail_quantile),
        ratio_eps=float(args.ratio_eps),
    )

    selected_models = ["rowagg", "allpool"] if args.model == "both" else [args.model]
    fitted = {}
    if "rowagg" in selected_models:
        rowagg_model = clean.make_rowagg_model(int(args.n_jobs))
        rowagg_model.fit(
            views.rowagg_fit,
            bundle.y_train_model,
            sample_weight=row_weights.to_numpy(dtype=np.float32),
        )
        fitted["rowagg"] = {
            "artifact_name": "et_rowagg_mf06_bs",
            "display_name": "ExtraTrees rowagg",
            "model": rowagg_model,
            "features": views.rowagg_fit,
        }

    if "allpool" in selected_models:
        allpool_model = clean.make_allpool_model(int(args.n_jobs))
        allpool_model.fit(
            views.allpool_fit,
            bundle.y_train_model,
            sample_weight=row_weights.to_numpy(dtype=np.float32),
        )
        fitted["allpool"] = {
            "artifact_name": "et_allpool_3",
            "display_name": "ExtraTrees allpool",
            "model": allpool_model,
            "features": views.allpool_fit,
        }

    return bundle, fitted


def main() -> None:
    args = parse_args()
    shap = load_shap()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = best_pipeline.safe_file_stem(args.artifact_prefix)

    bundle, fitted_models = fit_models_for_shap(args)
    selected_target = select_target_name(bundle, args.target)

    summary: dict[str, object] = {
        "generated_at_utc": timestamp,
        "pipeline_reference": "Best 2ET nohumidity Dirichlet gas-detection pipeline",
        "note": (
            "The final winner is a target-wise Dirichlet blend of two ExtraTrees models. "
            "SHAP is therefore computed on each base tree model separately."
        ),
        "data_dir": display_path(resolve_path(args.data_dir)),
        "output_dir": display_path(output_dir),
        "weight_preset": str(args.weight_preset),
        "target_scope": selected_target or "all_modeled_targets",
        "models": {},
    }

    for _, payload in fitted_models.items():
        feature_frame = payload["features"]
        shap_frame = sample_frame(
            feature_frame,
            max_rows=int(args.max_shap_rows),
            random_state=int(args.random_state),
        )
        if args.verbose:
            best_pipeline.log_progress(
                f"Computing SHAP for {payload['artifact_name']} on {len(shap_frame)} rows and {shap_frame.shape[1]} features"
            )

        explainer = shap.TreeExplainer(payload["model"])
        raw_shap_values = explainer.shap_values(shap_frame, check_additivity=False)
        shap_array = normalize_shap_values(
            raw_shap_values,
            sample_count=len(shap_frame),
            feature_count=shap_frame.shape[1],
            target_count=len(bundle.y_train_model.columns),
        )

        importance = mean_abs_shap_importance(
            shap_array,
            list(shap_frame.columns),
            target_names=list(bundle.y_train_model.columns),
            selected_target=selected_target,
        )

        scope_label = selected_target or "all_targets"
        csv_path = output_dir / f"{prefix}_{timestamp}_{payload['artifact_name']}_{scope_label}_importance.csv"
        png_path = output_dir / f"{prefix}_{timestamp}_{payload['artifact_name']}_{scope_label}_top{int(args.top_k)}.png"
        sample_path = output_dir / f"{prefix}_{timestamp}_{payload['artifact_name']}_{scope_label}_sample_index.csv"

        importance.to_csv(csv_path, index=False)
        pd.DataFrame({"row_index": shap_frame.index}).to_csv(sample_path, index=False)
        plot_generated = False
        plot_error = None
        try:
            save_importance_plot(
                importance,
                title=f"{payload['display_name']} SHAP importance ({scope_label})",
                top_k=int(args.top_k),
                output_path=png_path,
            )
            plot_generated = True
        except ModuleNotFoundError as exc:
            plot_error = str(exc)

        summary["models"][payload["artifact_name"]] = {
            "feature_count": int(shap_frame.shape[1]),
            "rows_used_for_shap": int(len(shap_frame)),
            "importance_csv_path": display_path(csv_path),
            "topk_plot_path": display_path(png_path) if plot_generated else None,
            "plot_generated": plot_generated,
            "plot_error": plot_error,
            "sample_index_path": display_path(sample_path),
            "top_features": importance.head(int(args.top_k)).to_dict(orient="records"),
        }

    summary_path = output_dir / f"{prefix}_{timestamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_pipeline.log_progress(f"SHAP artifacts written to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
