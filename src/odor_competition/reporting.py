from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def build_ydata_profile(
    dataframe: pd.DataFrame,
    output_path: Path | str,
    title: str,
    *,
    minimal: bool = False,
    correlation_focus: bool = True,
    explorative: bool = True,
    with_interactions: bool = False,
) -> Path:
    try:
        from ydata_profiling import ProfileReport
        from ydata_profiling.config import Settings
    except ImportError as exc:
        raise RuntimeError(
            "ydata-profiling is not installed. Install requirements.txt to enable HTML profiling."
        ) from exc

    config = Settings()
    config.samples.head = 5
    config.samples.tail = 5
    config.samples.random = 5

    if correlation_focus:
        config.correlations["auto"].calculate = True
        config.correlations["auto"].threshold = 0.8
        config.correlations["auto"].warn_high_correlations = 20
        config.correlations["pearson"].calculate = True
        config.correlations["pearson"].threshold = 0.8
        config.correlations["pearson"].warn_high_correlations = 20
        config.correlations["spearman"].calculate = True
        config.correlations["spearman"].threshold = 0.8
        config.correlations["spearman"].warn_high_correlations = 20
        config.interactions.continuous = with_interactions

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = ProfileReport(
        dataframe,
        title=title,
        minimal=minimal,
        explorative=explorative,
        config=config,
    )
    profile.to_file(destination)
    return destination


def plot_smoothed_empirical_density(
    p_train: pd.DataFrame | pd.Series | np.ndarray,
    p_test: pd.DataFrame | pd.Series | np.ndarray,
    *,
    output_path: Path | str | None = None,
    title: str = "Smoothed empirical density: Ptrain vs Ptest",
    train_label: str = "Ptrain",
    test_label: str = "Ptest",
    bins: int = 50,
    bandwidth: float | None = None,
    grid_size: int = 512,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot train/test empirical densities with a smoothed KDE estimate.

    The inputs can be NumPy arrays, Series, or DataFrames. When a DataFrame is
    passed, all numeric values are flattened into a single one-dimensional sample.
    """
    p_train, p_test = _align_prediction_frames(p_train, p_test)
    train_values = _flatten_numeric_values(p_train, name=train_label)
    test_values = _flatten_numeric_values(p_test, name=test_label)

    train_bandwidth = _resolve_bandwidth(train_values, bandwidth)
    test_bandwidth = _resolve_bandwidth(test_values, bandwidth)

    lower = min(train_values.min(), test_values.min())
    upper = max(train_values.max(), test_values.max())
    span = upper - lower
    padding = max(0.05 * span, 2.0 * max(train_bandwidth, test_bandwidth), 1e-3)
    if span == 0.0:
        padding = max(padding, 0.1)

    grid = np.linspace(lower - padding, upper + padding, grid_size)
    train_density = _estimate_kernel_density(train_values, grid, train_bandwidth)
    test_density = _estimate_kernel_density(test_values, grid, test_bandwidth)

    fig, ax = plt.subplots(figsize=figsize)
    hist_range = (grid[0], grid[-1])

    ax.hist(
        train_values,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=0.20,
        color="tab:blue",
        label=f"{train_label} empirical density",
    )
    ax.hist(
        test_values,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=0.20,
        color="tab:orange",
        label=f"{test_label} empirical density",
    )
    ax.plot(grid, train_density, color="tab:blue", linewidth=2.2, label=f"{train_label} smoothed KDE")
    ax.plot(grid, test_density, color="tab:orange", linewidth=2.2, label=f"{test_label} smoothed KDE")

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")

    return fig, ax


def _flatten_numeric_values(
    values: pd.DataFrame | pd.Series | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        numeric_frame = values.select_dtypes(include=["number"]).copy()
        columns_to_drop = [
            column
            for column in numeric_frame.columns
            if column == "ID" or column.lower().startswith("unnamed")
        ]
        numeric_values = numeric_frame.drop(columns=columns_to_drop, errors="ignore").to_numpy(dtype=float)
    elif isinstance(values, pd.Series):
        numeric_values = values.to_numpy(dtype=float)
    else:
        numeric_values = np.asarray(values, dtype=float)

    flattened = np.ravel(numeric_values)
    flattened = flattened[np.isfinite(flattened)]
    if flattened.size == 0:
        raise ValueError(f"{name} does not contain any finite numeric values.")
    return flattened


def _align_prediction_frames(
    p_train: pd.DataFrame | pd.Series | np.ndarray,
    p_test: pd.DataFrame | pd.Series | np.ndarray,
) -> tuple[pd.DataFrame | pd.Series | np.ndarray, pd.DataFrame | pd.Series | np.ndarray]:
    if not isinstance(p_train, pd.DataFrame) or not isinstance(p_test, pd.DataFrame):
        return p_train, p_test

    train_columns = [
        column
        for column in p_train.select_dtypes(include=["number"]).columns
        if column != "ID" and not column.lower().startswith("unnamed")
    ]
    test_columns = [
        column
        for column in p_test.select_dtypes(include=["number"]).columns
        if column != "ID" and not column.lower().startswith("unnamed")
    ]
    common_columns = [column for column in train_columns if column in test_columns]

    if common_columns:
        return p_train[common_columns].copy(), p_test[common_columns].copy()
    return p_train, p_test


def _resolve_bandwidth(values: np.ndarray, bandwidth: float | None) -> float:
    if bandwidth is not None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be strictly positive.")
        return float(bandwidth)
    return _silverman_bandwidth(values)


def _silverman_bandwidth(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.05

    std = float(np.std(values, ddof=1))
    q75, q25 = np.percentile(values, [75.0, 25.0])
    iqr = float(q75 - q25)
    robust_scale = iqr / 1.34 if iqr > 0.0 else 0.0

    scale_candidates = [candidate for candidate in (std, robust_scale) if np.isfinite(candidate) and candidate > 0.0]
    if not scale_candidates:
        unique_values = np.unique(values)
        if unique_values.size < 2:
            return 0.05
        scale = float(np.std(unique_values, ddof=1))
    else:
        scale = min(scale_candidates)

    bandwidth = 0.9 * scale * (values.size ** (-1.0 / 5.0))
    return max(float(bandwidth), 1e-3)


def _estimate_kernel_density(values: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(values.reshape(-1, 1))
    log_density = kde.score_samples(grid.reshape(-1, 1))
    return np.exp(log_density)
