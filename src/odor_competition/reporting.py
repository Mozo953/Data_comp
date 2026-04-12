from __future__ import annotations

from pathlib import Path

import pandas as pd


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
