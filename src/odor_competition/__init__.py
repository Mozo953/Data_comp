"""Utilities for the odor-detection competition workflow."""

from .data import (
    COMPETITION_TARGETS,
    CompetitionData,
    ModelingDataBundle,
    TargetSchema,
    build_submission_frame,
    feature_target_signal,
    engineer_features,
    infer_target_schema,
    load_competition_data,
    load_modeling_data,
    prune_correlated_features,
    raw_features,
)
from .data_shift import ImportanceWeightingResult, compute_soft_test_proximity_weights
from .metrics import competition_rmse

try:
    from .reporting import plot_smoothed_empirical_density
except ModuleNotFoundError:
    plot_smoothed_empirical_density = None

__all__ = [
    "COMPETITION_TARGETS",
    "CompetitionData",
    "ImportanceWeightingResult",
    "ModelingDataBundle",
    "TargetSchema",
    "build_submission_frame",
    "competition_rmse",
    "compute_soft_test_proximity_weights",
    "feature_target_signal",
    "engineer_features",
    "infer_target_schema",
    "load_competition_data",
    "load_modeling_data",
    "prune_correlated_features",
    "raw_features",
]

if plot_smoothed_empirical_density is not None:
    __all__.append("plot_smoothed_empirical_density")
