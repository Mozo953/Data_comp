from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_FEATURES = [
    "Env",
    "X12",
    "X13",
    "X14",
    "X15",
    "X4",
    "X5",
    "X6",
    "X7",
    "Z",
    "Y1",
    "Y2",
    "Y3",
]
MODELING_FEATURES = [feature for feature in COMPETITION_FEATURES if feature != "Env"]
COMPETITION_TARGETS = [f"d{i:02d}" for i in range(1, 24)]


@dataclass(frozen=True)
class CompetitionData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame


@dataclass(frozen=True)
class ModelingDataBundle:
    data: CompetitionData
    schema: "TargetSchema"
    x_train_raw: pd.DataFrame
    x_test_raw: pd.DataFrame
    y_train_full: pd.DataFrame
    y_train_model: pd.DataFrame


@dataclass(frozen=True)
class FeatureCleaningProfile:
    lower_bounds: pd.Series
    upper_bounds: pd.Series


@dataclass(frozen=True)
class TargetSchema:
    original_targets: list[str]
    model_targets: list[str]
    duplicate_groups: list[list[str]]
    representative_for_target: dict[str, str]
    constant_targets: dict[str, float]

    def expand_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        expanded = {}
        for target in self.original_targets:
            if target in self.constant_targets:
                expanded[target] = np.full(len(predictions), self.constant_targets[target])
            else:
                representative = self.representative_for_target[target]
                expanded[target] = predictions[representative].to_numpy()
        return pd.DataFrame(expanded, columns=self.original_targets, index=predictions.index)


def load_competition_data(data_dir: Path | str = ".") -> CompetitionData:
    data_path = Path(data_dir)
    x_train = pd.read_csv(data_path / "X_train.csv")
    x_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv")

    if not x_train["ID"].equals(y_train["ID"]):
        raise ValueError("Training feature IDs do not match training target IDs.")

    return CompetitionData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
    )


def load_modeling_data(data_dir: Path | str = ".") -> ModelingDataBundle:
    """Load and prepare competition data in a single reusable call.

    Returns raw X features, full Y targets (without ID), and the modeled targets
    according to inferred duplicate/constant-target schema.
    """
    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    x_train_raw, x_test_raw, _ = prepare_feature_frames(data.x_train, data.x_test)
    y_train_full = data.y_train.drop(columns=["ID"]).copy() if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    return ModelingDataBundle(
        data=data,
        schema=schema,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
        y_train_full=y_train_full,
        y_train_model=y_train_model,
    )


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    base = features.copy()
    if "ID" in base.columns:
        base = base.drop(columns=["ID"])

    block_a = ["X12", "X13", "X14", "X15"]
    block_b = ["X4", "X5", "X6", "X7"]
    block_c = ["Y1", "Y2", "Y3", "Z"]

    engineered = base.copy()
    for name, columns in {
        "block_a": block_a,
        "block_b": block_b,
        "support": block_c,
    }.items():
        engineered[f"{name}_mean"] = base[columns].mean(axis=1)
        engineered[f"{name}_std"] = base[columns].std(axis=1)
        engineered[f"{name}_min"] = base[columns].min(axis=1)
        engineered[f"{name}_max"] = base[columns].max(axis=1)
        engineered[f"{name}_range"] = engineered[f"{name}_max"] - engineered[f"{name}_min"]
        engineered[f"{name}_energy"] = np.square(base[columns]).sum(axis=1)

    aligned_pairs = [("X12", "X4"), ("X13", "X5"), ("X14", "X6"), ("X15", "X7")]
    for left, right in aligned_pairs:
        engineered[f"{left}_minus_{right}"] = base[left] - base[right]
        engineered[f"{left}_plus_{right}"] = base[left] + base[right]

    sequential_pairs = [("X12", "X13"), ("X13", "X14"), ("X14", "X15"), ("X4", "X5"), ("X5", "X6"), ("X6", "X7")]
    for left, right in sequential_pairs:
        engineered[f"{left}_minus_{right}"] = base[left] - base[right]

    engineered["block_gap"] = engineered["block_a_mean"] - engineered["block_b_mean"]
    engineered["block_energy_gap"] = engineered["block_a_energy"] - engineered["block_b_energy"]
    engineered["env_times_block_a"] = base["Env"] * engineered["block_a_mean"]
    engineered["env_times_block_b"] = base["Env"] * engineered["block_b_mean"]
    engineered["env_times_support"] = base["Env"] * engineered["support_mean"]
    engineered["z_minus_y_mean"] = base["Z"] - base[["Y1", "Y2", "Y3"]].mean(axis=1)
    engineered["y_spread"] = base[["Y1", "Y2", "Y3"]].max(axis=1) - base[["Y1", "Y2", "Y3"]].min(axis=1)

    return engineered


def engineer_env_focus_features(features: pd.DataFrame) -> pd.DataFrame:
    base = _extract_feature_block(features, include_env=True)
    env_focus = base.copy()

    block_groups = {
        "block_a": ["X12", "X13", "X14", "X15"],
        "block_b": ["X4", "X5", "X6", "X7"],
        "support": ["Y1", "Y2", "Y3", "Z"],
    }

    block_means: dict[str, pd.Series] = {}
    for name, columns in block_groups.items():
        block_mean = base[columns].mean(axis=1)
        block_means[name] = block_mean
        env_focus[f"{name}_mean"] = block_mean
        env_focus[f"{name}_range"] = base[columns].max(axis=1) - base[columns].min(axis=1)

    env_focus["block_gap"] = block_means["block_a"] - block_means["block_b"]
    env_focus["support_gap"] = block_means["support"] - base["Env"]
    env_focus["env_times_block_a"] = base["Env"] * block_means["block_a"]
    env_focus["env_times_block_b"] = base["Env"] * block_means["block_b"]
    env_focus["env_times_support"] = base["Env"] * block_means["support"]

    for column in COMPETITION_FEATURES:
        if column == "Env":
            continue
        env_focus[f"{column}_minus_env"] = base[column] - base["Env"]

    return env_focus


def _extract_feature_block(features: pd.DataFrame, *, include_env: bool = True) -> pd.DataFrame:
    base = features.copy()
    if "ID" in base.columns:
        base = base.drop(columns=["ID"])
    columns = COMPETITION_FEATURES if include_env else MODELING_FEATURES
    return base[columns].copy()


def fit_feature_cleaning_profile(
    x_train: pd.DataFrame,
    *,
    tail_quantile: float = 0.001,
) -> FeatureCleaningProfile:
    if not 0.0 < tail_quantile < 0.5:
        raise ValueError("tail_quantile must be between 0 and 0.5.")

    train_base = _extract_feature_block(x_train, include_env=True)
    lower_bounds = train_base.quantile(tail_quantile)
    upper_bounds = train_base.quantile(1.0 - tail_quantile)
    lower_bounds["Env"] = 0.0
    upper_bounds["Env"] = 1.0

    return FeatureCleaningProfile(
        lower_bounds=lower_bounds.reindex(COMPETITION_FEATURES),
        upper_bounds=upper_bounds.reindex(COMPETITION_FEATURES),
    )


def apply_feature_cleaning(features: pd.DataFrame, profile: FeatureCleaningProfile) -> pd.DataFrame:
    base = _extract_feature_block(features, include_env=True)
    return base.clip(lower=profile.lower_bounds, upper=profile.upper_bounds, axis="columns")


def prepare_feature_frames(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame | None = None,
    *,
    tail_quantile: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame | None, FeatureCleaningProfile]:
    profile = fit_feature_cleaning_profile(x_train, tail_quantile=tail_quantile)
    cleaned_train = apply_feature_cleaning(x_train, profile)[MODELING_FEATURES].copy()
    cleaned_test = None if x_test is None else apply_feature_cleaning(x_test, profile)[MODELING_FEATURES].copy()
    return cleaned_train, cleaned_test, profile


def raw_features(features: pd.DataFrame) -> pd.DataFrame:
    return _extract_feature_block(features, include_env=False)


def infer_target_schema(targets: pd.DataFrame) -> TargetSchema:
    if "ID" in targets.columns:
        target_frame = targets.drop(columns=["ID"])
    else:
        target_frame = targets.copy()

    original_targets = list(target_frame.columns)
    remaining = original_targets.copy()
    duplicate_groups: list[list[str]] = []
    representative_for_target: dict[str, str] = {}
    constant_targets: dict[str, float] = {}
    model_targets: list[str] = []

    while remaining:
        representative = remaining.pop(0)
        group = [representative]
        keep = []
        for candidate in remaining:
            if target_frame[representative].equals(target_frame[candidate]):
                group.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        duplicate_groups.append(group)

    for group in duplicate_groups:
        representative = group[0]
        series = target_frame[representative]
        if series.nunique(dropna=False) == 1:
            constant_targets[representative] = float(series.iloc[0])
        else:
            model_targets.append(representative)

        for target in group:
            representative_for_target[target] = representative
            if representative in constant_targets:
                constant_targets[target] = constant_targets[representative]

    return TargetSchema(
        original_targets=original_targets,
        model_targets=model_targets,
        duplicate_groups=duplicate_groups,
        representative_for_target=representative_for_target,
        constant_targets=constant_targets,
    )


def compress_targets(targets: pd.DataFrame, schema: TargetSchema) -> pd.DataFrame:
    target_frame = targets.drop(columns=["ID"]) if "ID" in targets.columns else targets
    return target_frame[schema.model_targets].copy()


def feature_target_signal(features: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    target_frame = targets.drop(columns=["ID"]) if "ID" in targets.columns else targets
    joined = pd.concat([features.reset_index(drop=True), target_frame.reset_index(drop=True)], axis=1)
    signal = joined.corr().loc[features.columns, target_frame.columns].abs().mean(axis=1)
    return signal.fillna(0.0).reindex(features.columns)


def prune_correlated_features(
    features: pd.DataFrame,
    ordered_features: list[str],
    *,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    missing = [column for column in ordered_features if column not in features.columns]
    if missing:
        raise ValueError(f"ordered_features contains columns not present in features: {missing}")

    correlation = features[ordered_features].corr().abs()
    kept: list[str] = []
    dropped: list[str] = []

    for column in ordered_features:
        if any(correlation.loc[column, kept_column] >= threshold for kept_column in kept):
            dropped.append(column)
        else:
            kept.append(column)

    kept_set = set(kept)
    kept_in_original_order = [column for column in features.columns if column in kept_set]
    return features[kept_in_original_order].copy(), dropped


def build_submission_frame(test_ids: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    submission = pd.DataFrame({"ID": test_ids.to_numpy()})
    for column in COMPETITION_TARGETS:
        submission[column] = np.clip(predictions[column].to_numpy(), 0.0, 1.0)
    return submission
