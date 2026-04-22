from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


@dataclass(frozen=True)
class ImportanceWeightingResult:
    sample_weights: pd.Series
    test_probability: pd.Series
    density_ratio: pd.Series
    feature_columns: list[str]
    prior_test_probability: float


def compute_soft_test_proximity_weights(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    min_weight: float = 1.0,
    max_weight: float = 1.5,
    temperature: float = 2.0,
    probability_clip: float = 1e-4,
    classifier: GradientBoostingClassifier | None = None,
) -> ImportanceWeightingResult:
    """Estimate a very mild, bounded importance weighting from a train/test classifier."""
    if min_weight <= 0.0:
        raise ValueError("min_weight must be > 0.")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight.")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must be between 0 and 0.5.")

    train_frame, test_frame, used_columns = _prepare_domain_classifier_frames(
        x_train,
        x_test,
        feature_columns=feature_columns,
    )
    if train_frame.empty:
        raise ValueError("No usable feature columns were found for importance weighting.")

    X_domain = pd.concat([train_frame, test_frame], axis=0, ignore_index=True)
    y_domain = np.concatenate(
        [
            np.zeros(len(train_frame), dtype=np.int8),
            np.ones(len(test_frame), dtype=np.int8),
        ]
    )
    prior_test_probability = len(test_frame) / len(X_domain)

    domain_classifier = classifier
    if domain_classifier is None:
        domain_classifier = GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=120,
            min_samples_leaf=max(32, min(len(train_frame) // 200, 256)),
            subsample=0.9,
            random_state=42,
        )

    domain_classifier.fit(X_domain, y_domain)
    test_probability = np.clip(
        domain_classifier.predict_proba(train_frame)[:, 1],
        probability_clip,
        1.0 - probability_clip,
    )

    prior_logit = np.log(prior_test_probability / (1.0 - prior_test_probability))
    sample_logit = np.log(test_probability / (1.0 - test_probability))
    log_density_ratio = sample_logit - prior_logit
    density_ratio = np.exp(np.clip(log_density_ratio, -20.0, 20.0))

    positive_shift = np.maximum(log_density_ratio, 0.0)
    closeness = 1.0 - np.exp(-positive_shift / temperature)
    sample_weights = min_weight + (max_weight - min_weight) * closeness

    return ImportanceWeightingResult(
        sample_weights=pd.Series(sample_weights, index=x_train.index, name="sample_weight"),
        test_probability=pd.Series(test_probability, index=x_train.index, name="p_test_given_x"),
        density_ratio=pd.Series(density_ratio, index=x_train.index, name="density_ratio"),
        feature_columns=used_columns,
        prior_test_probability=float(prior_test_probability),
    )


def _prepare_domain_classifier_frames(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if feature_columns is None:
        shared_columns = [column for column in x_train.columns if column in x_test.columns and column != "ID"]
    else:
        missing = [column for column in feature_columns if column not in x_train.columns or column not in x_test.columns]
        if missing:
            raise ValueError(f"feature_columns contains unknown columns: {missing}")
        shared_columns = [column for column in feature_columns if column != "ID"]

    numeric_columns: list[str] = []
    train_parts: list[pd.Series] = []
    test_parts: list[pd.Series] = []

    for column in shared_columns:
        train_numeric = pd.to_numeric(x_train[column], errors="coerce")
        test_numeric = pd.to_numeric(x_test[column], errors="coerce")

        if train_numeric.isna().any() or test_numeric.isna().any():
            continue

        numeric_columns.append(column)
        train_parts.append(train_numeric)
        test_parts.append(test_numeric)

    train_frame = pd.concat(train_parts, axis=1) if train_parts else pd.DataFrame(index=x_train.index)
    test_frame = pd.concat(test_parts, axis=1) if test_parts else pd.DataFrame(index=x_test.index)

    if numeric_columns:
        train_frame.columns = numeric_columns
        test_frame.columns = numeric_columns

    return train_frame.astype(np.float32), test_frame.astype(np.float32), numeric_columns
