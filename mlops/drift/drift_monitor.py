"""Simple drift monitoring placeholder using PSI-like comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DriftReport:
    """Container for feature drift signals."""

    drift_detected: bool
    feature_scores: dict[str, float]
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the drift report."""
        return {
            "drift_detected": self.drift_detected,
            "feature_scores": self.feature_scores,
            "threshold": self.threshold,
        }


def _population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bucket_count: int = 10,
) -> float:
    expected_clean = expected.dropna().astype(float)
    actual_clean = actual.dropna().astype(float)
    if expected_clean.empty or actual_clean.empty:
        return 0.0

    bins = np.linspace(expected_clean.min(), expected_clean.max(), bucket_count + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_distribution, _ = np.histogram(expected_clean, bins=bins)
    actual_distribution, _ = np.histogram(actual_clean, bins=bins)

    expected_ratio = np.clip(expected_distribution / max(len(expected_clean), 1), 1e-6, None)
    actual_ratio = np.clip(actual_distribution / max(len(actual_clean), 1), 1e-6, None)
    return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))


def detect_drift(
    reference_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    threshold: float = 0.20,
) -> DriftReport:
    """Assess numeric feature drift against a PSI threshold."""
    numeric_columns = reference_dataframe.select_dtypes(include="number").columns.tolist()
    feature_scores = {
        column: _population_stability_index(reference_dataframe[column], current_dataframe[column])
        for column in numeric_columns
        if column in current_dataframe.columns
    }
    drift_detected = any(score >= threshold for score in feature_scores.values())
    return DriftReport(
        drift_detected=drift_detected,
        feature_scores=feature_scores,
        threshold=threshold,
    )