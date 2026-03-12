"""Prediction postprocessing helpers."""

from __future__ import annotations

import numpy as np


def postprocess_predictions(probabilities: np.ndarray, threshold: float = 0.50) -> list[str]:
    """Convert churn probabilities into business-facing labels."""
    return ["Yes" if score >= threshold else "No" for score in probabilities]