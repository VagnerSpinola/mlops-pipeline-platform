"""Inference-time preprocessing aligned with the training pipeline."""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import InferenceError
from src.data.transform import transform_customer_churn_data


def preprocess_inference_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Normalize request payloads and align them with the trained feature contract."""
    prepared = dataframe.copy()
    synthetic_target = False
    if "churn" not in prepared.columns:
        prepared["churn"] = "No"
        synthetic_target = True

    transformed = transform_customer_churn_data(prepared)
    if synthetic_target:
        transformed = transformed.drop(columns=["churn"])

    missing_features = [column for column in feature_columns if column not in transformed.columns]
    if missing_features:
        raise InferenceError(
            "Inference payload is missing required features.",
            details={"missing_features": missing_features},
        )

    return transformed[feature_columns]