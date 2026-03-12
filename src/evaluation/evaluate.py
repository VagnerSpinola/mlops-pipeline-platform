"""Reusable evaluation entrypoint used by training and batch scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import compute_classification_metrics
from src.features.feature_engineering import encode_target


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation output containing metrics and dataset metadata."""

    dataset_name: str
    metrics: dict[str, Any]
    record_count: int


def evaluate_pipeline(
    model_pipeline: Pipeline,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "churn",
    threshold: float = 0.50,
    dataset_name: str = "validation",
) -> EvaluationResult:
    """Evaluate a fitted pipeline on a named dataset split."""
    y_true = encode_target(dataframe[target_column]).to_numpy()
    y_scores = model_pipeline.predict_proba(dataframe[feature_columns])[:, 1]
    metrics = compute_classification_metrics(y_true, y_scores, threshold=threshold)
    return EvaluationResult(
        dataset_name=dataset_name,
        metrics=metrics,
        record_count=len(dataframe),
    )