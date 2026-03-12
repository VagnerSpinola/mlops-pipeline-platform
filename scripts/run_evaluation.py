"""Evaluate the latest registered model on the saved test split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.settings import get_settings
from app.inference.predictor import ModelPredictor
from src.evaluation.metrics import compute_classification_metrics
from src.features.feature_engineering import encode_target


def main() -> None:
    settings = get_settings()
    predictor = ModelPredictor()
    test_dataframe = pd.read_csv(settings.processed_data_dir / "test.csv")
    prediction_result = predictor.predict_dataframe(test_dataframe)
    probabilities = prediction_result["probabilities"]
    y_true = encode_target(test_dataframe["churn"]).to_numpy()
    metrics = compute_classification_metrics(y_true, probabilities, settings.prediction_threshold)
    print({
        "model_version": prediction_result["model_version"],
        "model_alias": prediction_result["model_alias"],
        "dataset_version": prediction_result["dataset_version"],
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()