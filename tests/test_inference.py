"""Tests for inference-time model loading and scoring."""

from __future__ import annotations

import pandas as pd

from app.inference.predictor import ModelPredictor
from src.training.train import run_training_job
from tests.conftest import build_test_settings


def test_model_predictor_scores_records(tmp_path) -> None:
    settings = build_test_settings(tmp_path)
    run_training_job(settings=settings, tune_model=False)

    predictor = ModelPredictor(settings=settings)
    dataframe = pd.read_csv("data/raw/customer_churn.csv").head(3).drop(columns=["churn"])
    result = predictor.predict_dataframe(dataframe)

    assert result["model_alias"] == "champion"
    assert result["model_version"].endswith("0001")
    assert len(result["predictions"]) == 3
    assert len(result["probabilities"]) == 3