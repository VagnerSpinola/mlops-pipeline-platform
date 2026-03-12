"""Tests for data ingestion, validation, transformation, and splitting."""

from __future__ import annotations

from pathlib import Path

from src.data.ingest import ingest_raw_data
from src.data.split import split_dataset
from src.data.transform import transform_customer_churn_data
from src.data.validate import validate_dataframe


def test_data_pipeline_end_to_end() -> None:
    data_path = Path("data/raw/customer_churn.csv")
    raw_dataframe = ingest_raw_data(data_path)
    report = validate_dataframe(raw_dataframe)
    assert report.is_valid

    transformed = transform_customer_churn_data(raw_dataframe)
    assert transformed["total_charges"].isna().sum() == 0

    split = split_dataset(transformed, target_column="churn")
    assert len(split.train) + len(split.validation) + len(split.test) == len(transformed)
    assert set(split.train["churn"].unique()) == {"Yes", "No"}