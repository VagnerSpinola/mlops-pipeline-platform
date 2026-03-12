"""Raw data ingestion for the churn prediction workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn",
]


def ingest_raw_data(source_path: Path) -> pd.DataFrame:
    """Load raw CSV data into a DataFrame."""
    if not source_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {source_path}")

    dataframe = pd.read_csv(source_path)
    LOGGER.info(
        "Ingested raw dataset",
        extra={
            "event": "data_ingested",
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "path": str(source_path),
        },
    )
    return dataframe