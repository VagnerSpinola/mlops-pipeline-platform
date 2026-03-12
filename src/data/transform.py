"""Business-aware transformations for churn modeling."""

from __future__ import annotations

import pandas as pd

YES_NO_COLUMNS = [
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
    "churn",
]

SERVICE_COLUMNS = [
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


def transform_customer_churn_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw data into a consistent training-ready tabular format."""
    transformed = dataframe.copy()
    transformed.columns = [column.strip().lower() for column in transformed.columns]

    if "customer_id" in transformed.columns:
        transformed["customer_id"] = transformed["customer_id"].astype(str).str.strip()
        transformed = transformed.drop_duplicates(subset=["customer_id"], keep="last")

    transformed["gender"] = transformed["gender"].astype(str).str.title().str.strip()
    transformed["payment_method"] = (
        transformed["payment_method"].astype(str).str.title().str.strip()
    )
    transformed["contract"] = transformed["contract"].astype(str).str.title().str.strip()
    transformed["internet_service"] = (
        transformed["internet_service"].astype(str).str.title().str.strip()
    )

    transformed["senior_citizen"] = transformed["senior_citizen"].astype(int).astype(str)

    for column in YES_NO_COLUMNS:
        transformed[column] = transformed[column].astype(str).str.title().str.strip()

    for column in SERVICE_COLUMNS:
        transformed[column] = transformed[column].astype(str).str.title().str.strip()
        transformed[column] = transformed[column].replace({"No Internet Service": "NoService"})
        transformed[column] = transformed[column].replace({"No Phone Service": "NoService"})

    transformed["tenure"] = pd.to_numeric(transformed["tenure"], errors="coerce").fillna(0).astype(int)
    transformed["monthly_charges"] = pd.to_numeric(
        transformed["monthly_charges"], errors="coerce"
    ).fillna(transformed["monthly_charges"].median())
    transformed["total_charges"] = pd.to_numeric(
        transformed["total_charges"], errors="coerce"
    )
    transformed["total_charges"] = transformed["total_charges"].fillna(
        transformed["monthly_charges"] * transformed["tenure"].clip(lower=1)
    )

    return transformed