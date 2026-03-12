"""Dataset validation for schema, null checks, and target sanity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.data.ingest import EXPECTED_COLUMNS


@dataclass(slots=True)
class DataValidationReport:
    """Structured validation outcome for ingestion artifacts."""

    is_valid: bool
    row_count: int
    column_count: int
    missing_columns: list[str]
    extra_columns: list[str]
    missing_ratio_by_column: dict[str, float]
    duplicate_customer_ids: int
    target_distribution: dict[str, int]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def validate_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "churn",
    max_missing_ratio: float = 0.15,
) -> DataValidationReport:
    """Validate structure and basic quality constraints for churn data."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    extra_columns = [column for column in dataframe.columns if column not in EXPECTED_COLUMNS]

    errors: list[str] = []
    warnings: list[str] = []

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if target_column not in dataframe.columns:
        errors.append(f"Target column '{target_column}' not found")

    missing_ratio_by_column = (
        dataframe.isna().mean().sort_values(ascending=False).round(4).to_dict()
        if not dataframe.empty
        else {}
    )
    for column, ratio in missing_ratio_by_column.items():
        if ratio > max_missing_ratio:
            warnings.append(
                f"Column '{column}' exceeds missing value threshold with ratio {ratio:.2%}"
            )

    duplicate_customer_ids = 0
    if "customer_id" in dataframe.columns:
        duplicate_customer_ids = int(dataframe["customer_id"].duplicated().sum())
        if duplicate_customer_ids > 0:
            warnings.append(f"Found {duplicate_customer_ids} duplicated customer_id values")

    target_distribution: dict[str, int] = {}
    if target_column in dataframe.columns:
        target_distribution = dataframe[target_column].astype(str).value_counts().to_dict()
        if len(target_distribution) < 2:
            errors.append("Target column must contain at least two classes")

    return DataValidationReport(
        is_valid=not errors,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        missing_ratio_by_column=missing_ratio_by_column,
        duplicate_customer_ids=duplicate_customer_ids,
        target_distribution=target_distribution,
        errors=errors,
        warnings=warnings,
    )