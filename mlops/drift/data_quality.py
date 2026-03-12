"""Data quality checks used ahead of retraining decisions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_data_quality(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Produce a lightweight profile for quality monitoring."""
    categorical_summary = {}
    categorical_columns = dataframe.select_dtypes(exclude="number").columns.tolist()
    for column in categorical_columns:
        categorical_summary[column] = dataframe[column].astype(str).value_counts().head(5).to_dict()

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "missing_ratio_by_column": dataframe.isna().mean().round(4).to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numerical_summary": dataframe.select_dtypes(include="number").describe().round(4).to_dict(),
        "categorical_summary": categorical_summary,
    }