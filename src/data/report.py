"""Data quality report generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mlops.drift.data_quality import summarize_data_quality
from src.data.validate import DataValidationReport
from src.utils.io import save_json


def generate_data_quality_report(
    dataframe: pd.DataFrame,
    validation_report: DataValidationReport,
    dataset_version: str,
    output_path: Path,
) -> dict[str, Any]:
    """Create and persist a JSON data quality report."""
    quality_summary = summarize_data_quality(dataframe)
    report = {
        "dataset_version": dataset_version,
        "validation": validation_report.to_dict(),
        "quality_summary": quality_summary,
    }
    save_json(report, output_path)
    return report