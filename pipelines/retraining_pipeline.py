"""Retraining workflow that can react to drift signals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops.drift.data_quality import summarize_data_quality
from mlops.drift.drift_monitor import DriftReport, detect_drift
from mlops.drift.retraining_policy import evaluate_retraining_need
from pipelines.training_pipeline import execute_training_pipeline
from src.data.versioning import build_dataset_version
from src.utils.io import save_json
from app.core.settings import get_settings


def execute_retraining_pipeline(
    reference_data_path: Path,
    current_data_path: Path,
    force_retrain: bool = False,
) -> tuple[DriftReport, dict[str, object] | None]:
    """Evaluate drift, then retrain when thresholds are crossed."""
    settings = get_settings()
    reference_dataframe = pd.read_csv(reference_data_path)
    current_dataframe = pd.read_csv(current_data_path)
    drift_report = detect_drift(reference_dataframe, current_dataframe)
    quality_summary = summarize_data_quality(current_dataframe)
    reference_dataset_version = build_dataset_version(
        reference_dataframe,
        source_path=reference_data_path,
        dataset_name=settings.dataset_name,
    )
    current_dataset_version = build_dataset_version(
        current_dataframe,
        source_path=current_data_path,
        dataset_name=settings.dataset_name,
    )
    retraining_decision = evaluate_retraining_need(
        drift_report=drift_report,
        reference_dataset_version=reference_dataset_version.dataset_version,
        current_dataset_version=current_dataset_version.dataset_version,
        quality_summary=quality_summary,
        force_retrain=force_retrain,
    )
    decision_path = settings.reports_dir / "latest_retraining_decision.json"
    save_json(retraining_decision.to_dict(), decision_path)

    if not retraining_decision.should_retrain:
        return drift_report, {
            "message": "No retraining triggered.",
            "data_quality": quality_summary,
            "dataset_version": current_dataset_version.to_dict(),
            "retraining_decision": retraining_decision.to_dict(),
        }

    output = execute_training_pipeline(data_path=current_data_path, tune_model=True)
    return drift_report, {
        "model_version": output.model_version,
        "serving_alias": output.registry_entry.serving_alias,
        "registry_stage": output.registry_entry.stage,
        "test_metrics": output.test_metrics,
        "dataset_version": output.dataset_version.to_dict(),
        "retraining_decision": retraining_decision.to_dict(),
    }