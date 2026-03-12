"""Batch inference workflow over CSV inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.settings import get_settings
from app.inference.predictor import ModelPredictor
from src.utils.io import append_csv_row, save_json, write_csv


@dataclass(slots=True)
class BatchInferenceOutput:
    """Batch inference output artifacts and metadata."""

    predictions_path: Path
    manifest_path: Path
    runs_table_path: Path
    model_version: str
    model_alias: str
    dataset_version: str
    row_count: int


def execute_batch_inference(input_path: Path, output_path: Path, model_alias: str = "champion") -> BatchInferenceOutput:
    """Run batch inference and persist predictions plus run metadata."""
    settings = get_settings()
    dataframe = pd.read_csv(input_path)
    predictor = ModelPredictor()
    result = predictor.predict_dataframe(dataframe, model_alias=model_alias)
    output_dataframe = dataframe.copy()
    output_dataframe["churn_probability"] = result["probabilities"]
    output_dataframe["predicted_churn"] = result["predictions"]
    write_csv(output_dataframe, output_path)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = settings.batch_output_dir / f"batch_inference_{run_timestamp}.json"
    runs_table_path = settings.batch_output_dir / "batch_inference_runs.csv"
    manifest = {
        "input_path": str(input_path),
        "predictions_path": str(output_path),
        "model_version": result["model_version"],
        "version_number": result["version_number"],
        "model_alias": result["model_alias"],
        "dataset_version": result["dataset_version"],
        "row_count": int(len(dataframe)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(manifest, manifest_path)
    append_csv_row(manifest, runs_table_path)

    return BatchInferenceOutput(
        predictions_path=output_path,
        manifest_path=manifest_path,
        runs_table_path=runs_table_path,
        model_version=result["model_version"],
        model_alias=result["model_alias"],
        dataset_version=result["dataset_version"],
        row_count=int(len(dataframe)),
    )