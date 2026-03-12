"""Prefect flow demonstrating Python-native orchestration."""

from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from pipelines.training_pipeline import execute_training_pipeline


@task(name="run-training-pipeline")
def run_training_task(data_path: str | None = None, tune_model: bool = True) -> dict[str, object]:
    """Execute the training pipeline inside a Prefect task."""
    output = execute_training_pipeline(data_path=Path(data_path) if data_path else None, tune_model=tune_model)
    return {
        "model_version": output.model_version,
        "stage": output.registry_entry.stage,
        "test_metrics": output.test_metrics,
    }


@flow(name="customer-churn-training-flow")
def training_flow(data_path: str | None = None, tune_model: bool = True) -> dict[str, object]:
    """Example Prefect flow that orchestrates model training."""
    return run_training_task(data_path=data_path, tune_model=tune_model)


if __name__ == "__main__":
    training_flow()