"""Training pipeline wrapper used by scripts and orchestrators."""

from __future__ import annotations

from pathlib import Path

from src.training.train import TrainingRunOutput, run_training_job


def execute_training_pipeline(data_path: Path | None = None, tune_model: bool = True) -> TrainingRunOutput:
    """Run the platform training pipeline."""
    return run_training_job(data_path=data_path, tune_model=tune_model)