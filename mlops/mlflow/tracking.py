"""MLflow experiment tracking helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mlflow


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure the MLflow backend and active experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str) -> Iterator[mlflow.ActiveRun]:
    """Start and yield an MLflow run context."""
    with mlflow.start_run(run_name=run_name) as active_run:
        yield active_run