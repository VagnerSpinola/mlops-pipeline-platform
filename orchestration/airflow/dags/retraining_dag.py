"""Airflow DAG for periodic retraining checks."""

from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover
    DAG = None
    PythonOperator = None

from app.core.settings import get_settings
from pipelines.retraining_pipeline import execute_retraining_pipeline


def retrain_if_needed() -> None:
    """Airflow task callable for retraining based on current raw data."""
    settings = get_settings()
    execute_retraining_pipeline(
        reference_data_path=settings.processed_data_dir / "train.csv",
        current_data_path=settings.data_path,
        force_retrain=False,
    )


if DAG is not None and PythonOperator is not None:
    with DAG(
        dag_id="customer_churn_retraining",
        start_date=datetime(2026, 1, 1),
        schedule="0 3 * * 6",
        catchup=False,
        tags=["mlops", "retraining"],
    ) as dag:
        PythonOperator(task_id="retrain_if_needed", python_callable=retrain_if_needed)
else:  # pragma: no cover
    dag = None