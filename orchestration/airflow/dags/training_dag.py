"""Airflow DAG for scheduled model training."""

from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover
    DAG = None
    PythonOperator = None

from pipelines.training_pipeline import execute_training_pipeline


def train_model() -> None:
    """Airflow task callable for scheduled training."""
    execute_training_pipeline(tune_model=True)


if DAG is not None and PythonOperator is not None:
    with DAG(
        dag_id="customer_churn_training",
        start_date=datetime(2026, 1, 1),
        schedule="0 2 * * 1",
        catchup=False,
        tags=["mlops", "training"],
    ) as dag:
        PythonOperator(task_id="train_model", python_callable=train_model)
else:  # pragma: no cover
    dag = None