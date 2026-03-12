"""Top-level training entrypoint for the customer churn model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from app.core.config import ensure_runtime_directories
from app.core.exceptions import DataValidationError
from app.core.logging import configure_logging
from app.core.settings import AppSettings, get_settings
from mlops.mlflow.tracking import configure_mlflow, start_run
from src.data.ingest import ingest_raw_data
from src.data.report import generate_data_quality_report
from src.data.split import DatasetSplit, split_dataset
from src.data.transform import transform_customer_churn_data
from src.data.validate import DataValidationReport, validate_dataframe
from src.data.versioning import DatasetVersion, build_dataset_version
from src.evaluation.evaluate import EvaluationResult, evaluate_pipeline
from src.evaluation.validation import validate_model_performance
from src.features.feature_engineering import FeatureBundle, build_feature_pipeline
from src.registry.model_registry import LocalModelRegistry, RegistryEntry
from src.training.trainer import ModelTrainer, TrainerConfig
from src.training.tune import tune_hyperparameters
from src.utils.io import save_json, save_pickle, write_csv
from src.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingRunOutput:
    """Return type for training jobs."""

    model_version: str
    model_path: Path
    registry_entry: RegistryEntry
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    data_validation_report: DataValidationReport
    dataset_version: DatasetVersion
    data_quality_report_path: Path


def _persist_datasets(splits: DatasetSplit, settings: AppSettings) -> None:
    write_csv(splits.train, settings.processed_data_dir / "train.csv")
    write_csv(splits.validation, settings.processed_data_dir / "validation.csv")
    write_csv(splits.test, settings.processed_data_dir / "test.csv")


def _persist_feature_manifest(feature_bundle: FeatureBundle, settings: AppSettings) -> None:
    save_json(
        {
            "feature_columns": feature_bundle.feature_columns,
            "numeric_columns": feature_bundle.numeric_columns,
            "categorical_columns": feature_bundle.categorical_columns,
        },
        settings.features_data_dir / "feature_manifest.json",
    )


def _log_evaluation(prefix: str, evaluation_result: EvaluationResult) -> None:
    mlflow.log_metrics(
        {f"{prefix}_{metric_name}": metric_value for metric_name, metric_value in evaluation_result.metrics.items()}
    )


def run_training_job(
    data_path: Path | None = None,
    tune_model: bool = True,
    settings: AppSettings | None = None,
) -> TrainingRunOutput:
    """Execute the end-to-end training workflow with MLflow tracking."""
    runtime_settings = settings or get_settings()
    configure_logging(runtime_settings.log_level)
    ensure_runtime_directories()
    set_global_seed(runtime_settings.random_seed)

    training_data_path = data_path or runtime_settings.data_path
    raw_dataframe = ingest_raw_data(training_data_path)
    validation_report = validate_dataframe(
        raw_dataframe,
        target_column="churn",
        max_missing_ratio=runtime_settings.max_missing_ratio,
    )
    if not validation_report.is_valid:
        raise DataValidationError(
            "Training dataset failed validation gates.",
            details={"errors": validation_report.errors},
        )

    transformed_dataframe = transform_customer_churn_data(raw_dataframe)
    dataset_version = build_dataset_version(
        transformed_dataframe,
        source_path=training_data_path,
        dataset_name=runtime_settings.dataset_name,
    )
    dataset_split = split_dataset(
        transformed_dataframe,
        target_column="churn",
        random_state=runtime_settings.random_seed,
    )
    _persist_datasets(dataset_split, runtime_settings)

    feature_bundle = build_feature_pipeline(transformed_dataframe, target_column="churn")
    _persist_feature_manifest(feature_bundle, runtime_settings)

    data_quality_report_path = (
        runtime_settings.reports_dir / f"data_quality_{dataset_version.dataset_version}.json"
    )
    data_quality_report = generate_data_quality_report(
        transformed_dataframe,
        validation_report=validation_report,
        dataset_version=dataset_version.dataset_version,
        output_path=data_quality_report_path,
    )

    trainer = ModelTrainer(TrainerConfig(random_state=runtime_settings.random_seed))
    training_pipeline = trainer.build_pipeline(feature_bundle)

    registry = LocalModelRegistry(runtime_settings.model_registry_dir)
    version_number = registry.get_next_version_number(runtime_settings.model_name)
    model_version = registry.build_version_label(runtime_settings.model_name, version_number)

    best_hyperparameters: dict[str, Any] = {}
    if tune_model:
        best_hyperparameters = tune_hyperparameters(
            training_pipeline,
            dataset_split.train,
            feature_bundle.feature_columns,
            random_state=runtime_settings.random_seed,
        )
        training_pipeline = trainer.build_pipeline(feature_bundle, best_hyperparameters)

    fitted_pipeline = trainer.fit(
        training_pipeline,
        dataset_split.train,
        feature_bundle.feature_columns,
    )

    validation_result = evaluate_pipeline(
        fitted_pipeline,
        dataset_split.validation,
        feature_bundle.feature_columns,
        dataset_name="validation",
        threshold=runtime_settings.prediction_threshold,
    )
    test_result = evaluate_pipeline(
        fitted_pipeline,
        dataset_split.test,
        feature_bundle.feature_columns,
        dataset_name="test",
        threshold=runtime_settings.prediction_threshold,
    )

    validation_issues = validate_model_performance(
        test_result.metrics,
        minimum_roc_auc=runtime_settings.minimum_roc_auc,
        minimum_f1_score=runtime_settings.minimum_f1_score,
    )

    model_path = runtime_settings.exported_models_dir / f"{runtime_settings.model_name}_{model_version}.pkl"
    model_package = {
        "pipeline": fitted_pipeline,
        "feature_columns": feature_bundle.feature_columns,
        "target_column": "churn",
        "model_version": model_version,
        "version_number": version_number,
        "dataset_version": dataset_version.to_dict(),
    }
    save_pickle(model_package, model_path)

    configure_mlflow(
        runtime_settings.mlflow_tracking_uri,
        runtime_settings.mlflow_experiment_name,
    )
    with start_run(run_name=f"train-{model_version}") as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(
            {
                "model_name": runtime_settings.model_name,
                "tune_model": tune_model,
                "random_seed": runtime_settings.random_seed,
                "train_rows": len(dataset_split.train),
                "validation_rows": len(dataset_split.validation),
                "test_rows": len(dataset_split.test),
                "dataset_version": dataset_version.dataset_version,
                "dataset_fingerprint": dataset_version.fingerprint,
                "model_version": model_version,
                "model_version_number": version_number,
                **best_hyperparameters,
            }
        )
        _log_evaluation("validation", validation_result)
        _log_evaluation("test", test_result)
        mlflow.log_dict(dataset_version.to_dict(), "artifacts/dataset_version.json")
        mlflow.log_dict(validation_report.to_dict(), "artifacts/data_validation_report.json")
        mlflow.log_dict(data_quality_report, "artifacts/data_quality_report.json")
        mlflow.log_artifact(str(model_path), artifact_path="exported_model")
        mlflow.sklearn.log_model(fitted_pipeline, artifact_path="model")

        try:
            infer_signature(
                dataset_split.train[feature_bundle.feature_columns],
                fitted_pipeline.predict_proba(dataset_split.train[feature_bundle.feature_columns])[:, 1],
            )
        except Exception:
            LOGGER.warning(
                "Failed to infer MLflow signature",
                extra={"event": "mlflow_signature_warning", "run_id": run_id},
            )

    registry_entry = registry.register_model(
        model_name=runtime_settings.model_name,
        version=model_version,
        stage=runtime_settings.default_model_stage,
        artifact_path=model_path,
        run_id=run_id,
        metrics={k: float(v) for k, v in test_result.metrics.items() if k != "threshold"},
        validation_issues=validation_issues,
        version_number=version_number,
        dataset_version=dataset_version.dataset_version,
        dataset_fingerprint=dataset_version.fingerprint,
    )

    LOGGER.info(
        "Training job completed",
        extra={
            "event": "training_completed",
            "model_version": model_version,
            "run_id": registry_entry.run_id,
            "stage": registry_entry.stage,
        },
    )

    return TrainingRunOutput(
        model_version=model_version,
        model_path=model_path,
        registry_entry=registry_entry,
        validation_metrics=validation_result.metrics,
        test_metrics=test_result.metrics,
        data_validation_report=validation_report,
        dataset_version=dataset_version,
        data_quality_report_path=data_quality_report_path,
    )