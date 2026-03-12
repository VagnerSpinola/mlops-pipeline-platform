"""Environment-aware application settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import (
    BASE_DIR,
    BATCH_OUTPUT_DIR,
    BENCHMARK_DIR,
    CONFIG_DIR,
    FEATURES_DATA_DIR,
    MODEL_EXPORT_DIR,
    MODEL_REGISTRY_DIR,
    MLRUNS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RAW_DATA_DIR,
)
from app.core.exceptions import ConfigurationError


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load flat YAML settings if a config file exists."""
    if not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML configuration at {config_path}",
            details={"config_path": str(config_path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError(
            f"YAML configuration must be a mapping at {config_path}",
            details={"config_path": str(config_path)},
        )
    return dict(payload)


def _resolve_repo_path(path_value: Path) -> Path:
    """Resolve repository-relative paths into absolute paths."""
    return path_value if path_value.is_absolute() else (BASE_DIR / path_value).resolve()


class AppSettings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLOPS_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "mlops-pipeline-platform"
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    random_seed: int = 42
    config_path: Path = Field(default=CONFIG_DIR / "platform.yaml")

    data_path: Path = Field(default=RAW_DATA_DIR / "customer_churn.csv")
    processed_data_dir: Path = Field(default=PROCESSED_DATA_DIR)
    features_data_dir: Path = Field(default=FEATURES_DATA_DIR)
    reports_dir: Path = Field(default=REPORTS_DIR)
    batch_output_dir: Path = Field(default=BATCH_OUTPUT_DIR)
    benchmark_dir: Path = Field(default=BENCHMARK_DIR)
    model_registry_dir: Path = Field(default=MODEL_REGISTRY_DIR)
    exported_models_dir: Path = Field(default=MODEL_EXPORT_DIR)

    mlflow_tracking_uri: str = Field(default=f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow_experiment_name: str = "customer-churn-platform"
    model_name: str = "customer-churn-classifier"
    default_model_stage: str = "production"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    dataset_name: str = "customer-churn-dataset"
    benchmark_iterations: int = 100
    benchmark_warmup_iterations: int = 10

    max_missing_ratio: float = 0.15
    minimum_roc_auc: float = 0.70
    minimum_f1_score: float = 0.65
    prediction_threshold: float = 0.50

    prometheus_namespace: str = "mlops_platform"

    prometheus_multiproc_dir: Path = Field(default=BASE_DIR / ".prometheus")

    def model_post_init(self, __context: Any) -> None:
        """Normalize filesystem paths so runtime code always receives absolute paths."""
        path_fields = (
            "config_path",
            "data_path",
            "processed_data_dir",
            "features_data_dir",
            "reports_dir",
            "batch_output_dir",
            "benchmark_dir",
            "model_registry_dir",
            "exported_models_dir",
            "prometheus_multiproc_dir",
        )
        for field_name in path_fields:
            setattr(self, field_name, _resolve_repo_path(getattr(self, field_name)))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached settings instance."""
    config_path = Path(os.getenv("MLOPS_CONFIG_PATH", CONFIG_DIR / "platform.yaml"))
    yaml_values = _load_yaml_config(config_path)
    env_snapshot = AppSettings()
    env_overrides = {
        field_name: getattr(env_snapshot, field_name)
        for field_name in env_snapshot.model_fields_set
    }
    merged_values = {**yaml_values, **env_overrides, "config_path": config_path}
    return AppSettings(**merged_values)