"""Custom platform exception hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlatformError(Exception):
    """Base error type for operationally meaningful failures."""

    message: str
    error_code: str = "platform_error"
    status_code: int = 500
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class ConfigurationError(PlatformError):
    """Raised when runtime configuration is invalid."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, error_code="configuration_error", status_code=500, details=details)


class DataValidationError(PlatformError):
    """Raised when input data fails validation gates."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, error_code="data_validation_error", status_code=422, details=details)


class ModelRegistryError(PlatformError):
    """Raised when registry state or model resolution fails."""

    def __init__(self, message: str, status_code: int = 503, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, error_code="model_registry_error", status_code=status_code, details=details)


class InferenceError(PlatformError):
    """Raised for inference-time contract violations or scoring failures."""

    def __init__(self, message: str, status_code: int = 422, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, error_code="inference_error", status_code=status_code, details=details)


class PipelineExecutionError(PlatformError):
    """Raised when orchestration or pipeline execution fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, error_code="pipeline_execution_error", status_code=500, details=details)