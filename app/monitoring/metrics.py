"""Prometheus metrics exposed by the inference API."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Info, generate_latest

REQUEST_COUNT = Counter(
    "mlops_api_requests_total",
    "Total number of API requests.",
    labelnames=("method", "path", "status_code"),
)
REQUEST_LATENCY = Histogram(
    "mlops_api_request_latency_seconds",
    "Latency of API requests in seconds.",
    labelnames=("method", "path"),
)
PREDICTION_COUNT = Counter(
    "mlops_predictions_total",
    "Total number of generated predictions.",
)
ERROR_COUNT = Counter(
    "mlops_api_errors_total",
    "Total number of failed API requests.",
    labelnames=("path",),
)
MODEL_INFO = Info(
    "mlops_model",
    "Current model metadata exposed by the inference service.",
)


def set_model_info(model_version: str, model_alias: str, dataset_version: str) -> None:
    """Expose the active model version to Prometheus."""
    MODEL_INFO.info({"version": model_version, "alias": model_alias, "dataset_version": dataset_version})


def render_metrics() -> tuple[bytes, str]:
    """Serialize Prometheus metrics for HTTP responses."""
    return generate_latest(), CONTENT_TYPE_LATEST