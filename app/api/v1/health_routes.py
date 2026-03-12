"""Health and operational endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.core.exceptions import ModelRegistryError
from app.inference.predictor import get_predictor
from app.monitoring.metrics import render_metrics, set_model_info

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, object]:
    """Return service and model readiness."""
    predictor = get_predictor()
    summary = predictor.get_registry_summary()
    champion = summary.champion
    if champion is None:
        raise ModelRegistryError("No champion model is currently available.")
    set_model_info(champion.version, champion.serving_alias, champion.dataset_version)
    return {
        "status": "ok",
        "model_version": champion.version,
        "model_alias": champion.serving_alias,
        "dataset_version": champion.dataset_version,
        "registry": summary.to_dict(),
    }


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus-compatible metrics."""
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)