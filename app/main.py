"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app.api.v1.health_routes import router as health_router
from app.api.v1.predict_routes import router as predict_router
from app.core.config import ensure_runtime_directories
from app.core.exceptions import PlatformError
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.monitoring.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY

settings = get_settings()
configure_logging(settings.log_level)
ensure_runtime_directories()
LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Pipeline Platform",
    version="1.0.0",
    description="Production-oriented inference API for customer churn prediction.",
)
app.include_router(health_router)
app.include_router(predict_router, prefix="/api/v1")


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Capture request metrics and structured request IDs."""
    request_id = request.headers.get("X-Request-Id", str(uuid4()))
    start_time = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-Id"] = request_id
        return response
    except PlatformError as exc:
        status_code = exc.status_code
        ERROR_COUNT.labels(path=request.url.path).inc()
        LOGGER.warning(
            "Handled platform error",
            extra={"event": exc.error_code, "request_id": request_id},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "request_id": request_id,
            },
        )
    except HTTPException as exc:
        status_code = exc.status_code
        raise exc
    except Exception as exc:
        ERROR_COUNT.labels(path=request.url.path).inc()
        LOGGER.exception(
            "Unhandled API exception",
            extra={"event": "api_error", "request_id": request_id},
        )
        return JSONResponse(status_code=500, content={"detail": str(exc), "request_id": request_id})
    finally:
        duration = perf_counter() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(duration)


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint with a basic service banner."""
    return {"service": settings.app_name, "status": "running"}