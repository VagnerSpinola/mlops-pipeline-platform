"""Structured logging helpers shared by training and inference."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Render logs in a compact JSON format for machine readability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for attribute in (
            "event",
            "request_id",
            "rows",
            "columns",
            "path",
            "duration_ms",
            "model_version",
            "stage",
            "run_id",
        ):
            value = getattr(record, attribute, None)
            if value is not None:
                payload[attribute] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for both scripts and the API."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level.upper())
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())