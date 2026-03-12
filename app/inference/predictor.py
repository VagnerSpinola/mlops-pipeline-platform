"""Model loading and inference service."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import InferenceError, ModelRegistryError
from app.core.logging import configure_logging
from app.core.settings import AppSettings, get_settings
from app.inference.postprocess import postprocess_predictions
from app.inference.preprocess import preprocess_inference_dataframe
from src.registry.model_registry import LocalModelRegistry, RegistryEntry, RegistrySummary
from src.utils.io import load_pickle

LOGGER = logging.getLogger(__name__)


class ModelPredictor:
    """Lazy model loader backed by the local registry abstraction."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)
        self.registry = LocalModelRegistry(self.settings.model_registry_dir)

    @lru_cache(maxsize=4)
    def _load_registered_model(self, model_alias: str = "champion") -> tuple[RegistryEntry, dict[str, Any]]:
        entry = self.registry.get_model_by_alias(
            model_name=self.settings.model_name,
            alias=model_alias,
        )
        model_package = load_pickle(Path(entry.artifact_path))
        LOGGER.info(
            "Loaded model for inference",
            extra={
                "event": "model_loaded",
                "model_version": entry.version,
                "stage": entry.stage,
                "run_id": entry.run_id,
            },
        )
        return entry, model_package

    def get_model_version(self, model_alias: str = "champion") -> str:
        """Return the active model version used for predictions."""
        entry, _ = self._load_registered_model(model_alias=model_alias)
        return entry.version

    def get_registry_summary(self) -> RegistrySummary:
        """Return champion and challenger metadata for health reporting."""
        return self.registry.get_registry_summary(self.settings.model_name)

    def predict_dataframe(self, dataframe: pd.DataFrame, model_alias: str = "champion") -> dict[str, Any]:
        """Score a tabular inference payload."""
        if model_alias not in {self.settings.champion_alias, self.settings.challenger_alias}:
            raise InferenceError(
                f"Unsupported model alias {model_alias!r}.",
                details={"model_alias": model_alias},
            )

        try:
            entry, model_package = self._load_registered_model(model_alias=model_alias)
        except ModelRegistryError:
            raise

        feature_columns = model_package["feature_columns"]
        pipeline = model_package["pipeline"]
        try:
            preprocessed = preprocess_inference_dataframe(dataframe, feature_columns)
            probabilities = pipeline.predict_proba(preprocessed)[:, 1]
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("Unexpected scoring failure.", status_code=500) from exc

        predictions = postprocess_predictions(
            probabilities,
            threshold=self.settings.prediction_threshold,
        )

        return {
            "model_version": entry.version,
            "version_number": entry.version_number,
            "model_alias": entry.serving_alias,
            "dataset_version": entry.dataset_version,
            "probabilities": probabilities.tolist(),
            "predictions": predictions,
        }


@lru_cache(maxsize=1)
def get_predictor() -> ModelPredictor:
    """Return a cached predictor instance for the API runtime."""
    return ModelPredictor()