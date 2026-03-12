"""Centralized repository path configuration."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "app"
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
REPORTS_DIR = PROCESSED_DATA_DIR / "reports"
BATCH_OUTPUT_DIR = PROCESSED_DATA_DIR / "batch_inference"
BENCHMARK_DIR = PROCESSED_DATA_DIR / "benchmarks"
MODELS_DIR = BASE_DIR / "models"
MODEL_CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
MODEL_EXPORT_DIR = MODELS_DIR / "exported"
MODEL_REGISTRY_DIR = MODELS_DIR / "registry"
MLRUNS_DIR = BASE_DIR / "mlruns"


def ensure_runtime_directories() -> None:
    """Create directories used by local runs if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DATA_DIR,
        REPORTS_DIR,
        BATCH_OUTPUT_DIR,
        BENCHMARK_DIR,
        MODEL_CHECKPOINT_DIR,
        MODEL_EXPORT_DIR,
        MODEL_REGISTRY_DIR,
        MLRUNS_DIR,
        CONFIG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)