"""Shared pytest helpers for repository-wide consistency."""

from __future__ import annotations

from pathlib import Path

from app.core.settings import AppSettings


def build_test_settings(tmp_path, **overrides) -> AppSettings:
    """Create isolated test settings with repository-consistent defaults."""
    defaults = {
        "data_path": Path("data/raw/customer_churn.csv"),
        "processed_data_dir": tmp_path / "processed",
        "features_data_dir": tmp_path / "features",
        "reports_dir": tmp_path / "reports",
        "batch_output_dir": tmp_path / "batch_inference",
        "benchmark_dir": tmp_path / "benchmarks",
        "model_registry_dir": tmp_path / "registry",
        "exported_models_dir": tmp_path / "exported",
        "mlflow_tracking_uri": f"file:///{(tmp_path / 'mlruns').as_posix()}",
        "minimum_roc_auc": 0.0,
        "minimum_f1_score": 0.0,
        "default_model_stage": "production",
    }
    defaults.update(overrides)
    return AppSettings(**defaults)