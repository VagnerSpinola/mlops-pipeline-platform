"""Tests for the FastAPI inference service."""

from __future__ import annotations

from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.settings import AppSettings, get_settings
from app.inference.predictor import get_predictor
from src.training.train import run_training_job


def test_api_predict_and_health_endpoints(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLOPS_DATA_PATH", str(Path("data/raw/customer_churn.csv").resolve()))
    monkeypatch.setenv("MLOPS_PROCESSED_DATA_DIR", str((tmp_path / "processed").resolve()))
    monkeypatch.setenv("MLOPS_FEATURES_DATA_DIR", str((tmp_path / "features").resolve()))
    monkeypatch.setenv("MLOPS_REPORTS_DIR", str((tmp_path / "reports").resolve()))
    monkeypatch.setenv("MLOPS_BATCH_OUTPUT_DIR", str((tmp_path / "batch_inference").resolve()))
    monkeypatch.setenv("MLOPS_BENCHMARK_DIR", str((tmp_path / "benchmarks").resolve()))
    monkeypatch.setenv("MLOPS_MODEL_REGISTRY_DIR", str((tmp_path / "registry").resolve()))
    monkeypatch.setenv("MLOPS_EXPORTED_MODELS_DIR", str((tmp_path / "exported").resolve()))
    monkeypatch.setenv("MLOPS_MLFLOW_TRACKING_URI", f"file:///{(tmp_path / 'mlruns').as_posix()}")
    monkeypatch.setenv("MLOPS_MINIMUM_ROC_AUC", "0.0")
    monkeypatch.setenv("MLOPS_MINIMUM_F1_SCORE", "0.0")
    monkeypatch.setenv("MLOPS_DEFAULT_MODEL_STAGE", "production")

    get_settings.cache_clear()
    get_predictor.cache_clear()
    settings = AppSettings()
    run_training_job(settings=settings, tune_model=False)

    import app.main as app_main

    reload(app_main)
    get_predictor.cache_clear()
    client = TestClient(app_main.app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["model_alias"] == "champion"

    payload = {
        "records": [
            {
                "customer_id": "CUST-9001",
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 92.4,
                "total_charges": 1108.8
            }
        ]
    }
    predict_response = client.post("/api/v1/predict", json=payload)
    assert predict_response.status_code == 200
    response_payload = predict_response.json()
    assert response_payload["model_alias"] == "champion"
    assert response_payload["version_number"] == 1
    assert response_payload["predictions"][0]["predicted_churn"] in {"Yes", "No"}


def test_yaml_config_support(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "platform.yaml"
    config_path.write_text(
        "app_name: yaml-configured-platform\napi_port: 8123\ndataset_name: yaml-dataset\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLOPS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("MLOPS_API_PORT", raising=False)

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.app_name == "yaml-configured-platform"
    assert settings.api_port == 8123
    assert settings.dataset_name == "yaml-dataset"