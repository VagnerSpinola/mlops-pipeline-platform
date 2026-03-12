"""Tests for model training and local registration."""

from __future__ import annotations

from src.registry.model_registry import LocalModelRegistry
from src.training.train import run_training_job
from tests.conftest import build_test_settings


def test_training_job_produces_model_artifacts(tmp_path) -> None:
    settings = build_test_settings(tmp_path)

    output = run_training_job(settings=settings, tune_model=False)

    assert output.model_path.exists()
    assert output.registry_entry.stage == "production"
    assert output.registry_entry.serving_alias == "champion"
    assert output.registry_entry.version.endswith("0001")
    assert 0.0 <= output.test_metrics["roc_auc"] <= 1.0
    assert output.data_quality_report_path.exists()


def test_second_candidate_is_registered_as_challenger(tmp_path) -> None:
    settings = build_test_settings(tmp_path)

    first_output = run_training_job(settings=settings, tune_model=False)
    second_output = run_training_job(settings=settings, tune_model=False)
    registry = LocalModelRegistry(settings.model_registry_dir)
    summary = registry.get_registry_summary(settings.model_name)

    assert first_output.registry_entry.serving_alias == "champion"
    assert second_output.registry_entry.serving_alias == "challenger"
    assert summary.champion is not None
    assert summary.challenger is not None


def test_promoting_challenger_reassigns_aliases_cleanly(tmp_path) -> None:
    settings = build_test_settings(tmp_path)

    first_output = run_training_job(settings=settings, tune_model=False)
    second_output = run_training_job(settings=settings, tune_model=False)
    registry = LocalModelRegistry(settings.model_registry_dir)

    promoted_entry = registry.promote_model_version(
        settings.model_name,
        second_output.registry_entry.version,
    )
    summary = registry.get_registry_summary(settings.model_name)

    assert first_output.registry_entry.version != second_output.registry_entry.version
    assert promoted_entry.version == second_output.registry_entry.version
    assert summary.champion is not None
    assert summary.champion.version == second_output.registry_entry.version
    assert summary.challenger is None