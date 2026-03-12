"""Manual model registration helper."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.settings import get_settings
from src.registry.model_registry import LocalModelRegistry
from src.utils.io import load_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an exported model artifact manually.")
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--stage", type=str, default="staging")
    args = parser.parse_args()

    settings = get_settings()
    registry = LocalModelRegistry(settings.model_registry_dir)
    version_number = registry.get_next_version_number(settings.model_name)
    version = registry.build_version_label(settings.model_name, version_number)
    model_package = load_pickle(args.artifact_path)
    dataset_version = model_package.get("dataset_version", {})
    entry = registry.register_model(
        model_name=settings.model_name,
        version=version,
        stage=args.stage,
        artifact_path=args.artifact_path,
        run_id="manual-registration",
        metrics={},
        validation_issues=[],
        version_number=version_number,
        dataset_version=dataset_version.get("dataset_version", "manual-dataset-version"),
        dataset_fingerprint=dataset_version.get("fingerprint", "manual-dataset-fingerprint"),
    )
    print(entry)


if __name__ == "__main__":
    main()