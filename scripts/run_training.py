"""CLI entrypoint to run the training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.training_pipeline import execute_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the churn model training pipeline.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to the raw CSV dataset")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Disable hyperparameter tuning for faster local iterations",
    )
    args = parser.parse_args()

    output = execute_training_pipeline(data_path=args.data_path, tune_model=not args.skip_tuning)
    print({
        "model_version": output.model_version,
        "serving_alias": output.registry_entry.serving_alias,
        "stage": output.registry_entry.stage,
        "dataset_version": output.dataset_version.dataset_version,
        "test_metrics": output.test_metrics,
    })


if __name__ == "__main__":
    main()