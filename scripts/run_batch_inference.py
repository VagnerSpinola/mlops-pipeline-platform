"""CLI wrapper for batch inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.batch_inference_pipeline import execute_batch_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference over a CSV file.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-alias", type=str, default="champion", choices=["champion", "challenger"])
    args = parser.parse_args()

    output = execute_batch_inference(args.input_path, args.output_path, model_alias=args.model_alias)
    print(
        {
            "predictions_path": str(output.predictions_path),
            "manifest_path": str(output.manifest_path),
            "runs_table_path": str(output.runs_table_path),
            "model_version": output.model_version,
            "model_alias": output.model_alias,
        }
    )


if __name__ == "__main__":
    main()