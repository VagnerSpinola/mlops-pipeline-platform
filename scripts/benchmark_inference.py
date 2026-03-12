"""Benchmark inference latency for champion or challenger models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

import pandas as pd

from app.core.settings import get_settings
from app.inference.predictor import ModelPredictor
from src.utils.io import save_json


def percentile(sorted_values: list[float], q: float) -> float:
    """Return a simple percentile from pre-sorted values."""
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[index]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Benchmark inference latency.")
    parser.add_argument("--input-path", type=Path, default=settings.data_path)
    parser.add_argument("--records", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=settings.benchmark_iterations)
    parser.add_argument("--warmup", type=int, default=settings.benchmark_warmup_iterations)
    parser.add_argument("--model-alias", choices=["champion", "challenger"], default="champion")
    args = parser.parse_args()

    predictor = ModelPredictor(settings=settings)
    dataframe = pd.read_csv(args.input_path).drop(columns=["churn"], errors="ignore").head(args.records)

    for _ in range(args.warmup):
        predictor.predict_dataframe(dataframe, model_alias=args.model_alias)

    measurements_ms: list[float] = []
    for _ in range(args.iterations):
        start_time = perf_counter()
        predictor.predict_dataframe(dataframe, model_alias=args.model_alias)
        measurements_ms.append((perf_counter() - start_time) * 1000)

    measurements_ms.sort()
    benchmark_report = {
        "model_alias": args.model_alias,
        "records": args.records,
        "iterations": args.iterations,
        "mean_latency_ms": round(mean(measurements_ms), 3),
        "p50_latency_ms": round(percentile(measurements_ms, 0.50), 3),
        "p95_latency_ms": round(percentile(measurements_ms, 0.95), 3),
        "p99_latency_ms": round(percentile(measurements_ms, 0.99), 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = settings.benchmark_dir / f"inference_benchmark_{args.model_alias}.json"
    save_json(benchmark_report, output_path)
    print({**benchmark_report, "output_path": str(output_path)})


if __name__ == "__main__":
    main()