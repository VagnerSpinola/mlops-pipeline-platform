"""I/O helpers for structured artifacts and model files."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file from disk."""
    return pd.read_csv(path)


def write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def append_csv_row(row: dict[str, Any], path: Path) -> Path:
    """Append a single-row record to a CSV file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([row])
    if path.exists():
        dataframe.to_csv(path, mode="a", header=False, index=False)
    else:
        dataframe.to_csv(path, index=False)
    return path


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Serialize a JSON-compatible dictionary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk if the file exists."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(payload: Any, path: Path) -> Path:
    """Serialize a Python object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)
    return path


def load_pickle(path: Path) -> Any:
    """Deserialize a pickled object from disk."""
    with path.open("rb") as file_handle:
        return pickle.load(file_handle)