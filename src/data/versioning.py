"""Dataset version placeholder support for lineage and reproducibility."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DatasetVersion:
    """Metadata describing a versioned dataset snapshot."""

    dataset_name: str
    dataset_version: str
    fingerprint: str
    source_path: str
    row_count: int
    column_count: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset version metadata."""
        return asdict(self)


def build_dataset_version(
    dataframe: pd.DataFrame,
    source_path: Path,
    dataset_name: str,
) -> DatasetVersion:
    """Create a lightweight dataset version using a content fingerprint."""
    payload = pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    fingerprint = hashlib.sha256(payload).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DatasetVersion(
        dataset_name=dataset_name,
        dataset_version=f"{dataset_name}-{timestamp}-{fingerprint}",
        fingerprint=fingerprint,
        source_path=str(source_path),
        row_count=int(len(dataframe)),
        column_count=int(len(dataframe.columns)),
        created_at=datetime.now(timezone.utc).isoformat(),
    )