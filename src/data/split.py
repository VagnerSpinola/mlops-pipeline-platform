"""Train, validation, and test dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class DatasetSplit:
    """Container for split datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    dataframe: pd.DataFrame,
    target_column: str = "churn",
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplit:
    """Perform a stratified train/validation/test split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    train_validation, test = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe[target_column],
        random_state=random_state,
    )

    relative_validation_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        stratify=train_validation[target_column],
        random_state=random_state,
    )

    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )