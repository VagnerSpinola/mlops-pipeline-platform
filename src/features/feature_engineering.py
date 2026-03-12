"""Feature engineering for the churn prediction use case."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMNS = {"customer_id"}


@dataclass(slots=True)
class FeatureBundle:
    """Encapsulates preprocessor components and selected feature columns."""

    preprocessor: ColumnTransformer
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


def encode_target(target: pd.Series) -> pd.Series:
    """Convert Yes/No churn labels into binary integers."""
    mapping = {"No": 0, "Yes": 1}
    encoded = target.map(mapping)
    if encoded.isna().any():
        invalid_values = sorted(set(target[encoded.isna()].astype(str).tolist()))
        raise ValueError(f"Unexpected target labels encountered: {invalid_values}")
    return encoded.astype(int)


def build_feature_pipeline(
    dataframe: pd.DataFrame,
    target_column: str = "churn",
) -> FeatureBundle:
    """Build the preprocessing graph used by both training and inference."""
    feature_columns = [
        column for column in dataframe.columns if column not in ID_COLUMNS and column != target_column
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return FeatureBundle(
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )