"""Model trainer for churn classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.feature_engineering import FeatureBundle, encode_target


@dataclass(slots=True)
class TrainerConfig:
    """Configuration applied to the estimator."""

    random_state: int = 42
    n_estimators: int = 250
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.90
    colsample_bytree: float = 0.85
    min_child_weight: int = 1
    reg_lambda: float = 1.0


class ModelTrainer:
    """Build and fit an end-to-end preprocessing plus model pipeline."""

    def __init__(self, trainer_config: TrainerConfig | None = None) -> None:
        self.trainer_config = trainer_config or TrainerConfig()

    def build_pipeline(
        self,
        feature_bundle: FeatureBundle,
        hyperparameters: dict[str, Any] | None = None,
    ) -> Pipeline:
        """Create a scikit-learn pipeline with preprocessing and XGBoost."""
        config = self.trainer_config
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.random_state,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            reg_lambda=config.reg_lambda,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", feature_bundle.preprocessor),
                ("classifier", classifier),
            ]
        )
        if hyperparameters:
            pipeline.set_params(**hyperparameters)
        return pipeline

    def fit(
        self,
        pipeline: Pipeline,
        train_dataframe: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "churn",
    ) -> Pipeline:
        """Fit the pipeline on the provided training data."""
        X_train = train_dataframe[feature_columns]
        y_train = encode_target(train_dataframe[target_column])
        pipeline.fit(X_train, y_train)
        return pipeline