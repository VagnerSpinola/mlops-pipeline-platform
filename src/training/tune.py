"""Hyperparameter tuning utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import encode_target


def tune_hyperparameters(
    base_pipeline: Pipeline,
    train_dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "churn",
    random_state: int = 42,
    n_iter: int = 10,
) -> dict[str, Any]:
    """Run a bounded random search over XGBoost hyperparameters."""
    X_train = train_dataframe[feature_columns]
    y_train = encode_target(train_dataframe[target_column])
    smallest_class = int(y_train.value_counts().min())
    cv_folds = max(2, min(3, smallest_class))

    param_distributions = {
        "classifier__n_estimators": [100, 150, 200, 250, 300],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.03, 0.05, 0.08, 0.10],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 2, 4, 6],
        "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return dict(search.best_params_)