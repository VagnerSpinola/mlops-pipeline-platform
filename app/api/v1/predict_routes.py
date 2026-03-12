"""Prediction routes and request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, Query
import pandas as pd

from app.core.exceptions import PlatformError
from app.inference.predictor import get_predictor
from app.monitoring.metrics import PREDICTION_COUNT, set_model_info

router = APIRouter(tags=["predictions"])


class CustomerFeatures(BaseModel):
    """Request schema for a single churn scoring record."""

    customer_id: str = Field(..., examples=["CUST-001"])
    gender: str
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: str
    dependents: str
    tenure: int = Field(..., ge=0)
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)


class PredictionRequest(BaseModel):
    """Batch request wrapper."""

    records: list[CustomerFeatures] = Field(..., min_length=1)


class PredictionRecord(BaseModel):
    """Prediction output for a single record."""

    customer_id: str
    churn_probability: float
    predicted_churn: str


class PredictionResponse(BaseModel):
    """Inference response payload."""

    model_version: str
    version_number: int
    model_alias: str
    dataset_version: str
    predictions: list[PredictionRecord]


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    model_alias: str = Query(default="champion", pattern="^(champion|challenger)$"),
) -> PredictionResponse:
    """Generate churn probabilities for one or more customers."""
    predictor = get_predictor()
    dataframe = pd.DataFrame([record.model_dump() for record in payload.records])
    try:
        result = predictor.predict_dataframe(dataframe, model_alias=model_alias)
    except PlatformError:
        raise

    set_model_info(result["model_version"], result["model_alias"], result["dataset_version"])
    PREDICTION_COUNT.inc(len(payload.records))
    predictions = [
        PredictionRecord(
            customer_id=payload.records[index].customer_id,
            churn_probability=round(probability, 4),
            predicted_churn=prediction,
        )
        for index, (probability, prediction) in enumerate(
            zip(result["probabilities"], result["predictions"], strict=True)
        )
    ]
    return PredictionResponse(
        model_version=result["model_version"],
        version_number=result["version_number"],
        model_alias=result["model_alias"],
        dataset_version=result["dataset_version"],
        predictions=predictions,
    )