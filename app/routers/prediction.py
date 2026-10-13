"""HTTP endpoints."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database.models import Prediction
from app.database.session import get_db
from app.schemas.house import (
    HouseFeatures,
    ModelInfoResponse,
    PredictionListResponse,
    PredictionResponse,
    describe_features,
)
from app.services.prediction_service import PredictionService, get_prediction_service

logger = logging.getLogger(__name__)
router = APIRouter()

DbSession = Annotated[Session, Depends(get_db)]
Service = Annotated[PredictionService, Depends(get_prediction_service)]
AppSettings = Annotated[Settings, Depends(get_settings)]


@router.get("/", tags=["status"])
def root() -> dict[str, str]:
    return {
        "message": "House Price Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@router.get("/health", tags=["status"])
def health(service: Service) -> dict[str, object]:
    return {"status": "ok", "model_loaded": service.is_loaded}


@router.get("/health/database", tags=["status"])
def health_database(db: DbSession) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("Database health check failed: %s", exc.__class__.__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database is unavailable.")
    return {"status": "ok", "database": "reachable"}


@router.get("/model/features", response_model=ModelInfoResponse, tags=["model"])
def model_features(settings: AppSettings) -> ModelInfoResponse:
    """Feature names, limits and examples (the Streamlit form is built from this)."""
    return ModelInfoResponse(
        currency=settings.price_currency,
        price_unit_multiplier=settings.price_unit_multiplier,
        features=describe_features(),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["predictions"],
)
def predict(
    features: HouseFeatures, db: DbSession, service: Service, settings: AppSettings
) -> PredictionResponse:
    record = service.predict_and_save(db, features)
    return PredictionResponse.from_record(record, settings.price_currency)


@router.get("/predictions", response_model=PredictionListResponse, tags=["predictions"])
def list_predictions(
    db: DbSession,
    settings: AppSettings,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PredictionListResponse:
    total = db.scalar(select(func.count()).select_from(Prediction)) or 0
    rows = db.scalars(
        select(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset)
    ).all()
    return PredictionListResponse(
        items=[PredictionResponse.from_record(r, settings.price_currency) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/predictions/{prediction_id}", response_model=PredictionResponse, tags=["predictions"]
)
def get_prediction(prediction_id: int, db: DbSession, settings: AppSettings) -> PredictionResponse:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found.")
    return PredictionResponse.from_record(record, settings.price_currency)


@router.delete(
    "/predictions/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["predictions"],
)
def delete_prediction(prediction_id: int, db: DbSession, settings: AppSettings) -> Response:
    if not settings.allow_delete:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Deleting predictions is disabled.")
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found.")
    try:
        db.delete(record)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("Delete failed: %s", exc.__class__.__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database is unavailable.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
