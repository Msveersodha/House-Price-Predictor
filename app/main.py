"""FastAPI application entry point.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.database import connection
from app.logging_config import configure_logging
from app.routers import prediction
from app.services.prediction_service import (
    ModelNotLoadedError,
    PersistenceError,
    PredictionError,
    PredictionService,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    # Fail fast: a missing/incompatible model stops startup with a clear message.
    service = PredictionService(settings.resolved_model_path, settings.price_unit_multiplier)
    service.load()
    app.state.prediction_service = service
    logger.info("API ready. The database is checked lazily (see /health/database).")
    yield
    if connection.get_engine.cache_info().currsize:
        connection.get_engine().dispose()


def _json_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="House Price Prediction API",
        description=(
            "Predicts median house value for a California block group using a trained "
            "scikit-learn pipeline (imputer + LightGBM) and stores every prediction in SQL Server."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # Clients get a generic message; details go to the server log only.
    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded(_: Request, __: ModelNotLoadedError):
        return _json_error(503, "The prediction model is not available.")

    @app.exception_handler(PredictionError)
    async def _prediction_error(_: Request, __: PredictionError):
        return _json_error(500, "Prediction failed.")

    @app.exception_handler(PersistenceError)
    async def _persistence_error(_: Request, __: PersistenceError):
        return _json_error(503, "The prediction could not be saved. Try again later.")

    @app.exception_handler(SQLAlchemyError)
    async def _db_error(_: Request, exc: SQLAlchemyError):
        logger.error("Database error: %s", exc.__class__.__name__)
        return _json_error(503, "Database is unavailable.")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        # Drop the echoed `input` (it may be NaN/inf, which JSON cannot encode) and `ctx`.
        errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.include_router(prediction.router)
    return app


app = create_app()
