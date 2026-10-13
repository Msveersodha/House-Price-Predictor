"""Loads the trained pipeline once and turns validated input into predictions."""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Request
from sklearn.exceptions import InconsistentVersionWarning
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import Prediction
from app.schemas.house import FEATURE_NAMES, HouseFeatures

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """The model file is missing, unreadable, or incompatible with the API schema."""


class ModelNotLoadedError(RuntimeError):
    """predict() was called before the model was loaded."""


class PredictionError(RuntimeError):
    """The model failed or returned an unusable value."""


class PersistenceError(RuntimeError):
    """The prediction could not be saved to the database."""


class PredictionService:
    def __init__(self, model_path: Path, price_unit_multiplier: float = 100_000.0) -> None:
        self._model_path = Path(model_path)
        self._multiplier = price_unit_multiplier
        self._model: Any | None = None

    # ---- loading -------------------------------------------------------
    @classmethod
    def from_model(cls, model: Any, price_unit_multiplier: float = 100_000.0) -> "PredictionService":
        """Build a service around an in-memory model (used by tests)."""
        service = cls(Path("<in-memory>"), price_unit_multiplier)
        service._validate_model(model)
        service._model = model
        return service

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the .pkl (trusted local artifact only!) and validate it."""
        if not self._model_path.is_file():
            raise ModelLoadError(f"Model file not found: {self._model_path}")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = joblib.load(self._model_path)
            for w in caught:
                if issubclass(w.category, InconsistentVersionWarning):
                    logger.warning("Model/scikit-learn version mismatch: %s", w.message)
        except ModuleNotFoundError as exc:
            raise ModelLoadError(
                f"Model needs a package that is not installed: {exc.name}. "
                "Run: pip install -r requirements.txt"
            ) from exc
        except Exception as exc:  # corrupt file, wrong format, ...
            raise ModelLoadError(f"Could not load model: {exc.__class__.__name__}: {exc}") from exc

        self._validate_model(model)
        self._model = model
        logger.info("Model loaded from %s (%s)", self._model_path, type(model).__name__)

    @staticmethod
    def _validate_model(model: Any) -> None:
        if not hasattr(model, "predict"):
            raise ModelLoadError("Loaded object has no predict() method.")
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            if list(names) != list(FEATURE_NAMES):
                raise ModelLoadError(
                    f"Model expects features {list(names)} but the API schema defines "
                    f"{list(FEATURE_NAMES)}."
                )
        elif getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            raise ModelLoadError("Model feature count does not match the API schema.")

    # ---- predicting ----------------------------------------------------
    def predict_value(self, features: HouseFeatures) -> float:
        """Run the model; return the raw output (units of the training target)."""
        if self._model is None:
            raise ModelNotLoadedError("Model is not loaded.")
        # Named columns in the exact training order.
        frame = pd.DataFrame([features.model_dump()], columns=list(FEATURE_NAMES)).astype(float)
        try:
            raw = np.asarray(self._model.predict(frame), dtype=float).ravel()
        except Exception as exc:
            logger.exception("Model predict() failed")
            raise PredictionError("The model could not produce a prediction.") from exc
        if raw.size != 1 or not math.isfinite(raw[0]):
            logger.error("Model returned an unusable output: %r", raw)
            raise PredictionError("The model returned an invalid value.")
        return float(raw[0])

    def predict_and_save(self, db: Session, features: HouseFeatures) -> Prediction:
        """Predict, then persist input + output in one transaction."""
        value = self.predict_value(features)
        record = Prediction(
            **features.model_dump(),
            predicted_value=value,
            predicted_price=round(value * self._multiplier, 2),
        )
        try:
            db.add(record)
            db.commit()
            db.refresh(record)
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Saving prediction failed: %s", exc.__class__.__name__)
            logger.debug("details", exc_info=True)
            raise PersistenceError("Could not save the prediction.") from exc
        return record


def get_prediction_service(request: Request) -> PredictionService:
    """FastAPI dependency: the single service created at startup."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None or not service.is_loaded:
        raise ModelNotLoadedError("Model is not loaded.")
    return service
