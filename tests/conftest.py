import os

# Must be set before app modules are imported (Settings validates on first use).
os.environ.setdefault("DB_AUTH_MODE", "sql")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import Base
from app.database.session import get_db
from app.main import create_app
from app.schemas.house import FEATURE_NAMES
from app.services.prediction_service import PredictionService, get_prediction_service

VALID_PAYLOAD = {
    "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.9841, "AveBedrms": 1.0238,
    "Population": 322, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23,
}


class StubModel:
    """Fake model with the same interface; lets unit tests run without LightGBM."""

    feature_names_in_ = np.array(FEATURE_NAMES, dtype=object)
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, value: float = 3.5):
        self.value = value

    def predict(self, X):
        assert list(X.columns) == list(FEATURE_NAMES), "wrong column order"
        return np.full(len(X), self.value)


@pytest.fixture()
def db_session_factory():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def make_client(db_session_factory):
    def _make(model=None):
        app = create_app()
        service = PredictionService.from_model(model or StubModel())

        def _get_db():
            db = db_session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _get_db
        app.dependency_overrides[get_prediction_service] = lambda: service
        return TestClient(app, raise_server_exceptions=False)

    return _make


@pytest.fixture()
def client(make_client):
    return make_client()
