from pathlib import Path

import numpy as np
import pytest

from app.config import BASE_DIR
from app.schemas.house import HouseFeatures
from app.services.prediction_service import ModelLoadError, PredictionError, PredictionService
from tests.conftest import VALID_PAYLOAD, StubModel

REAL_MODEL = BASE_DIR / "app" / "models" / "house_price_model.pkl"


def test_missing_model_file_raises_clear_error(tmp_path):
    service = PredictionService(tmp_path / "nope.pkl")
    with pytest.raises(ModelLoadError, match="not found"):
        service.load()


def test_corrupt_model_file_raises(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"this is not a pickle")
    with pytest.raises(ModelLoadError):
        PredictionService(bad).load()


def test_model_with_wrong_features_is_rejected():
    class Wrong(StubModel):
        feature_names_in_ = np.array(["a", "b"], dtype=object)

    with pytest.raises(ModelLoadError, match="expects features"):
        PredictionService.from_model(Wrong())


def test_prediction_value_and_price_conversion():
    service = PredictionService.from_model(StubModel(2.5), price_unit_multiplier=100_000)
    assert service.predict_value(HouseFeatures(**VALID_PAYLOAD)) == pytest.approx(2.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_model_output_is_rejected(bad):
    service = PredictionService.from_model(StubModel(bad))
    with pytest.raises(PredictionError):
        service.predict_value(HouseFeatures(**VALID_PAYLOAD))


def test_model_exception_is_wrapped():
    class Boom(StubModel):
        def predict(self, X):
            raise ValueError("internal secret detail")

    service = PredictionService.from_model(Boom())
    with pytest.raises(PredictionError) as info:
        service.predict_value(HouseFeatures(**VALID_PAYLOAD))
    assert "secret" not in str(info.value)


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="real model file not present")
def test_real_model_loads_and_predicts():
    pytest.importorskip("lightgbm")
    service = PredictionService(REAL_MODEL)
    service.load()
    assert service.is_loaded
    value = service.predict_value(HouseFeatures(**VALID_PAYLOAD))
    # Value obtained by running the provided .pkl directly (scikit-learn 1.6.1).
    assert value == pytest.approx(3.9045, abs=1e-3)
