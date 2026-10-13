"""Frontend client <-> real FastAPI app (in-process, SQLite, stub model)."""
from api_client import ApiClient, ApiError
import pytest

from tests.conftest import VALID_PAYLOAD


class AsgiSession:
    """Adapts Starlette's TestClient to the tiny `requests`-like surface ApiClient needs."""

    def __init__(self, test_client):
        self.c = test_client

    def request(self, method, url, timeout=None, **kw):
        path = "/" + url.split("/", 3)[3]
        return self.c.request(method, path, **kw)


def test_client_round_trip(client):
    api = ApiClient("http://testserver", session=AsgiSession(client))
    assert api.health()["status"] == "ok"
    features = api.model_features()["features"]
    assert len(features) == 8
    saved = api.predict(VALID_PAYLOAD)
    assert api.get_prediction(saved["prediction_id"])["predicted_price"] == saved["predicted_price"]
    assert api.history()["total"] == 1
    api.delete_prediction(saved["prediction_id"])
    with pytest.raises(ApiError) as e:
        api.get_prediction(saved["prediction_id"])
    assert e.value.status_code == 404
