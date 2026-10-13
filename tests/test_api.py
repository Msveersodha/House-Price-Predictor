import pytest
from sqlalchemy.exc import OperationalError

from app.database.session import get_db
from tests.conftest import VALID_PAYLOAD, StubModel


def test_root_and_health(client):
    assert client.get("/").json()["status"] == "running"
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_health_database_ok(client):
    assert client.get("/health/database").status_code == 200


def test_health_database_failure_is_generic(client):
    class Broken:
        def execute(self, *a, **k):
            raise OperationalError("SELECT 1", {}, Exception("Login failed for user 'sa' password=hunter2"))

        def close(self):
            pass

    def broken_db():
        yield Broken()

    client.app.dependency_overrides[get_db] = broken_db
    r = client.get("/health/database")
    assert r.status_code == 503
    assert "hunter2" not in r.text and "sa" not in r.json()["detail"].split()


def test_predict_success_is_saved(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 201
    body = r.json()
    assert body["prediction_id"] == 1
    assert body["predicted_price"] == pytest.approx(350_000)
    assert body["currency"] == "USD"
    assert body["features"]["MedInc"] == 8.3252
    assert "created_at" in body
    assert client.get("/predictions/1").json()["predicted_price"] == pytest.approx(350_000)


@pytest.mark.parametrize("missing", list(VALID_PAYLOAD))
def test_missing_feature_rejected(client, missing):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing}
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_field_rejected(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "Bedrooms": 3}).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [("MedInc", -1), ("MedInc", 0), ("Latitude", 90), ("Longitude", 10), ("HouseAge", -5),
     ("Population", 0), ("MedInc", "abc"), ("MedInc", "3.5"), ("MedInc", None)],
)
def test_invalid_values_rejected(client, field, value):
    assert client.post("/predict", json={**VALID_PAYLOAD, field: value}).status_code == 422


def test_nan_and_infinity_rejected(client):
    for token in ("NaN", "Infinity", "-Infinity"):
        raw = '{"MedInc": %s, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0, "Population": 322,' \
              ' "AveOccup": 2.5, "Latitude": 37.88, "Longitude": -122.23}' % token
        r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, token


def test_bad_model_output_returns_500_without_details(make_client):
    c = make_client(StubModel(float("nan")))
    r = c.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 500
    assert r.json() == {"detail": "Prediction failed."}


def test_database_failure_on_save_returns_503(client, db_session_factory):
    def failing_db():
        db = db_session_factory()
        db.commit = lambda: (_ for _ in ()).throw(
            OperationalError("INSERT ...", {}, Exception("Cannot open server 'prod-sql' pwd=secret"))
        )
        try:
            yield db
        finally:
            db.close()

    client.app.dependency_overrides[get_db] = failing_db
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503
    assert "secret" not in r.text and "INSERT" not in r.text


def test_history_pagination(client):
    for _ in range(5):
        client.post("/predict", json=VALID_PAYLOAD)
    page = client.get("/predictions", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 5 and len(page["items"]) == 2
    assert [i["prediction_id"] for i in page["items"]] == [5, 4]  # newest first
    last = client.get("/predictions", params={"limit": 2, "offset": 4}).json()
    assert [i["prediction_id"] for i in last["items"]] == [1]
    assert client.get("/predictions", params={"limit": 0}).status_code == 422
    assert client.get("/predictions", params={"limit": 101}).status_code == 422


def test_unknown_prediction_id_is_404(client):
    assert client.get("/predictions/999").status_code == 404
    assert client.delete("/predictions/999").status_code == 404


def test_delete(client):
    client.post("/predict", json=VALID_PAYLOAD)
    assert client.delete("/predictions/1").status_code == 204
    assert client.get("/predictions/1").status_code == 404


def test_delete_can_be_disabled(client, monkeypatch):
    from app.config import get_settings

    client.post("/predict", json=VALID_PAYLOAD)
    get_settings.cache_clear()
    monkeypatch.setenv("ALLOW_DELETE", "false")
    try:
        assert client.delete("/predictions/1").status_code == 403
    finally:
        get_settings.cache_clear()


def test_model_features_endpoint(client):
    body = client.get("/model/features").json()
    names = [f["name"] for f in body["features"]]
    assert names == ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
                     "AveOccup", "Latitude", "Longitude"]
    medinc = body["features"][0]
    assert medinc["exclusive_minimum"] == 0 and medinc["maximum"] == 20
