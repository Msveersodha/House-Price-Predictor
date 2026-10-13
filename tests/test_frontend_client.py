import pytest
import requests

from api_client import ApiClient, ApiError, format_validation_detail


class FakeResponse:
    def __init__(self, status=200, json_data=None):
        self.status_code = status
        self._json = json_data
        self.content = b"x" if json_data is not None else b""

    def json(self):
        if self._json is None:
            raise ValueError
        return self._json


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self.exc:
            raise self.exc
        return self.response


def test_predict_posts_json_to_api():
    s = FakeSession(FakeResponse(201, {"prediction_id": 1}))
    out = ApiClient("http://api:8000/", session=s).predict({"MedInc": 3})
    assert out == {"prediction_id": 1}
    method, url, kw = s.calls[0]
    assert (method, url, kw["json"]) == ("POST", "http://api:8000/predict", {"MedInc": 3})


def test_connection_error_has_friendly_message():
    c = ApiClient("http://x", session=FakeSession(exc=requests.exceptions.ConnectionError()))
    with pytest.raises(ApiError, match="Cannot reach"):
        c.health()


def test_timeout_message():
    c = ApiClient("http://x", session=FakeSession(exc=requests.exceptions.Timeout()))
    with pytest.raises(ApiError, match="too long"):
        c.health()


def test_validation_errors_are_formatted():
    detail = [{"loc": ["body", "MedInc"], "msg": "Input should be greater than 0"}]
    s = FakeSession(FakeResponse(422, {"detail": detail}))
    with pytest.raises(ApiError) as info:
        ApiClient("http://x", session=s).predict({})
    assert "MedInc: Input should be greater than 0" in info.value.message
    assert info.value.status_code == 422


def test_204_returns_none():
    assert ApiClient("http://x", session=FakeSession(FakeResponse(204))).delete_prediction(1) is None


def test_format_detail_string():
    assert format_validation_detail("nope") == "nope"
