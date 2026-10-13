"""Thin HTTP client for the FastAPI backend (the frontend never touches the model or the DB)."""
from __future__ import annotations

from typing import Any

import requests


class ApiError(Exception):
    """Raised with a message that is safe to show to the user."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def format_validation_detail(detail: Any) -> str:
    """Turn FastAPI's 422 payload into readable lines."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        lines = []
        for item in detail:
            loc = ".".join(str(p) for p in item.get("loc", []) if p != "body")
            lines.append(f"{loc}: {item.get('msg', 'invalid value')}" if loc else item.get("msg", "invalid"))
        return "\n".join(lines)
    return "Request was rejected by the server."


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 15.0, session: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectionError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Is uvicorn running?") from exc
        except requests.exceptions.Timeout as exc:
            raise ApiError("The API took too long to respond.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"Request failed: {exc.__class__.__name__}") from exc

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = None
            raise ApiError(format_validation_detail(detail) if detail else f"HTTP {resp.status_code}", resp.status_code)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def health_database(self) -> dict:
        return self._request("GET", "/health/database")

    def model_features(self) -> dict:
        return self._request("GET", "/model/features")

    def predict(self, payload: dict[str, float]) -> dict:
        return self._request("POST", "/predict", json=payload)

    def history(self, limit: int = 20, offset: int = 0) -> dict:
        return self._request("GET", "/predictions", params={"limit": limit, "offset": offset})

    def get_prediction(self, prediction_id: int) -> dict:
        return self._request("GET", f"/predictions/{prediction_id}")

    def delete_prediction(self, prediction_id: int) -> None:
        self._request("DELETE", f"/predictions/{prediction_id}")
