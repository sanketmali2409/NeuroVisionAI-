"""Tests for the health/readiness endpoints.

These are the first tests in the suite because they exercise the whole
wiring path (config -> app factory -> DB init -> router) without needing
any trained model - a good smoke test that the scaffold itself is sound.
"""

from fastapi.testclient import TestClient

from backend.core.app_factory import create_app


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_liveness_returns_ok():
    client = _client()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "NeuroVision AI"


def test_readiness_reports_model_flags():
    client = _client()
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "classification_model" in body["models"]
    assert "segmentation_model" in body["models"]
    # No models are trained yet at scaffold stage, so these must be False
    # rather than raising - proves the readiness check fails gracefully.
    assert body["models"]["classification_model"] is False
