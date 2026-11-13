from fastapi.testclient import TestClient

from codeguard.main import app


def test_health_endpoint_returns_200() -> None:
    """Verify that the health endpoint returns HTTP 200."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_health_endpoint_returns_expected_response() -> None:
    """Verify the health endpoint response payload."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.json() == {
        "status": "healthy",
        "service": "CodeGuard API",
    }


def test_health_endpoint_returns_json() -> None:
    """Verify that the endpoint returns JSON."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.headers["content-type"].startswith("application/json")
