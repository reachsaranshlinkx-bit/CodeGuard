from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from codeguard.api.dependencies import get_pull_request_analyzer
from codeguard.core.config import Settings
from codeguard.github.exceptions import GitHubNotFoundError
from codeguard.main import create_app

API_KEY = "test-codeguard-api-key-for-integration-tests"
INVALID_API_KEY = "incorrect-codeguard-key"

API_PAYLOAD = {
    "owner": "octocat",
    "repo": "CodeGuard",
    "pull_number": 42,
}


class AnalyzerProbe:
    """Track whether an authenticated request reaches analysis."""

    def __init__(self) -> None:
        self.calls = 0

    async def analyze(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> Any:
        self.calls += 1

        raise GitHubNotFoundError(
            "Test GitHub repository not found.",
            status_code=404,
        )


def create_test_client(
    api_key: str | None,
) -> tuple[TestClient, AnalyzerProbe, Any]:
    settings_kwargs: dict[str, Any] = {
        "_env_file": None,
    }

    if api_key is not None:
        settings_kwargs["api_key"] = SecretStr(api_key)

    settings = Settings(**settings_kwargs)

    analyzer = AnalyzerProbe()
    app = create_app(settings=settings)

    app.dependency_overrides[get_pull_request_analyzer] = lambda: analyzer

    return TestClient(app), analyzer, app


def test_review_endpoint_rejects_missing_api_key() -> None:
    client, analyzer, app = create_test_client(API_KEY)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
            )

        assert response.status_code == 401
        assert analyzer.calls == 0

    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_rejects_invalid_api_key() -> None:
    client, analyzer, app = create_test_client(API_KEY)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers={
                    "X-CodeGuard-API-Key": INVALID_API_KEY,
                },
            )

        assert response.status_code == 401
        assert analyzer.calls == 0

    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_accepts_valid_api_key() -> None:
    client, analyzer, app = create_test_client(API_KEY)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers={
                    "X-CodeGuard-API-Key": API_KEY,
                },
            )

        # The authenticated request reaches the analyzer.
        # The analyzer intentionally returns a simulated 404.
        assert response.status_code == 404
        assert analyzer.calls == 1

    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_fails_when_api_key_is_not_configured() -> None:
    client, analyzer, app = create_test_client(None)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
            )

        assert response.status_code == 503
        assert analyzer.calls == 0

    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_remains_public() -> None:
    client, _, app = create_test_client(None)

    try:
        with client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    finally:
        app.dependency_overrides.clear()
