from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, status
from pydantic import SecretStr
from starlette.requests import Request

import codeguard.api.dependencies as dependencies
from codeguard.api.dependencies import get_db_session, get_webhook_job_queue
from codeguard.context.repository_context_builder import (
    RepositoryContextBuilder,
)
from codeguard.core.config import Settings
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer
from codeguard.webhooks.job_queue import WebhookJobQueue


class FakeGitHubClient:
    """Fake client for dependency lifecycle tests."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeAsyncSession:
    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> FakeAsyncSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True


async def test_github_client_dependency_uses_secret_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []

    def fake_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        dependencies,
        "GitHubClient",
        fake_factory,
    )

    settings = Settings(
        _env_file=None,
        github_token=SecretStr("test-github-token"),
    )

    generator = dependencies.get_github_client(
        settings=settings,
    )

    client = await anext(generator)

    assert client.token == "test-github-token"
    assert client.closed is False
    assert len(created_clients) == 1

    await generator.aclose()

    assert client.closed is True


async def test_github_client_dependency_supports_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []

    def fake_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        dependencies,
        "GitHubClient",
        fake_factory,
    )

    settings = Settings(_env_file=None)

    generator = dependencies.get_github_client(
        settings=settings,
    )

    client = await anext(generator)

    assert client.token is None

    await generator.aclose()

    assert client.closed is True


async def test_db_session_dependency_yields_and_closes_session() -> None:
    app = FastAPI()
    session = FakeAsyncSession()
    app.state.database_session_factory = lambda: session
    request = Request({"type": "http", "app": app})
    session_dependency = get_db_session(request)

    yielded_session = await anext(session_dependency)

    assert yielded_session is session
    assert session.closed is False
    await session_dependency.aclose()
    assert session.closed is True


async def test_db_session_dependency_returns_503_when_database_unavailable() -> None:
    app = FastAPI()
    app.state.database_session_factory = None
    request = Request({"type": "http", "app": app})
    session_dependency = get_db_session(request)

    with pytest.raises(HTTPException) as exc_info:
        await anext(session_dependency)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "Database is unavailable."


def test_analyzer_dependency_uses_provided_github_client() -> None:
    client = FakeGitHubClient()

    analyzer = dependencies.get_pull_request_analyzer(
        github_client=client,
    )

    assert isinstance(analyzer, PullRequestAnalyzer)
    assert analyzer._github_client is client
    assert analyzer._ai_review_engine is None
    assert analyzer._ai_execution_coordinator is None
    assert isinstance(
        analyzer._repository_context_builder,
        RepositoryContextBuilder,
    )

    assert analyzer._repository_context_builder._github_client is client


def test_get_webhook_job_queue_returns_configured_queue(
    tmp_path,
) -> None:
    app = FastAPI()
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    app.state.webhook_job_queue = queue

    request = Request(
        {
            "type": "http",
            "app": app,
        }
    )

    assert get_webhook_job_queue(request) is queue


def test_get_webhook_job_queue_raises_when_not_configured() -> None:
    app = FastAPI()

    request = Request(
        {
            "type": "http",
            "app": app,
        }
    )

    with pytest.raises(
        RuntimeError,
        match="not configured",
    ):
        get_webhook_job_queue(request)
