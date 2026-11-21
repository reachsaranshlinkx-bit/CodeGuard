import hashlib
import hmac
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from codeguard.core.config import Settings
from codeguard.main import create_app

SECRET = "test-github-webhook-secret"
ENDPOINT = "/api/v1/webhooks/github"


@pytest.fixture
def app(tmp_path: Path) -> Iterator[FastAPI]:
    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_database_path=tmp_path / "webhook_jobs.sqlite3",
    )

    application = create_app(settings=settings)

    yield application

    application.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def make_payload(action: str = "opened") -> dict[str, Any]:
    return {
        "action": action,
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "reachsaranshlinkx-bit",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }


def make_headers(
    body: bytes,
    *,
    event: str | None = "pull_request",
    delivery: str | None = "test-delivery-123",
) -> dict[str, str]:
    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
    }

    if event is not None:
        headers["X-GitHub-Event"] = event

    if delivery is not None:
        headers["X-GitHub-Delivery"] = delivery

    return headers


@pytest.mark.parametrize("action", ["opened", "synchronize"])
def test_accepts_valid_pull_request_events(
    client: TestClient,
    action: str,
) -> None:
    body = json.dumps(make_payload(action)).encode("utf-8")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body),
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "queued",
        "delivery_id": "test-delivery-123",
        "event": "pull_request",
        "action": action,
        "owner": "reachsaranshlinkx-bit",
        "repo": "CodeGuard",
        "pull_number": 42,
    }


def test_rejects_invalid_signature(client: TestClient) -> None:
    body = json.dumps(make_payload()).encode("utf-8")
    headers = make_headers(body)
    headers["X-Hub-Signature-256"] = "sha256=" + ("0" * 64)

    response = client.post(
        ENDPOINT,
        content=body,
        headers=headers,
    )

    assert response.status_code == 401


def test_rejects_missing_signature(client: TestClient) -> None:
    body = json.dumps(make_payload()).encode("utf-8")
    headers = make_headers(body)
    headers.pop("X-Hub-Signature-256")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=headers,
    )

    assert response.status_code == 401


def test_fails_when_webhook_secret_is_not_configured(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        webhook_database_path=tmp_path / "missing_secret.sqlite3",
    )
    application = create_app(settings=settings)

    with TestClient(application) as test_client:
        response = test_client.post(
            ENDPOINT,
            content=b"{}",
        )

    application.dependency_overrides.clear()

    assert response.status_code == 503


@pytest.mark.parametrize(
    "missing_header",
    ["X-GitHub-Event", "X-GitHub-Delivery"],
)
def test_rejects_missing_github_headers(
    client: TestClient,
    missing_header: str,
) -> None:
    body = json.dumps(make_payload()).encode("utf-8")
    headers = make_headers(body)
    headers.pop(missing_header)

    response = client.post(
        ENDPOINT,
        content=body,
        headers=headers,
    )

    assert response.status_code == 400


def test_ignores_unsupported_github_event(client: TestClient) -> None:
    body = json.dumps({"action": "opened"}).encode("utf-8")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body, event="issues"),
    )

    assert response.status_code == 202
    assert response.json()["status"] == "ignored"
    assert response.json()["reason"] == "unsupported_event"


def test_ignores_unsupported_pull_request_action(
    client: TestClient,
) -> None:
    body = json.dumps(make_payload("closed")).encode("utf-8")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body),
    )

    assert response.status_code == 202
    assert response.json()["status"] == "ignored"
    assert response.json()["reason"] == "unsupported_pull_request_action"


def test_rejects_invalid_json(client: TestClient) -> None:
    body = b"{invalid-json"

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body),
    )

    assert response.status_code == 400


def test_rejects_non_object_json_payload(client: TestClient) -> None:
    body = json.dumps(["not", "an", "object"]).encode("utf-8")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body),
    )

    assert response.status_code == 400


def test_rejects_invalid_pull_request_number(client: TestClient) -> None:
    payload = make_payload()
    payload["pull_request"]["number"] = True
    body = json.dumps(payload).encode("utf-8")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body),
    )

    assert response.status_code == 400


def test_duplicate_delivery_is_not_enqueued_twice(
    app: FastAPI,
    client: TestClient,
) -> None:
    body = json.dumps(make_payload()).encode("utf-8")
    headers = make_headers(body)

    first_response = client.post(
        ENDPOINT,
        content=body,
        headers=headers,
    )

    second_response = client.post(
        ENDPOINT,
        content=body,
        headers=headers,
    )

    expected_response = {
        "status": "queued",
        "delivery_id": "test-delivery-123",
        "event": "pull_request",
        "action": "opened",
        "owner": "reachsaranshlinkx-bit",
        "repo": "CodeGuard",
        "pull_number": 42,
    }

    assert first_response.status_code == 202
    assert first_response.json() == expected_response

    assert second_response.status_code == 202
    assert second_response.json() == expected_response

    queue = app.state.webhook_job_queue
    job = queue.get_job("test-delivery-123")

    assert job is not None
    assert job.status == "queued"
    assert job.attempts == 0


def test_in_progress_delivery_is_not_claimed_again(
    app: FastAPI,
    client: TestClient,
) -> None:
    queue = app.state.webhook_job_queue

    queue.enqueue_job(
        delivery_id="test-delivery-123",
        event="pull_request",
        action="opened",
        owner="reachsaranshlinkx-bit",
        repo="CodeGuard",
        pull_number=42,
    )

    claimed_job = queue.claim_next_job()

    assert claimed_job is not None
    assert claimed_job.status == "processing"

    body = json.dumps(make_payload()).encode("utf-8")

    response = client.post(
        ENDPOINT,
        content=body,
        headers=make_headers(body),
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "in_progress",
        "delivery_id": "test-delivery-123",
    }

    job = queue.get_job("test-delivery-123")

    assert job is not None
    assert job.status == "processing"
    assert job.attempts == 1
