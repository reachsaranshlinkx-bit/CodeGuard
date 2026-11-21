import hashlib
import hmac
import json
import time
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
from pydantic import SecretStr

import codeguard.main as main
from codeguard.core.config import Settings
from codeguard.services.webhook_job_worker import WebhookJobWorker

SECRET = "test-webhook-worker-secret"


class FakeGitHubClient:
    """Fake GitHub client for webhook worker integration tests."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeReviewProcessor:
    """Fake review processor that records processed pull requests."""

    def __init__(
        self,
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> None:
        self.arguments = None
        self.processed_event = Event()

    async def process(
        self,
        *,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> None:
        self.arguments = (owner, repo, pull_number)
        self.processed_event.set()


def test_webhook_is_received_and_processed_by_background_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors: list[FakeReviewProcessor] = []

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> FakeReviewProcessor:
        processor = FakeReviewProcessor(
            analyzer=analyzer,
            github_client=github_client,
            publishing_enabled=publishing_enabled,
        )
        processors.append(processor)
        return processor

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        fake_processor_factory,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "webhook_worker.sqlite3",
    )

    app = main.create_app(settings=settings)

    delivery_id = "webhook-worker-integration-001"

    payload = {
        "action": "opened",
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "octocat",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert response.json()["delivery_id"] == delivery_id

        assert len(processors) == 1

        processor = processors[0]

        assert processor.processed_event.wait(timeout=5)

        assert processor.arguments == (
            "octocat",
            "CodeGuard",
            42,
        )

        deadline = time.monotonic() + 3
        job = app.state.webhook_job_queue.get_job(delivery_id)

        while (
            job is not None
            and job.status != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = app.state.webhook_job_queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "completed"
        assert job.attempts == 1
        assert job.lease_expires_at is None

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_webhook_worker_retries_failed_processing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors = []
    succeeded_event = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    class FlakyReviewProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            self.calls = 0

        async def process(
            self,
            *,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> None:
            self.calls += 1

            if self.calls == 1:
                raise RuntimeError("Temporary GitHub API error")

            succeeded_event.set()

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> FlakyReviewProcessor:
        processor = FlakyReviewProcessor(
            analyzer=analyzer,
            github_client=github_client,
            publishing_enabled=publishing_enabled,
        )
        processors.append(processor)
        return processor

    original_worker_class = main.WebhookJobWorker

    def fake_worker_factory(
        *,
        queue,
        processor,
    ) -> WebhookJobWorker:
        return original_worker_class(
            queue=queue,
            processor=processor,
            retry_delay_seconds=0,
        )

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        fake_processor_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookJobWorker",
        fake_worker_factory,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "webhook_retry.sqlite3",
    )

    app = main.create_app(settings=settings)

    delivery_id = "webhook-retry-integration-001"

    payload = {
        "action": "opened",
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "octocat",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    queue = app.state.webhook_job_queue

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"

        assert succeeded_event.wait(timeout=5)

        assert len(processors) == 1
        assert processors[0].calls == 2

        deadline = time.monotonic() + 3
        job = queue.get_job(delivery_id)

        while (
            job is not None
            and job.status != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "completed"
        assert job.attempts == 2
        assert job.last_error is None
        assert job.lease_expires_at is None

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_webhook_worker_stops_after_maximum_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors = []
    second_attempt_event = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    class AlwaysFailingProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            self.calls = 0

        async def process(
            self,
            *,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> None:
            self.calls += 1

            if self.calls == 2:
                second_attempt_event.set()

            raise RuntimeError("Persistent processing failure")

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> AlwaysFailingProcessor:
        processor = AlwaysFailingProcessor(
            analyzer=analyzer,
            github_client=github_client,
            publishing_enabled=publishing_enabled,
        )
        processors.append(processor)
        return processor

    original_worker_class = main.WebhookJobWorker

    def fake_worker_factory(
        *,
        queue,
        processor,
    ) -> WebhookJobWorker:
        return original_worker_class(
            queue=queue,
            processor=processor,
            retry_delay_seconds=0,
            max_attempts=2,
        )

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        fake_processor_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookJobWorker",
        fake_worker_factory,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "webhook_exhaustion.sqlite3",
    )

    app = main.create_app(settings=settings)

    delivery_id = "webhook-exhaustion-integration-001"

    payload = {
        "action": "opened",
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "octocat",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    queue = app.state.webhook_job_queue

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"

        assert second_attempt_event.wait(timeout=5)

        deadline = time.monotonic() + 3
        job = queue.get_job(delivery_id)

        while (
            job is not None and job.status != "failed" and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "failed"
        assert job.attempts == 2
        assert job.lease_expires_at is None
        assert job.last_error == "Persistent processing failure"

        assert len(processors) == 1
        assert processors[0].calls == 2

    assert len(created_clients) == 1
    assert created_clients[0].closed is True
