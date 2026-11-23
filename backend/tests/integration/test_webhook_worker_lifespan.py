import asyncio
import time
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

import codeguard.main as main
from codeguard.context.repository_context_builder import (
    RepositoryContextBuilder,
)
from codeguard.core.config import Settings
from codeguard.main import run_webhook_worker


class FakeGitHubClient:
    """Fake GitHub client for application lifecycle tests."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_worker_remains_disabled_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_client_creation(
        token: str | None = None,
    ) -> FakeGitHubClient:
        raise AssertionError("GitHub client should not be created.")

    monkeypatch.setattr(
        main,
        "GitHubClient",
        unexpected_client_creation,
    )

    settings = Settings(
        _env_file=None,
        webhook_database_path=tmp_path / "disabled.sqlite3",
    )

    app = main.create_app(settings=settings)

    with TestClient(app):
        assert app.state.webhook_worker is None


def test_enabled_worker_starts_and_shuts_down_cleanly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    worker_started = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    async def fake_worker_loop(
        *,
        worker,
        poll_interval_seconds: int,
        stop_event,
    ) -> None:
        assert worker is not None
        assert poll_interval_seconds == 2
        worker_started.set()
        await stop_event.wait()

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "run_webhook_worker",
        fake_worker_loop,
    )

    settings = Settings(
        _env_file=None,
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=2,
        webhook_database_path=tmp_path / "enabled.sqlite3",
    )

    app = main.create_app(settings=settings)

    with TestClient(app):
        assert worker_started.wait(timeout=2)
        assert app.state.webhook_worker is not None
        assert len(created_clients) == 1
        assert created_clients[0].closed is False

    # TestClient exiting triggers application shutdown.
    assert created_clients[0].closed is True


async def test_background_loop_continues_after_processing_job() -> None:
    stop_event = asyncio.Event()

    class FakeWorker:
        def __init__(self) -> None:
            self.calls = 0

        async def run_once(self) -> bool:
            self.calls += 1

            if self.calls == 1:
                return True

            stop_event.set()
            return False

    worker = FakeWorker()

    await run_webhook_worker(
        worker=worker,
        poll_interval_seconds=1,
        stop_event=stop_event,
    )

    assert worker.calls == 2


def test_enabled_lifespan_processes_queued_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors = []

    processed_event = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    class FakeReviewProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            self.arguments = None
            self.analyzer = analyzer
            self.github_client = github_client

        async def process(
            self,
            *,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> None:
            self.arguments = (owner, repo, pull_number)
            processed_event.set()

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
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "processing.sqlite3",
    )

    app = main.create_app(settings=settings)

    queue = app.state.webhook_job_queue

    queue.enqueue_job(
        delivery_id="integration-worker-001",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    with TestClient(app):
        assert processed_event.wait(timeout=3)

        assert len(processors) == 1

        processor = processors[0]

        assert isinstance(
            processor.analyzer._repository_context_builder,
            RepositoryContextBuilder,
        )

        assert (
            processor.analyzer._repository_context_builder._github_client
            is processor.github_client
        )

        assert processor.arguments == (
            "octocat",
            "CodeGuard",
            42,
        )

        deadline = time.monotonic() + 3

        saved = queue.get_job("integration-worker-001")

        while (
            saved is not None
            and saved.status != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            saved = queue.get_job("integration-worker-001")

        assert saved is not None
        assert saved.status == "completed"
        assert saved.attempts == 1
        assert saved.lease_expires_at is None

        assert app.state.webhook_worker is not None

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_shutdown_waits_for_active_review_to_finish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processing_started = Event()
    processing_finished = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    class SlowFakeReviewProcessor:
        async def process(
            self,
            *,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> None:
            processing_started.set()

            # Simulate an active review that needs time to finish.
            await asyncio.sleep(1)

            processing_finished.set()

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> SlowFakeReviewProcessor:
        return SlowFakeReviewProcessor()

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
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "graceful_shutdown.sqlite3",
    )

    app = main.create_app(settings=settings)

    queue = app.state.webhook_job_queue

    queue.enqueue_job(
        delivery_id="shutdown-worker-001",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    with TestClient(app):
        assert processing_started.wait(timeout=3)

        # The review must still be running at this point.
        assert not processing_finished.is_set()

    # Application shutdown must wait for the active review.
    assert processing_finished.is_set()

    job = queue.get_job("shutdown-worker-001")

    assert job is not None
    assert job.status == "completed"
    assert job.attempts == 1
    assert job.lease_expires_at is None

    assert len(created_clients) == 1
    assert created_clients[0].closed is True
