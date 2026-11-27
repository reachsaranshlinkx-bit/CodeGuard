import asyncio
import sqlite3
import time
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import codeguard.ai.factory as ai_factory
import codeguard.main as main
from codeguard.ai.exceptions import AIProviderConfigurationError
from codeguard.ai.execution_coordinator import AIExecutionCoordinator
from codeguard.ai.factory import (
    AIReviewProviderSelection,
    create_ai_review_provider_selection,
)
from codeguard.ai.mock_provider import MockAIReviewProvider
from codeguard.ai.openai_provider import OpenAIReviewProvider
from codeguard.ai.operation_repository import AIOperationRepository
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


class ClosableMockAIReviewProvider(MockAIReviewProvider):
    def __init__(self, *, profile_identity: str) -> None:
        super().__init__(profile_identity=profile_identity)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_worker_remains_disabled_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider_selection_calls = []

    def unexpected_client_creation(
        token: str | None = None,
    ) -> FakeGitHubClient:
        raise AssertionError("GitHub client should not be created.")

    def record_provider_selection_factory(settings: Settings) -> None:
        provider_selection_calls.append(settings)

    monkeypatch.setattr(
        main,
        "GitHubClient",
        unexpected_client_creation,
    )
    monkeypatch.setattr(
        main,
        "create_ai_review_provider_selection",
        record_provider_selection_factory,
        raising=False,
    )

    settings = Settings(
        _env_file=None,
        webhook_database_path=tmp_path / "disabled.sqlite3",
    )

    app = main.create_app(settings=settings)

    with TestClient(app):
        assert app.state.webhook_worker is None
        assert provider_selection_calls == []


def test_mock_provider_is_injected_into_analyzer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    analyzers = []
    provider_selections = []
    coordinator_arguments = []
    coordinators = []
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
        worker_started.set()
        await stop_event.wait()

    class CapturingReviewProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            analyzers.append(analyzer)

    def provider_selection_factory_spy(settings: Settings):
        selection = create_ai_review_provider_selection(settings)
        provider_selections.append((settings, selection))
        return selection

    real_coordinator = main.AIExecutionCoordinator

    def coordinator_factory_spy(**kwargs):
        coordinator_arguments.append(kwargs)
        coordinator = real_coordinator(**kwargs)
        coordinators.append(coordinator)
        return coordinator

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(main, "run_webhook_worker", fake_worker_loop)
    monkeypatch.setattr(main, "WebhookReviewProcessor", CapturingReviewProcessor)
    monkeypatch.setattr(
        main,
        "create_ai_review_provider_selection",
        provider_selection_factory_spy,
        raising=False,
    )
    monkeypatch.setattr(main, "AIExecutionCoordinator", coordinator_factory_spy)

    settings = Settings(
        _env_file=None,
        environment="testing",
        ai_provider="mock",
        webhook_worker_enabled=True,
        webhook_database_path=tmp_path / "mock-provider.sqlite3",
    )
    app = main.create_app(settings=settings)

    with TestClient(app):
        assert worker_started.wait(timeout=2)
        assert len(analyzers) == 1
        assert analyzers[0]._ai_review_engine is None
        assert isinstance(
            analyzers[0]._ai_execution_coordinator,
            AIExecutionCoordinator,
        )
        assert isinstance(
            analyzers[0]._ai_execution_coordinator._provider_selection.provider,
            MockAIReviewProvider,
        )
        assert len(provider_selections) == 1
        selected_settings, selection = provider_selections[0]
        assert selected_settings is settings
        assert selection is not None
        assert len(coordinator_arguments) == 1
        assert coordinator_arguments[0]["provider_selection"] is selection
        assert coordinator_arguments[0]["repository"] is (
            app.state.ai_operation_repository
        )
        assert len(coordinators) == 1
        assert analyzers[0]._ai_execution_coordinator is coordinators[0]
        assert coordinators[0]._provider_selection is selection
        assert coordinators[0]._repository is app.state.ai_operation_repository
        assert coordinators[0]._lease_duration_seconds == 300.0
        assert isinstance(app.state.ai_operation_repository, AIOperationRepository)
        assert app.state.ai_operation_repository.database_path == (
            settings.webhook_database_path
        )
        assert app.state.webhook_job_queue.database_path == (
            settings.webhook_database_path
        )
        with sqlite3.connect(settings.webhook_database_path) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (3,)

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_selected_provider_client_is_closed_during_lifespan_shutdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    selected_providers: list[ClosableMockAIReviewProvider] = []
    worker_started = Event()
    original_selection_factory = main.create_ai_review_provider_selection

    def fake_client_factory(token: str | None = None) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    def provider_selection_factory(settings: Settings):
        baseline = original_selection_factory(settings)
        assert baseline is not None
        profile_identity = baseline.metadata.model_identity
        provider = ClosableMockAIReviewProvider(
            profile_identity=profile_identity,
        )
        selected_providers.append(provider)
        return AIReviewProviderSelection(
            provider=provider,
            metadata=baseline.metadata,
        )

    async def fake_worker_loop(
        *,
        worker,
        poll_interval_seconds: int,
        stop_event,
    ) -> None:
        worker_started.set()
        await stop_event.wait()

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(
        main,
        "create_ai_review_provider_selection",
        provider_selection_factory,
    )
    monkeypatch.setattr(main, "run_webhook_worker", fake_worker_loop)

    app = main.create_app(
        settings=Settings(
            _env_file=None,
            environment="testing",
            ai_provider="mock",
            webhook_worker_enabled=True,
            webhook_database_path=tmp_path / "provider-lifecycle.sqlite3",
        )
    )

    with TestClient(app):
        assert worker_started.wait(timeout=2)
        assert app.state.webhook_worker is not None
        assert len(selected_providers) == 1
        assert selected_providers[0].closed is False

    assert selected_providers[0].closed is True
    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_openai_provider_is_selected_injected_and_closed_during_lifespan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    created_providers: list[OpenAIReviewProvider] = []
    analyzers = []
    worker_started = Event()
    provider_requests: list[httpx.Request] = []

    def fake_client_factory(token: str | None = None) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    def provider_transport(request: httpx.Request) -> httpx.Response:
        provider_requests.append(request)
        return httpx.Response(200, json={}, request=request)

    class MockTransportOpenAIReviewProvider(OpenAIReviewProvider):
        def __init__(
            self,
            *,
            model_identity: str,
            api_key: SecretStr | str,
            timeout: float = 30.0,
        ) -> None:
            super().__init__(
                model_identity=model_identity,
                api_key=api_key,
                timeout=timeout,
                transport=httpx.MockTransport(provider_transport),
            )
            created_providers.append(self)

    class CapturingReviewProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            analyzers.append(analyzer)

    async def fake_worker_loop(
        *,
        worker,
        poll_interval_seconds: int,
        stop_event,
    ) -> None:
        worker_started.set()
        await stop_event.wait()

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        CapturingReviewProcessor,
    )
    monkeypatch.setattr(main, "run_webhook_worker", fake_worker_loop)
    monkeypatch.setattr(
        ai_factory,
        "OpenAIReviewProvider",
        MockTransportOpenAIReviewProvider,
    )

    settings = Settings(
        _env_file=None,
        environment="testing",
        ai_provider="openai",
        openai_api_key="test-openai-key-not-a-real-credential",
        openai_model="gpt-test-model",
        openai_request_timeout_seconds=8.5,
        webhook_worker_enabled=True,
        webhook_worker_max_attempts=3,
        webhook_worker_retry_delay_seconds=11,
        webhook_database_path=tmp_path / "openai-provider-lifecycle.sqlite3",
    )
    app = main.create_app(settings=settings)

    with TestClient(app):
        assert worker_started.wait(timeout=2)
        assert len(created_providers) == 1
        assert len(analyzers) == 1

        coordinator = analyzers[0]._ai_execution_coordinator
        assert isinstance(coordinator, AIExecutionCoordinator)
        assert coordinator._provider_selection.provider is created_providers[0]
        assert coordinator._provider_selection.metadata.provider_identity == "openai"
        assert (
            coordinator._provider_selection.metadata.model_identity == "gpt-test-model"
        )
        assert app.state.webhook_worker._max_attempts == 3
        assert app.state.webhook_worker._retry_delay_seconds == 11
        assert created_providers[0]._client._client.is_closed is False
        assert created_providers[0]._client._client.timeout.connect == 8.5
        assert provider_requests == []

    assert created_providers[0]._client._client.is_closed is True
    assert len(created_clients) == 1
    assert created_clients[0].closed is True
    assert provider_requests == []


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
        assert processor.analyzer._ai_review_engine is None
        assert processor.analyzer._ai_execution_coordinator is None

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


def test_github_client_is_closed_if_provider_selection_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    def failing_provider_selection_factory(settings: Settings):
        raise RuntimeError("provider selection failed")

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(
        main,
        "create_ai_review_provider_selection",
        failing_provider_selection_factory,
        raising=False,
    )

    settings = Settings(
        _env_file=None,
        webhook_worker_enabled=True,
        webhook_database_path=tmp_path / "provider-failure.sqlite3",
    )
    app = main.create_app(settings=settings)

    with pytest.raises(RuntimeError, match="provider selection failed"):
        with TestClient(app):
            pass

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_coordinator_construction_failure_does_not_fall_back_to_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    analyzer_construction_attempts = []

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    def failing_coordinator_factory(**kwargs):
        raise RuntimeError("coordinator construction failed")

    def record_analyzer_construction(**kwargs):
        analyzer_construction_attempts.append(kwargs)
        raise AssertionError("Analyzer construction must not be attempted.")

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(
        main,
        "AIExecutionCoordinator",
        failing_coordinator_factory,
    )
    monkeypatch.setattr(main, "PullRequestAnalyzer", record_analyzer_construction)

    app = main.create_app(
        settings=Settings(
            _env_file=None,
            environment="testing",
            ai_provider="mock",
            webhook_worker_enabled=True,
            webhook_database_path=tmp_path / "coordinator-failure.sqlite3",
        )
    )

    with pytest.raises(RuntimeError, match="coordinator construction failed"):
        with TestClient(app):
            pass

    assert analyzer_construction_attempts == []
    assert len(created_clients) == 1
    assert created_clients[0].closed is True


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_mock_provider_cannot_start_in_staging_or_production(
    environment: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []

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
        await stop_event.wait()

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(main, "run_webhook_worker", fake_worker_loop)

    settings = Settings.model_construct(
        environment=environment,
        ai_provider="mock",
        webhook_worker_enabled=True,
        webhook_database_path=tmp_path / f"{environment}-mock.sqlite3",
    )
    app = main.create_app(settings=settings)

    with pytest.raises(AIProviderConfigurationError):
        with TestClient(app):
            pass

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
