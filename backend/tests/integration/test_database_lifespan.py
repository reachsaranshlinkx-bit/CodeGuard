from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import codeguard.main as main
from codeguard.core.config import Settings


class FakeConnection:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.statements: list[str] = []

    async def execute(self, statement: object) -> None:
        self.statements.append(str(statement))
        if self.failure is not None:
            raise self.failure


class FakeConnectionContext:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class FakeDatabaseEngine:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.disposed = False

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self.connection)

    async def dispose(self) -> None:
        self.disposed = True


class FakeGitHubClient:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_error = close_error
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@pytest.mark.parametrize("worker_enabled", [False, True])
def test_database_initializes_and_disposes_with_either_worker_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_enabled: bool,
) -> None:
    connection = FakeConnection()
    engine = FakeDatabaseEngine(connection)
    session_factory = object()

    def create_engine(database_url: SecretStr) -> FakeDatabaseEngine:
        assert database_url.get_secret_value().endswith("@localhost:5434/codeguard")
        return engine

    def create_factory(actual_engine: FakeDatabaseEngine) -> object:
        assert actual_engine is engine
        return session_factory

    async def wait_for_shutdown(*, stop_event, **kwargs: object) -> None:
        await stop_event.wait()

    monkeypatch.setattr(main, "create_database_engine", create_engine)
    monkeypatch.setattr(main, "create_session_factory", create_factory)
    monkeypatch.setattr(main, "run_webhook_worker", wait_for_shutdown)
    app = main.create_app(
        settings=Settings(
            _env_file=None,
            database_url=SecretStr(
                "postgresql+psycopg://test_user:test_password@localhost:5434/codeguard"
            ),
            webhook_worker_enabled=worker_enabled,
            webhook_database_path=tmp_path / f"worker-{worker_enabled}.sqlite3",
        )
    )

    with TestClient(app):
        assert app.state.database_engine is engine
        assert app.state.database_session_factory is session_factory
        assert connection.statements == ["SELECT 1"]
        if worker_enabled:
            assert app.state.webhook_worker is not None
        else:
            assert app.state.webhook_worker is None

    assert engine.disposed is True
    assert app.state.database_engine is None
    assert app.state.database_session_factory is None


def test_database_is_unavailable_without_a_configured_url(tmp_path: Path) -> None:
    app = main.create_app(
        settings=Settings(
            _env_file=None,
            webhook_database_path=tmp_path / "no-database.sqlite3",
        )
    )

    assert app.state.database_engine is None
    assert app.state.database_session_factory is None

    with TestClient(app):
        assert app.state.database_engine is None
        assert app.state.database_session_factory is None


def test_database_engine_is_disposed_when_connectivity_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_url = "postgresql+psycopg://test_user:test_password@localhost:5434/codeguard"
    connection = FakeConnection(failure=RuntimeError(secret_url))
    engine = FakeDatabaseEngine(connection)
    monkeypatch.setattr(main, "create_database_engine", lambda database_url: engine)
    monkeypatch.setattr(main, "create_session_factory", lambda actual_engine: object())
    app = main.create_app(
        settings=Settings(
            _env_file=None,
            database_url=SecretStr(secret_url),
            webhook_database_path=tmp_path / "startup-failure.sqlite3",
        )
    )

    with pytest.raises(
        RuntimeError, match="Database initialization failed"
    ) as exc_info:
        with TestClient(app):
            pytest.fail("Application startup should fail.")

    assert secret_url not in str(exc_info.value)
    assert engine.disposed is True
    assert app.state.database_engine is None
    assert app.state.database_session_factory is None


def test_database_engine_is_disposed_when_worker_startup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeDatabaseEngine(FakeConnection())
    github_client = FakeGitHubClient()

    def fail_provider_setup(settings: Settings) -> None:
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(main, "create_database_engine", lambda database_url: engine)
    monkeypatch.setattr(main, "create_session_factory", lambda actual_engine: object())
    monkeypatch.setattr(main, "GitHubClient", lambda token=None: github_client)
    monkeypatch.setattr(
        main,
        "create_ai_review_provider_selection",
        fail_provider_setup,
    )
    app = main.create_app(
        settings=Settings(
            _env_file=None,
            database_url=SecretStr(
                "postgresql+psycopg://test_user:test_password@localhost:5434/codeguard"
            ),
            webhook_worker_enabled=True,
            webhook_database_path=tmp_path / "worker-startup-failure.sqlite3",
        )
    )

    with pytest.raises(RuntimeError, match="provider setup failed"):
        with TestClient(app):
            pytest.fail("Application startup should fail.")

    assert github_client.closed is True
    assert engine.disposed is True


def test_database_engine_is_disposed_if_client_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeDatabaseEngine(FakeConnection())
    github_client = FakeGitHubClient(close_error=RuntimeError("client close failed"))

    async def wait_for_shutdown(*, stop_event, **kwargs: object) -> None:
        await stop_event.wait()

    monkeypatch.setattr(main, "create_database_engine", lambda database_url: engine)
    monkeypatch.setattr(main, "create_session_factory", lambda actual_engine: object())
    monkeypatch.setattr(main, "GitHubClient", lambda token=None: github_client)
    monkeypatch.setattr(main, "run_webhook_worker", wait_for_shutdown)
    app = main.create_app(
        settings=Settings(
            _env_file=None,
            database_url=SecretStr(
                "postgresql+psycopg://test_user:test_password@localhost:5434/codeguard"
            ),
            webhook_worker_enabled=True,
            webhook_database_path=tmp_path / "shutdown-failure.sqlite3",
        )
    )

    with pytest.raises(RuntimeError, match="client close failed"):
        with TestClient(app):
            assert app.state.database_engine is engine

    assert github_client.closed is True
    assert engine.disposed is True
    assert app.state.database_engine is None
    assert app.state.database_session_factory is None
