import pytest
from pydantic import ValidationError

from codeguard.core.config import Settings


def test_settings_use_expected_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "CodeGuard API"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.log_level == "INFO"


def test_settings_read_environment_variables(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODEGUARD_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "testing")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.environment == "testing"


def test_github_token_defaults_to_none(monkeypatch) -> None:
    monkeypatch.delenv("CODEGUARD_GITHUB_TOKEN", raising=False)

    settings = Settings(_env_file=None)

    assert settings.github_token is None


def test_github_token_is_loaded_as_secret(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "CODEGUARD_GITHUB_TOKEN",
        "test-github-token",
    )

    settings = Settings(_env_file=None)

    assert settings.github_token is not None
    assert settings.github_token.get_secret_value() == "test-github-token"
    assert str(settings.github_token) == "**********"
    assert "test-github-token" not in repr(settings)


def test_api_key_defaults_to_none(monkeypatch) -> None:
    monkeypatch.delenv("CODEGUARD_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.api_key is None


def test_api_key_is_loaded_as_secret(monkeypatch) -> None:
    monkeypatch.setenv(
        "CODEGUARD_API_KEY",
        "test-codeguard-api-key",
    )

    settings = Settings(_env_file=None)

    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "test-codeguard-api-key"
    assert str(settings.api_key) == "**********"
    assert "test-codeguard-api-key" not in repr(settings)


def test_production_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "production")
    monkeypatch.setenv("CODEGUARD_DEBUG", "false")
    monkeypatch.setenv("CODEGUARD_GITHUB_TOKEN", "test-github-token")
    monkeypatch.delenv("CODEGUARD_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="CODEGUARD_API_KEY"):
        Settings(_env_file=None)


def test_production_requires_github_token(monkeypatch) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "production")
    monkeypatch.setenv("CODEGUARD_DEBUG", "false")
    monkeypatch.setenv("CODEGUARD_API_KEY", "test-api-key")
    monkeypatch.delenv("CODEGUARD_GITHUB_TOKEN", raising=False)

    with pytest.raises(ValidationError, match="CODEGUARD_GITHUB_TOKEN"):
        Settings(_env_file=None)


def test_production_rejects_debug_mode(monkeypatch) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "production")
    monkeypatch.setenv("CODEGUARD_DEBUG", "true")
    monkeypatch.setenv("CODEGUARD_API_KEY", "test-api-key")
    monkeypatch.setenv("CODEGUARD_GITHUB_TOKEN", "test-github-token")

    with pytest.raises(ValidationError, match="CODEGUARD_DEBUG"):
        Settings(_env_file=None)


def test_production_accepts_valid_configuration(monkeypatch) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "production")
    monkeypatch.setenv("CODEGUARD_DEBUG", "false")
    monkeypatch.setenv("CODEGUARD_API_KEY", "test-api-key")
    monkeypatch.setenv("CODEGUARD_GITHUB_TOKEN", "test-github-token")
    monkeypatch.setenv(
        "CODEGUARD_GITHUB_WEBHOOK_SECRET",
        "test-webhook-secret",
    )
    monkeypatch.setenv("CODEGUARD_ENABLE_REVIEW_PUBLISHING", "false")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.debug is False
    assert settings.api_key is not None
    assert settings.github_token is not None
    assert settings.github_webhook_secret is not None
    assert settings.enable_review_publishing is False


def test_github_webhook_secret_defaults_to_none(monkeypatch) -> None:
    monkeypatch.delenv("CODEGUARD_GITHUB_WEBHOOK_SECRET", raising=False)

    settings = Settings(_env_file=None)

    assert settings.github_webhook_secret is None


def test_github_webhook_secret_is_loaded_as_secret(monkeypatch) -> None:
    monkeypatch.setenv(
        "CODEGUARD_GITHUB_WEBHOOK_SECRET",
        "test-webhook-secret",
    )

    settings = Settings(_env_file=None)

    assert settings.github_webhook_secret is not None
    assert settings.github_webhook_secret.get_secret_value() == "test-webhook-secret"
    assert str(settings.github_webhook_secret) == "**********"
    assert "test-webhook-secret" not in repr(settings)


def test_production_requires_github_webhook_secret(monkeypatch) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "production")
    monkeypatch.setenv("CODEGUARD_DEBUG", "false")
    monkeypatch.setenv("CODEGUARD_API_KEY", "test-api-key")
    monkeypatch.setenv("CODEGUARD_GITHUB_TOKEN", "test-github-token")
    monkeypatch.delenv("CODEGUARD_GITHUB_WEBHOOK_SECRET", raising=False)

    with pytest.raises(
        ValidationError,
        match="CODEGUARD_GITHUB_WEBHOOK_SECRET",
    ):
        Settings(_env_file=None)


def test_webhook_worker_settings_use_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEGUARD_WEBHOOK_WORKER_ENABLED", raising=False)
    monkeypatch.delenv(
        "CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS",
        raising=False,
    )

    settings = Settings(_env_file=None)

    assert settings.webhook_worker_enabled is False
    assert settings.webhook_worker_poll_interval_seconds == 1


def test_webhook_worker_settings_load_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEGUARD_WEBHOOK_WORKER_ENABLED", "true")
    monkeypatch.setenv(
        "CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS",
        "3",
    )

    settings = Settings(_env_file=None)

    assert settings.webhook_worker_enabled is True
    assert settings.webhook_worker_poll_interval_seconds == 3


def test_webhook_worker_rejects_invalid_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS",
        "0",
    )

    with pytest.raises(
        ValidationError,
        match="CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS",
    ):
        Settings(_env_file=None)
