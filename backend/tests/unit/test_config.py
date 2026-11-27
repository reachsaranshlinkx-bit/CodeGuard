import pytest
from pydantic import ValidationError

from codeguard.core.config import Settings


def test_settings_use_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEGUARD_AI_PROVIDER", raising=False)
    monkeypatch.delenv("CODEGUARD_OPENAI_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("CODEGUARD_WEBHOOK_WORKER_RETRY_DELAY_SECONDS", raising=False)
    settings = Settings(_env_file=None)

    assert settings.app_name == "CodeGuard API"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.ai_provider == "disabled"
    assert settings.openai_request_timeout_seconds == 30.0
    assert settings.webhook_worker_max_attempts == 5
    assert settings.webhook_worker_retry_delay_seconds == 30


@pytest.mark.parametrize("provider", ["disabled", "mock"])
def test_ai_provider_is_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "development")
    monkeypatch.setenv("CODEGUARD_AI_PROVIDER", provider)

    settings = Settings(_env_file=None)

    assert settings.ai_provider == provider


def test_openai_provider_is_loaded_with_required_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEGUARD_ENVIRONMENT", "testing")
    monkeypatch.setenv("CODEGUARD_AI_PROVIDER", "openai")
    monkeypatch.setenv("CODEGUARD_OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("CODEGUARD_OPENAI_MODEL", "gpt-test-model")

    settings = Settings(_env_file=None)

    assert settings.ai_provider == "openai"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-openai-secret"
    assert str(settings.openai_api_key) == "**********"
    assert "test-openai-secret" not in repr(settings)
    assert settings.openai_model == "gpt-test-model"


def test_openai_credentials_do_not_enable_provider_by_themselves() -> None:
    settings = Settings(
        _env_file=None,
        environment="testing",
        openai_api_key="test-openai-secret",
        openai_model="gpt-test-model",
    )

    assert settings.ai_provider == "disabled"


def test_openai_request_timeout_is_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEGUARD_OPENAI_REQUEST_TIMEOUT_SECONDS", "7.5")

    settings = Settings(_env_file=None)

    assert settings.openai_request_timeout_seconds == 7.5


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_openai_request_timeout_rejects_nonpositive_or_nonfinite_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("CODEGUARD_OPENAI_REQUEST_TIMEOUT_SECONDS", value)

    with pytest.raises(
        ValidationError,
        match="CODEGUARD_OPENAI_REQUEST_TIMEOUT_SECONDS",
    ):
        Settings(_env_file=None)


def test_openai_request_timeout_rejects_boolean_values() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_request_timeout_seconds=True,  # type: ignore[arg-type]
        )


def test_ai_provider_rejects_unsupported_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEGUARD_AI_PROVIDER", "unsupported")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("openai_api_key", "openai_model", "expected_error"),
    [
        (None, "gpt-test-model", "CODEGUARD_OPENAI_API_KEY"),
        ("   ", "gpt-test-model", "CODEGUARD_OPENAI_API_KEY"),
        ("invalid\r\nheader", "gpt-test-model", "CODEGUARD_OPENAI_API_KEY"),
        ("test-openai-secret", None, "CODEGUARD_OPENAI_MODEL"),
        ("test-openai-secret", "   ", "CODEGUARD_OPENAI_MODEL"),
        ("test-openai-secret", "gpt-\x00model", "CODEGUARD_OPENAI_MODEL"),
    ],
)
def test_openai_provider_rejects_invalid_configuration_without_secret_leaks(
    openai_api_key: str | None,
    openai_model: str | None,
    expected_error: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            environment="testing",
            ai_provider="openai",
            openai_api_key=openai_api_key,
            openai_model=openai_model,
        )

    error_text = str(exc_info.value)
    assert expected_error in error_text
    assert "test-openai-secret" not in error_text


@pytest.mark.parametrize("environment", ["development", "testing"])
def test_mock_ai_provider_is_allowed_in_nonproduction_environments(
    environment: str,
) -> None:
    settings = Settings(
        _env_file=None,
        environment=environment,
        ai_provider="mock",
    )

    assert settings.ai_provider == "mock"


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_mock_ai_provider_is_rejected_outside_development_and_testing(
    environment: str,
) -> None:
    with pytest.raises(ValidationError, match="mock"):
        Settings(
            _env_file=None,
            environment=environment,
            ai_provider="mock",
            debug=False,
            api_key="test-codeguard-api-key",
            github_token="test-github-token",
            github_webhook_secret="test-webhook-secret",
        )


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


def test_codeguard_api_key_does_not_select_ai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEGUARD_AI_PROVIDER", raising=False)
    monkeypatch.setenv("CODEGUARD_API_KEY", "test-codeguard-api-key")

    settings = Settings(_env_file=None)

    assert settings.ai_provider == "disabled"
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "test-codeguard-api-key"


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
    monkeypatch.delenv("CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("CODEGUARD_WEBHOOK_WORKER_RETRY_DELAY_SECONDS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.webhook_worker_enabled is False
    assert settings.webhook_worker_poll_interval_seconds == 1
    assert settings.webhook_worker_max_attempts == 5
    assert settings.webhook_worker_retry_delay_seconds == 30


def test_webhook_worker_settings_load_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEGUARD_WEBHOOK_WORKER_ENABLED", "true")
    monkeypatch.setenv(
        "CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS",
        "3",
    )
    monkeypatch.setenv("CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("CODEGUARD_WEBHOOK_WORKER_RETRY_DELAY_SECONDS", "12")

    settings = Settings(_env_file=None)

    assert settings.webhook_worker_enabled is True
    assert settings.webhook_worker_poll_interval_seconds == 3
    assert settings.webhook_worker_max_attempts == 4
    assert settings.webhook_worker_retry_delay_seconds == 12


@pytest.mark.parametrize(
    ("environment_name", "value", "error_message"),
    [
        (
            "CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS",
            "0",
            "CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS",
        ),
        (
            "CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS",
            "-1",
            "CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS",
        ),
        (
            "CODEGUARD_WEBHOOK_WORKER_RETRY_DELAY_SECONDS",
            "-1",
            "CODEGUARD_WEBHOOK_WORKER_RETRY_DELAY_SECONDS",
        ),
    ],
)
def test_webhook_worker_rejects_invalid_retry_controls(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    value: str,
    error_message: str,
) -> None:
    monkeypatch.setenv(environment_name, value)

    with pytest.raises(ValidationError, match=error_message):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "setting_name",
    [
        "webhook_worker_max_attempts",
        "webhook_worker_retry_delay_seconds",
    ],
)
def test_webhook_worker_retry_controls_reject_boolean_values(
    setting_name: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            **{setting_name: True},
        )


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
