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
