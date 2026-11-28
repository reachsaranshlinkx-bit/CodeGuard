import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal[
    "development",
    "testing",
    "staging",
    "production",
]

LogLevel = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


class Settings(BaseSettings):
    """Centralized application configuration."""

    app_name: str = "CodeGuard API"
    app_version: str = "0.1.0"
    app_description: str = "AI-powered GitHub Pull Request Reviewer"

    environment: Environment = "development"
    debug: bool = False
    log_level: LogLevel = "INFO"
    ai_provider: Literal["disabled", "mock", "openai"] = "disabled"

    github_token: SecretStr | None = None
    api_key: SecretStr | None = None
    database_url: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_request_timeout_seconds: float = 30.0
    webhook_database_path: Path = Path("data/webhook_deliveries.sqlite3")
    github_webhook_secret: SecretStr | None = None
    enable_review_publishing: bool = False
    webhook_worker_enabled: bool = False
    webhook_worker_poll_interval_seconds: int = 1
    webhook_worker_max_attempts: int = 5
    webhook_worker_retry_delay_seconds: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CODEGUARD_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        """Validate the PostgreSQL connection URL without exposing credentials."""

        if value is None:
            return None

        normalized = value.get_secret_value().strip()

        if not normalized:
            raise ValueError("CODEGUARD_DATABASE_URL cannot be empty.")

        if not normalized.startswith("postgresql+psycopg://"):
            raise ValueError(
                "CODEGUARD_DATABASE_URL must use the postgresql+psycopg driver."
            )

        if "\r" in normalized or "\n" in normalized:
            raise ValueError("CODEGUARD_DATABASE_URL contains invalid characters.")

        return SecretStr(normalized)

    @field_validator(
        "openai_request_timeout_seconds",
        "webhook_worker_max_attempts",
        "webhook_worker_retry_delay_seconds",
        mode="before",
    )
    @classmethod
    def reject_boolean_retry_configuration(cls, value: object) -> object:
        """Do not accept booleans as numeric timeout or retry controls."""

        if isinstance(value, bool):
            raise ValueError("Numeric timeout and retry settings cannot be booleans.")
        return value

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce required configuration in production."""
        if self.webhook_worker_poll_interval_seconds < 1:
            raise ValueError(
                "CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS must be at least 1."
            )

        try:
            timeout_is_finite = math.isfinite(self.openai_request_timeout_seconds)
        except OverflowError:
            timeout_is_finite = False
        if not timeout_is_finite or self.openai_request_timeout_seconds <= 0:
            raise ValueError(
                "CODEGUARD_OPENAI_REQUEST_TIMEOUT_SECONDS must be finite "
                "and greater than zero."
            )

        if self.webhook_worker_max_attempts < 1:
            raise ValueError(
                "CODEGUARD_WEBHOOK_WORKER_MAX_ATTEMPTS must be at least 1."
            )

        if self.webhook_worker_retry_delay_seconds < 0:
            raise ValueError(
                "CODEGUARD_WEBHOOK_WORKER_RETRY_DELAY_SECONDS must be non-negative."
            )

        if self.ai_provider == "mock" and self.environment not in (
            "development",
            "testing",
        ):
            raise ValueError(
                "CODEGUARD_AI_PROVIDER=mock is only permitted in development "
                "and testing."
            )

        if self.ai_provider == "openai":
            openai_api_key = (
                self.openai_api_key.get_secret_value()
                if self.openai_api_key is not None
                else None
            )
            if (
                openai_api_key is None
                or not openai_api_key.strip()
                or "\r" in openai_api_key
                or "\n" in openai_api_key
            ):
                raise ValueError(
                    "CODEGUARD_OPENAI_API_KEY must be configured as a "
                    "non-empty valid secret."
                )

            if self.openai_model is None or not self.openai_model.strip():
                raise ValueError(
                    "CODEGUARD_OPENAI_MODEL must be configured as a non-empty "
                    "model identity."
                )

            normalized_model = self.openai_model.strip()
            if any(not character.isprintable() for character in normalized_model):
                raise ValueError("CODEGUARD_OPENAI_MODEL contains invalid characters.")
            object.__setattr__(self, "openai_model", normalized_model)

        if self.environment != "production":
            return self

        if self.debug:
            raise ValueError("CODEGUARD_DEBUG must be false in production.")

        required_secrets = {
            "CODEGUARD_API_KEY": self.api_key,
            "CODEGUARD_GITHUB_TOKEN": self.github_token,
            "CODEGUARD_GITHUB_WEBHOOK_SECRET": self.github_webhook_secret,
        }

        for variable_name, secret in required_secrets.items():
            if secret is None or not secret.get_secret_value().strip():
                raise ValueError(f"{variable_name} must be configured in production.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
