from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
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

    github_token: SecretStr | None = None
    api_key: SecretStr | None = None
    webhook_database_path: Path = Path("data/webhook_deliveries.sqlite3")
    github_webhook_secret: SecretStr | None = None
    enable_review_publishing: bool = False
    webhook_worker_enabled: bool = False
    webhook_worker_poll_interval_seconds: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CODEGUARD_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce required configuration in production."""
        if self.webhook_worker_poll_interval_seconds < 1:
            raise ValueError(
                "CODEGUARD_WEBHOOK_WORKER_POLL_INTERVAL_SECONDS must be at least 1."
            )

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
