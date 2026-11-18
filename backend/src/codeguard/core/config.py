from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
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
    enable_review_publishing: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CODEGUARD_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
