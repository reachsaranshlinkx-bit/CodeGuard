import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from codeguard.core.config import Settings, get_settings
from codeguard.github.client import GitHubClient
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer


async def get_github_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[GitHubClient]:
    """Provide a GitHub client and manage its HTTP lifecycle."""

    token = (
        settings.github_token.get_secret_value()
        if settings.github_token is not None
        else None
    )

    client = GitHubClient(token=token)

    try:
        yield client
    finally:
        await client.aclose()


def get_pull_request_analyzer(
    github_client: Annotated[
        GitHubClient,
        Depends(get_github_client),
    ],
) -> PullRequestAnalyzer:
    """Provide a pull request analyzer using the injected GitHub client."""

    return PullRequestAnalyzer(github_client=github_client)


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    provided_api_key: Annotated[
        str | None,
        Header(alias="X-CodeGuard-API-Key"),
    ] = None,
) -> None:
    """Reject requests without the configured CodeGuard API key."""

    configured_api_key = (
        settings.api_key.get_secret_value() if settings.api_key is not None else ""
    )

    # Fail closed when authentication is not configured.
    if not configured_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    # Reject absent or incorrect credentials.
    if provided_api_key is None or not secrets.compare_digest(
        configured_api_key.encode("utf-8"),
        provided_api_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
