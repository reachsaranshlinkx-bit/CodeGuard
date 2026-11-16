from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

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
