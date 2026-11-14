import pytest

from codeguard.github.exceptions import (
    GitHubAPIError,
    GitHubError,
    GitHubRateLimitError,
    GitHubRequestError,
)


def test_api_error_preserves_status_and_url() -> None:
    error = GitHubAPIError(
        "Request failed",
        status_code=500,
        request_url="https://api.github.com/repos/example/project",
    )

    assert error.status_code == 500
    assert error.request_url == ("https://api.github.com/repos/example/project")
    assert str(error) == "Request failed"


def test_rate_limit_error_preserves_retry_after() -> None:
    error = GitHubRateLimitError(
        "Rate limit exceeded",
        retry_after=60.0,
    )

    assert error.status_code == 429
    assert error.retry_after == 60.0


def test_rate_limit_error_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="HTTP 403 or 429"):
        GitHubRateLimitError(
            "Invalid rate limit status",
            status_code=500,
        )


def test_github_exceptions_share_base_class() -> None:
    assert issubclass(GitHubAPIError, GitHubError)
    assert issubclass(GitHubRequestError, GitHubError)
