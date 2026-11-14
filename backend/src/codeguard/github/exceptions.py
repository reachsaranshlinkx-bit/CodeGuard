class GitHubError(Exception):
    """Base exception for all CodeGuard GitHub integration errors."""


class GitHubRequestError(GitHubError):
    """Raised when a GitHub HTTP request fails at the transport level."""


class GitHubAPIError(GitHubError):
    """Raised when GitHub returns an unsuccessful API response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_url = request_url


class GitHubAuthenticationError(GitHubAPIError):
    """Raised when GitHub authentication fails."""


class GitHubPermissionError(GitHubAPIError):
    """Raised when the authenticated user lacks permission."""


class GitHubNotFoundError(GitHubAPIError):
    """Raised when the requested GitHub resource does not exist."""


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub rate limits an API request."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status_code: int = 429,
        request_url: str | None = None,
    ) -> None:
        if status_code not in (403, 429):
            raise ValueError("Rate limit status must be HTTP 403 or 429.")

        super().__init__(
            message,
            status_code=status_code,
            request_url=request_url,
        )
        self.retry_after = retry_after


class GitHubServerError(GitHubAPIError):
    """Raised when GitHub returns a server-side error."""


class GitHubResponseError(GitHubError):
    """Raised when GitHub returns an invalid or unexpected response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_url = request_url
