import math


def _validate_status_code(status_code: int) -> None:
    if (
        not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or not 100 <= status_code <= 599
    ):
        raise ValueError("HTTP status code must be an integer from 100 to 599.")


def _validate_request_id(request_id: str | None) -> None:
    if request_id is not None and (
        not isinstance(request_id, str) or not request_id.strip()
    ):
        raise ValueError("Request ID must be a non-empty string when provided.")


class AIProviderError(Exception):
    """Base exception for AI provider integration errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AIProviderConfigurationError(AIProviderError):
    """Raised when AI provider configuration is invalid."""


class AIProviderRequestError(AIProviderError):
    """Raised when an AI provider request fails."""


class AIProviderTimeoutError(AIProviderRequestError):
    """Raised when an AI provider request times out."""


class AIProviderHTTPError(AIProviderRequestError):
    """Raised when an AI provider returns an unsuccessful HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_id: str | None = None,
    ) -> None:
        _validate_status_code(status_code)
        _validate_request_id(request_id)

        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class AIProviderAuthenticationError(AIProviderHTTPError):
    """Raised when AI provider authentication fails."""


class AIProviderPermissionError(AIProviderHTTPError):
    """Raised when the authenticated identity lacks provider permission."""


class AIProviderRateLimitError(AIProviderHTTPError):
    """Raised when the AI provider rate limits a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        if retry_after is not None:
            if isinstance(retry_after, bool) or not isinstance(
                retry_after, (int, float)
            ):
                raise ValueError("Retry delay must be a finite non-negative number.")

            try:
                is_finite = math.isfinite(retry_after)
            except OverflowError:
                is_finite = False

            if not is_finite or retry_after < 0:
                raise ValueError("Retry delay must be a finite non-negative number.")

        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
        )
        self.retry_after = float(retry_after) if retry_after is not None else None


class AIProviderServerError(AIProviderHTTPError):
    """Raised when the AI provider returns a server-side error."""


class AIProviderResponseError(AIProviderError):
    """Raised when an AI provider response is invalid or unexpected."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        if status_code is not None:
            _validate_status_code(status_code)
        _validate_request_id(request_id)

        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
