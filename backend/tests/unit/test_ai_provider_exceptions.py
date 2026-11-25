import pytest

from codeguard.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderConfigurationError,
    AIProviderError,
    AIProviderHTTPError,
    AIProviderPermissionError,
    AIProviderRateLimitError,
    AIProviderRequestError,
    AIProviderResponseError,
    AIProviderServerError,
    AIProviderTimeoutError,
)


def test_all_provider_exceptions_inherit_from_base_error() -> None:
    exceptions = (
        AIProviderConfigurationError("configuration failed"),
        AIProviderRequestError("request failed"),
        AIProviderTimeoutError("request timed out"),
        AIProviderHTTPError("HTTP request failed", status_code=500),
        AIProviderAuthenticationError("authentication failed", status_code=401),
        AIProviderPermissionError("permission denied", status_code=403),
        AIProviderRateLimitError("rate limited", status_code=429),
        AIProviderServerError("provider failed", status_code=503),
        AIProviderResponseError("invalid response"),
    )

    assert all(isinstance(error, AIProviderError) for error in exceptions)
    assert issubclass(AIProviderTimeoutError, AIProviderRequestError)
    assert issubclass(AIProviderHTTPError, AIProviderRequestError)
    assert issubclass(AIProviderAuthenticationError, AIProviderHTTPError)
    assert issubclass(AIProviderPermissionError, AIProviderHTTPError)
    assert issubclass(AIProviderRateLimitError, AIProviderHTTPError)
    assert issubclass(AIProviderServerError, AIProviderHTTPError)


def test_http_error_preserves_status_and_request_id_without_exposing_metadata() -> None:
    error = AIProviderHTTPError(
        "Provider request failed.",
        status_code=502,
        request_id="provider-request-123",
    )

    assert error.status_code == 502
    assert error.request_id == "provider-request-123"
    assert str(error) == "Provider request failed."


def test_authentication_and_permission_errors_are_classified() -> None:
    authentication_error = AIProviderAuthenticationError(
        "Authentication rejected.", status_code=401
    )
    permission_error = AIProviderPermissionError("Permission denied.", status_code=403)

    assert isinstance(authentication_error, AIProviderHTTPError)
    assert isinstance(permission_error, AIProviderHTTPError)
    assert authentication_error.status_code == 401
    assert permission_error.status_code == 403


def test_rate_limit_error_preserves_retry_after() -> None:
    error = AIProviderRateLimitError(
        "Provider rate limit reached.",
        status_code=429,
        retry_after=2.5,
    )

    assert error.retry_after == 2.5
    assert error.status_code == 429


def test_timeout_is_distinct_from_other_transport_errors() -> None:
    timeout_error = AIProviderTimeoutError("Provider request timed out.")
    request_error = AIProviderRequestError("Provider connection failed.")

    assert isinstance(timeout_error, AIProviderRequestError)
    assert isinstance(timeout_error, AIProviderError)
    assert not isinstance(request_error, AIProviderTimeoutError)


def test_response_error_can_preserve_optional_http_status() -> None:
    without_status = AIProviderResponseError("Provider response was invalid.")
    with_status = AIProviderResponseError(
        "Provider response was invalid.",
        status_code=200,
    )

    assert without_status.status_code is None
    assert with_status.status_code == 200
    assert str(with_status) == "Provider response was invalid."


@pytest.mark.parametrize("status_code", [99, 600, True, "500", None])
def test_http_errors_reject_invalid_status_metadata(status_code: object) -> None:
    with pytest.raises(ValueError):
        AIProviderHTTPError("Invalid status.", status_code=status_code)  # type: ignore[arg-type]


@pytest.mark.parametrize("request_id", ["", "   ", 123])
def test_http_errors_reject_invalid_request_ids(request_id: object) -> None:
    with pytest.raises(ValueError):
        AIProviderHTTPError(
            "Invalid request ID.",
            status_code=500,
            request_id=request_id,  # type: ignore[arg-type]
        )


def test_rate_limit_errors_reject_extremely_large_integer_retry_after() -> None:
    with pytest.raises(ValueError):
        AIProviderRateLimitError(
            "Invalid retry delay.",
            status_code=429,
            retry_after=10**1000,
        )


@pytest.mark.parametrize("retry_after", [1, 0.5, 0])
def test_rate_limit_errors_preserve_valid_retry_after(retry_after: int | float) -> None:
    error = AIProviderRateLimitError(
        "Retry delay is valid.",
        status_code=429,
        retry_after=retry_after,
    )

    assert error.retry_after == retry_after


@pytest.mark.parametrize(
    "retry_after",
    [-1, float("inf"), float("-inf"), float("nan"), True, "1.5", object()],
)
def test_rate_limit_errors_reject_invalid_retry_after_values(
    retry_after: object,
) -> None:
    with pytest.raises(ValueError):
        AIProviderRateLimitError(
            "Invalid retry delay.",
            status_code=429,
            retry_after=retry_after,  # type: ignore[arg-type]
        )


def test_response_errors_reject_invalid_status_metadata() -> None:
    with pytest.raises(ValueError):
        AIProviderResponseError("Invalid response.", status_code=99)
