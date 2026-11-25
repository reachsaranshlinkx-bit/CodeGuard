import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from codeguard.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderConfigurationError,
    AIProviderHTTPError,
    AIProviderPermissionError,
    AIProviderRateLimitError,
    AIProviderRequestError,
    AIProviderResponseError,
    AIProviderServerError,
    AIProviderTimeoutError,
    AIRequestTransmissionState,
)
from codeguard.ai.http_client import AIProviderHTTPClient


async def test_successful_post_returns_json_object() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "review-123", "accepted": True},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.post_json("reviews", {"input": "review this"})

    assert result == {"id": "review-123", "accepted": True}


async def test_post_sends_json_body_and_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async with AIProviderHTTPClient(
        "https://provider.example/v1",
        api_key="test-provider-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.post_json("/reviews", {"model": "test", "input": "payload"})

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert json.loads(request.content) == {"model": "test", "input": "payload"}
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == "Bearer test-provider-key"


async def test_api_key_is_optional() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.post_json("reviews", {})

    assert "Authorization" not in requests[0].headers


def test_timeout_and_redirect_configuration() -> None:
    client = AIProviderHTTPClient(
        "https://provider.example",
        timeout=3.5,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={}, request=request)
        ),
    )

    assert client._client.timeout.connect == 3.5
    assert client._client.timeout.read == 3.5
    assert client._client.follow_redirects is False


@pytest.mark.parametrize(
    "base_url",
    [
        "http://provider.example",
        "ftp://provider.example",
        "https:///missing-host",
        "https://user:password@provider.example",
        "https://provider.example/v1?query=value",
        "https://provider.example/v1#fragment",
        "https://provider.example/v1?",
        "https://provider.example/v1#",
        "https://provider.example:65536",
        "https://provider.example:not-a-port",
        "not-a-url",
        "",
    ],
)
def test_invalid_base_urls_are_rejected(base_url: str) -> None:
    with pytest.raises(AIProviderConfigurationError):
        AIProviderHTTPClient(base_url)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, float("inf"), float("-inf"), float("nan"), "1", 10**1000],
)
def test_invalid_timeout_values_are_rejected(timeout: object) -> None:
    with pytest.raises(AIProviderConfigurationError):
        AIProviderHTTPClient(
            "https://provider.example",
            timeout=timeout,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("api_key", ["", "   ", 123])
def test_invalid_api_key_values_are_rejected(api_key: object) -> None:
    with pytest.raises(AIProviderConfigurationError):
        AIProviderHTTPClient(
            "https://provider.example",
            api_key=api_key,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "https://attacker.example/steal",
        "//attacker.example/steal",
        "../steal",
        "/../steal",
        r"\attacker.example\steal",
        "reviews?redirect=https://attacker.example",
        "reviews#fragment",
    ],
)
async def test_unsafe_paths_are_rejected_without_sending_requests(path: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ValueError):
            await client.post_json(path, {})

    assert requests == []


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, AIProviderAuthenticationError),
        (403, AIProviderPermissionError),
        (429, AIProviderRateLimitError),
        (500, AIProviderServerError),
        (503, AIProviderServerError),
        (400, AIProviderHTTPError),
    ],
)
async def test_http_statuses_map_to_provider_errors(
    status_code: int,
    expected_error: type[AIProviderHTTPError],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": "sensitive provider response body"},
            headers={"x-request-id": "provider-request-456"},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(expected_error) as exc_info:
            await client.post_json("reviews", {})

    error = exc_info.value
    assert error.status_code == status_code
    assert error.request_id == "provider-request-456"
    assert "sensitive provider response body" not in str(error)


async def test_timeout_errors_are_mapped_without_exposing_transport_details() -> None:
    secret = "sensitive-timeout-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderTimeoutError) as exc_info:
            await client.post_json("reviews", {})

    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    ("exception_factory", "expected_state", "expected_error"),
    [
        (
            lambda request: httpx.PoolTimeout("pool timeout", request=request),
            AIRequestTransmissionState.NOT_SENT,
            AIProviderTimeoutError,
        ),
        (
            lambda request: httpx.ConnectTimeout("connect timeout", request=request),
            AIRequestTransmissionState.NOT_SENT,
            AIProviderTimeoutError,
        ),
        (
            lambda request: httpx.ConnectError("connect failure", request=request),
            AIRequestTransmissionState.NOT_SENT,
            AIProviderRequestError,
        ),
        (
            lambda request: httpx.ReadTimeout("read timeout", request=request),
            AIRequestTransmissionState.UNKNOWN,
            AIProviderTimeoutError,
        ),
        (
            lambda request: httpx.WriteTimeout("write timeout", request=request),
            AIRequestTransmissionState.UNKNOWN,
            AIProviderTimeoutError,
        ),
        (
            lambda request: httpx.TimeoutException("generic timeout", request=request),
            AIRequestTransmissionState.UNKNOWN,
            AIProviderTimeoutError,
        ),
        (
            lambda request: httpx.RequestError("other request error", request=request),
            AIRequestTransmissionState.UNKNOWN,
            AIProviderRequestError,
        ),
    ],
)
async def test_transport_failures_preserve_conservative_transmission_state(
    exception_factory: Callable[[httpx.Request], Exception],
    expected_state: AIRequestTransmissionState,
    expected_error: type[BaseException],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_factory(request)

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(expected_error) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.transmission_state == expected_state


async def test_cancellation_is_not_converted_to_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post_json("reviews", {})


async def test_other_network_errors_are_mapped_without_details() -> None:
    secret = "sensitive-network-detail"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError(secret, request=request)

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderRequestError) as exc_info:
            await client.post_json("reviews", {})

    assert not isinstance(exc_info.value, AIProviderTimeoutError)
    assert secret not in str(exc_info.value)
    assert len(requests) == 1


async def test_rate_limit_error_extracts_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "2.5"},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderRateLimitError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.retry_after == 2.5


@pytest.mark.parametrize("retry_after", ["invalid", "-1", "nan", "inf", ""])
async def test_invalid_retry_after_headers_do_not_break_error_mapping(
    retry_after: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": retry_after},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderRateLimitError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.retry_after is None


async def test_invalid_utf8_json_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"\xff",
            headers={
                "Content-Type": "application/json",
                "x-request-id": "invalid-utf8-request",
            },
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderResponseError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.request_id == "invalid-utf8-request"


async def test_invalid_json_is_rejected_without_exposing_body() -> None:
    secret = "sensitive-raw-provider-response"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=secret.encode(),
            headers={
                "Content-Type": "application/json",
                "x-request-id": "invalid-json-request",
            },
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderResponseError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.status_code == 200
    assert exc_info.value.request_id == "invalid-json-request"
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize("payload", [[], "text", 7, None])
async def test_non_object_json_responses_are_rejected(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=payload,
            headers={"x-request-id": "unexpected-structure-request"},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderResponseError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.status_code == 200
    assert exc_info.value.request_id == "unexpected-structure-request"


async def test_response_error_without_request_id_preserves_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderResponseError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.request_id is None


async def test_api_key_is_not_in_client_repr_or_error_messages() -> None:
    api_key = "private-api-key-value"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text=f"response includes {api_key}",
            request=request,
        )

    client = AIProviderHTTPClient(
        "https://provider.example",
        api_key=api_key,
        transport=httpx.MockTransport(handler),
    )

    assert api_key not in repr(client)
    with pytest.raises(AIProviderAuthenticationError) as exc_info:
        await client.post_json("reviews", {})
    assert api_key not in str(exc_info.value)
    await client.aclose()


class CloseTrackingTransport(httpx.MockTransport):
    def __init__(self) -> None:
        super().__init__(
            lambda request: httpx.Response(200, json={"ok": True}, request=request)
        )
        self.was_closed = False

    async def aclose(self) -> None:
        self.was_closed = True
        await super().aclose()


async def test_context_manager_closes_transport() -> None:
    transport = CloseTrackingTransport()
    client = AIProviderHTTPClient(
        "https://provider.example",
        transport=transport,
    )

    async with client:
        assert not transport.was_closed

    assert transport.was_closed


async def test_explicit_cleanup_is_safe_to_repeat() -> None:
    transport = CloseTrackingTransport()
    client = AIProviderHTTPClient(
        "https://provider.example",
        transport=transport,
    )

    await client.aclose()
    await client.aclose()

    assert transport.was_closed


async def test_redirect_responses_are_not_followed() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"Location": "https://attacker.example/collect"},
            request=request,
        )

    async with AIProviderHTTPClient(
        "https://provider.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AIProviderHTTPError) as exc_info:
            await client.post_json("reviews", {})

    assert exc_info.value.status_code == 307
    assert len(requests) == 1
