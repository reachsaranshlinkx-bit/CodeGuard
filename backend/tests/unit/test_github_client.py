import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import (
    GitHubAuthenticationError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubRequestError,
    GitHubResponseError,
    GitHubServerError,
)


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (401, GitHubAuthenticationError),
        (403, GitHubPermissionError),
        (404, GitHubNotFoundError),
        (500, GitHubServerError),
    ],
)
async def test_http_errors_are_mapped(
    status_code: int,
    expected_exception: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"message": "GitHub request failed"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(expected_exception):
            await client.get_json("/repos/example/project")


async def test_successful_request_returns_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": 123, "name": "CodeGuard"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.get_json("/repos/example/project")

    assert result == {"id": 123, "name": "CodeGuard"}


async def test_client_sends_required_headers() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json={"ok": True},
            request=request,
        )

    async with GitHubClient(
        token="test-token",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        await client.get_json("/test")

    assert len(captured_requests) == 1

    headers = captured_requests[0].headers

    assert headers["Accept"] == "application/vnd.github+json"
    assert headers["X-GitHub-Api-Version"] == "2026-03-10"
    assert headers["User-Agent"] == "CodeGuard/0.1.0"
    assert headers["Authorization"] == "Bearer test-token"


async def test_rate_limit_response_preserves_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"message": "API rate limit exceeded"},
            headers={"Retry-After": "30"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubRateLimitError) as exc_info:
            await client.get_json("/repos/example/project")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 30.0


async def test_403_rate_limit_is_detected_from_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API limit exceeded"},
            headers={"X-RateLimit-Remaining": "0"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubRateLimitError):
            await client.get_json("/repos/example/project")


async def test_network_error_is_converted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection failed",
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubRequestError):
            await client.get_json("/repos/example/project")


async def test_invalid_json_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-valid-json",
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError) as exc_info:
            await client.get_json("/repos/example/project")

    assert exc_info.value.status_code == 200


async def test_full_url_is_rejected() -> None:
    async with GitHubClient(max_retries=0) as client:
        with pytest.raises(ValueError):
            await client.get_json("https://example.com/malicious")
