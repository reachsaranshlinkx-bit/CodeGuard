import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import (
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubServerError,
)


async def test_network_error_is_retried() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise httpx.ConnectError(
                "Temporary connection failure",
                request=request,
            )

        return httpx.Response(
            200,
            json={"ok": True},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=1,
        sleep=fake_sleep,
    ) as client:
        result = await client.get_json("/test")

    assert result == {"ok": True}
    assert attempts == 2
    assert delays == [0.5]


async def test_server_error_is_retried() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                500,
                json={"message": "Temporary server error"},
                request=request,
            )

        return httpx.Response(
            200,
            json={"ok": True},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=1,
        sleep=fake_sleep,
    ) as client:
        result = await client.get_json("/test")

    assert result == {"ok": True}
    assert attempts == 2
    assert delays == [0.5]


async def test_server_error_stops_after_max_retries() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            500,
            json={"message": "Server unavailable"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        sleep=fake_sleep,
    ) as client:
        with pytest.raises(GitHubServerError):
            await client.get_json("/test")

    assert attempts == 3
    assert delays == [0.5, 1.0]


async def test_rate_limit_respects_retry_after() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                429,
                json={"message": "API rate limit exceeded"},
                headers={"Retry-After": "5"},
                request=request,
            )

        return httpx.Response(
            200,
            json={"ok": True},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=1,
        sleep=fake_sleep,
    ) as client:
        result = await client.get_json("/test")

    assert result == {"ok": True}
    assert attempts == 2
    assert delays == [5.0]


async def test_secondary_rate_limit_uses_minimum_wait() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                403,
                json={"message": "You have exceeded a secondary rate limit."},
                request=request,
            )

        return httpx.Response(
            200,
            json={"ok": True},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=1,
        max_retry_delay=60.0,
        sleep=fake_sleep,
    ) as client:
        result = await client.get_json("/test")

    assert result == {"ok": True}
    assert attempts == 2
    assert delays == [60.0]


async def test_long_rate_limit_delay_is_not_retried_early() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            429,
            json={"message": "API rate limit exceeded"},
            headers={"Retry-After": "300"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        max_retry_delay=120.0,
        sleep=fake_sleep,
    ) as client:
        with pytest.raises(GitHubRateLimitError):
            await client.get_json("/test")

    assert attempts == 1
    assert delays == []


async def test_not_found_error_is_not_retried() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            404,
            json={"message": "Not found"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=3,
        sleep=fake_sleep,
    ) as client:
        with pytest.raises(GitHubNotFoundError):
            await client.get_json("/test")

    assert attempts == 1
    assert delays == []


@pytest.mark.parametrize(
    "max_retries",
    [-1, 6, True],
)
async def test_invalid_max_retries_is_rejected(
    max_retries: int,
) -> None:
    with pytest.raises(ValueError):
        GitHubClient(max_retries=max_retries)


@pytest.mark.parametrize(
    "max_retry_delay",
    [0, -1, float("inf"), float("nan"), True],
)
async def test_invalid_max_retry_delay_is_rejected(
    max_retry_delay: float,
) -> None:
    with pytest.raises(ValueError):
        GitHubClient(max_retry_delay=max_retry_delay)


async def test_total_retry_wait_budget_is_enforced() -> None:
    attempts = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            403,
            json={"message": "You have exceeded a secondary rate limit."},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        max_retry_delay=120.0,
        sleep=fake_sleep,
    ) as client:
        with pytest.raises(GitHubRateLimitError):
            await client.get_json("/test")

    assert attempts == 2
    assert delays == [60.0]
    assert sum(delays) <= 120.0
