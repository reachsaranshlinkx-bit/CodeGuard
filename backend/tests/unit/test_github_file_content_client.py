import base64

import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import (
    GitHubNotFoundError,
    GitHubResponseError,
)

FILE_SOURCE = "print('hello')\n"
FILE_CONTENT_BASE64 = base64.b64encode(FILE_SOURCE.encode("utf-8")).decode("ascii")

FILE_SHA = "a" * 40

FILE_RESPONSE = {
    "type": "file",
    "encoding": "base64",
    "path": "src/example.py",
    "sha": FILE_SHA,
    "size": len(FILE_SOURCE.encode("utf-8")),
    "content": FILE_CONTENT_BASE64,
}


async def test_get_file_content_returns_decoded_text() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json=FILE_RESPONSE,
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        content = await client.get_file_content(
            "octocat",
            "CodeGuard",
            "src/example.py",
            FILE_SHA,
        )

    assert content == FILE_SOURCE
    assert len(captured_requests) == 1
    assert captured_requests[0].url.path == (
        "/repos/octocat/CodeGuard/contents/src/example.py"
    )
    assert captured_requests[0].url.params["ref"] == FILE_SHA


async def test_get_file_content_handles_base64_line_breaks() -> None:
    payload = {
        **FILE_RESPONSE,
        "content": (FILE_CONTENT_BASE64[:8] + "\n" + FILE_CONTENT_BASE64[8:]),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        content = await client.get_file_content(
            "octocat", "CodeGuard", "src/example.py", FILE_SHA
        )

    assert content == FILE_SOURCE


@pytest.mark.parametrize(
    ("owner", "repo", "path", "ref"),
    [
        ("", "CodeGuard", "src/example.py", FILE_SHA),
        ("octocat", "", "src/example.py", FILE_SHA),
        ("octocat", "CodeGuard", "", FILE_SHA),
        ("octocat", "CodeGuard", "src/example.py", ""),
    ],
)
async def test_get_file_content_rejects_empty_identifiers(
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> None:
    async with GitHubClient() as client:
        with pytest.raises(ValueError):
            await client.get_file_content(owner, repo, path, ref)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.py",
        "../secret.py",
        "src/../secret.py",
        r"src\secret.py",
    ],
)
async def test_get_file_content_rejects_unsafe_paths(
    path: str,
) -> None:
    async with GitHubClient() as client:
        with pytest.raises(ValueError):
            await client.get_file_content("octocat", "CodeGuard", path, FILE_SHA)


async def test_get_file_content_rejects_invalid_base64() -> None:
    payload = {
        **FILE_RESPONSE,
        "content": "not base64!",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_file_content(
                "octocat", "CodeGuard", "src/example.py", FILE_SHA
            )


async def test_get_file_content_rejects_binary_content() -> None:
    payload = {
        **FILE_RESPONSE,
        "content": base64.b64encode(b"\xff\xfe").decode("ascii"),
        "size": 2,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_file_content(
                "octocat", "CodeGuard", "src/example.py", FILE_SHA
            )


async def test_get_file_content_rejects_size_mismatch() -> None:
    payload = {
        **FILE_RESPONSE,
        "size": FILE_RESPONSE["size"] + 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_file_content(
                "octocat", "CodeGuard", "src/example.py", FILE_SHA
            )


async def test_get_file_content_rejects_unsupported_encoding() -> None:
    payload = {
        **FILE_RESPONSE,
        "encoding": "none",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_file_content(
                "octocat", "CodeGuard", "src/example.py", FILE_SHA
            )


async def test_get_file_content_propagates_not_found_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"message": "Not Found"},
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubNotFoundError):
            await client.get_file_content(
                "octocat", "CodeGuard", "missing.py", FILE_SHA
            )


async def test_get_file_content_rejects_oversized_file() -> None:
    payload = {
        **FILE_RESPONSE,
        "size": 1_000_001,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=payload,
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_file_content(
                "octocat",
                "CodeGuard",
                "src/example.py",
                FILE_SHA,
            )
