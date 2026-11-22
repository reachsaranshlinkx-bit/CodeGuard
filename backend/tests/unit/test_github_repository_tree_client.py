import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import GitHubResponseError

TREE_SHA = "a" * 40

TREE_RESPONSE = {
    "sha": TREE_SHA,
    "truncated": False,
    "tree": [
        {
            "path": "src",
            "mode": "040000",
            "type": "tree",
            "sha": "b" * 40,
        },
        {
            "path": "src/example.py",
            "mode": "100644",
            "type": "blob",
            "sha": "c" * 40,
            "size": 128,
        },
    ],
}


async def test_get_repository_tree_returns_validated_tree() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=TREE_RESPONSE,
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.get_repository_tree(
            "example",
            "project",
            TREE_SHA,
        )

    assert result.sha == TREE_SHA
    assert result.truncated is False
    assert len(result.tree) == 2
    assert result.tree[1].path == "src/example.py"

    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert request.method == "GET"
    assert request.url.path == (f"/repos/example/project/git/trees/{TREE_SHA}")
    assert request.url.params["recursive"] == "1"


async def test_get_repository_tree_can_disable_recursion() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=TREE_RESPONSE,
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        await client.get_repository_tree(
            "example",
            "project",
            TREE_SHA,
            recursive=False,
        )

    assert len(captured_requests) == 1
    assert "recursive" not in captured_requests[0].url.params


async def test_get_repository_tree_rejects_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sha": TREE_SHA,
                "truncated": False,
                "tree": [{"invalid": "entry"}],
            },
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_repository_tree(
                "example",
                "project",
                TREE_SHA,
            )


async def test_get_repository_tree_rejects_unsafe_tree_sha() -> None:
    async with GitHubClient(max_retries=0) as client:
        with pytest.raises(ValueError):
            await client.get_repository_tree(
                "example",
                "project",
                "../unsafe",
            )
