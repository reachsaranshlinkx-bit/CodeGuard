import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import GitHubResponseError

COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40

COMMIT_RESPONSE = {
    "sha": COMMIT_SHA,
    "tree": {
        "sha": TREE_SHA,
        "url": (f"https://api.github.com/repos/octocat/CodeGuard/git/trees/{TREE_SHA}"),
    },
}


async def test_get_git_commit_returns_validated_commit() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=COMMIT_RESPONSE,
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.get_git_commit(
            "octocat",
            "CodeGuard",
            COMMIT_SHA,
        )

    assert result.sha == COMMIT_SHA
    assert result.tree.sha == TREE_SHA

    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert request.method == "GET"
    assert request.url.path == (f"/repos/octocat/CodeGuard/git/commits/{COMMIT_SHA}")


async def test_get_git_commit_rejects_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sha": COMMIT_SHA,
                "tree": {"invalid": "tree"},
            },
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_git_commit(
                "octocat",
                "CodeGuard",
                COMMIT_SHA,
            )


@pytest.mark.parametrize(
    "commit_sha",
    [
        "",
        " ",
        "../unsafe",
        "invalid/sha",
    ],
)
async def test_get_git_commit_rejects_invalid_sha(
    commit_sha: str,
) -> None:
    async with GitHubClient(max_retries=0) as client:
        with pytest.raises(ValueError):
            await client.get_git_commit(
                "octocat",
                "CodeGuard",
                commit_sha,
            )
