import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import GitHubResponseError
from codeguard.github.schemas import GitHubPullRequest

PULL_REQUEST_RESPONSE = {
    "url": ("https://api.github.com/repos/octocat/CodeGuard/pulls/42"),
    "html_url": ("https://github.com/octocat/CodeGuard/pull/42"),
    "number": 42,
    "state": "open",
    "title": "Fix authentication",
    "body": "Fix an authentication issue.",
    "draft": False,
    "merged": False,
    "user": {
        "login": "octocat",
        "id": 100,
    },
    "base": {
        "ref": "main",
        "sha": "a" * 40,
    },
    "head": {
        "ref": "feature/authentication",
        "sha": "b" * 40,
    },
}


async def test_get_pull_request_returns_validated_model() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=PULL_REQUEST_RESPONSE,
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        pull_request = await client.get_pull_request(
            "octocat",
            "CodeGuard",
            42,
        )

    assert isinstance(pull_request, GitHubPullRequest)
    assert pull_request.number == 42
    assert pull_request.title == "Fix authentication"
    assert pull_request.state == "open"
    assert pull_request.base.ref == "main"
    assert pull_request.head.ref == "feature/authentication"

    assert captured_requests[0].url.path == ("/repos/octocat/CodeGuard/pulls/42")


@pytest.mark.parametrize(
    "pull_number",
    [0, -1, True],
)
async def test_get_pull_request_rejects_invalid_number(
    pull_number: int,
) -> None:
    async with GitHubClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=PULL_REQUEST_RESPONSE,
                request=request,
            )
        )
    ) as client:
        with pytest.raises(ValueError):
            await client.get_pull_request(
                "octocat",
                "CodeGuard",
                pull_number,
            )


async def test_get_pull_request_rejects_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"number": 42, "title": "Incomplete PR"},
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError) as exc_info:
            await client.get_pull_request(
                "octocat",
                "CodeGuard",
                42,
            )

    assert exc_info.value.status_code == 200
    assert "expected schema" in str(exc_info.value)
