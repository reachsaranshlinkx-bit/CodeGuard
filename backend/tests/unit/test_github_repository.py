import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import GitHubResponseError
from codeguard.github.schemas import GitHubRepository

REPOSITORY_RESPONSE = {
    "id": 123456,
    "name": "CodeGuard",
    "full_name": "octocat/CodeGuard",
    "private": False,
    "html_url": "https://github.com/octocat/CodeGuard",
    "description": "AI-powered GitHub PR reviewer",
    "default_branch": "main",
    "owner": {
        "login": "octocat",
        "id": 100,
    },
}


async def test_get_repository_returns_validated_model() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=REPOSITORY_RESPONSE,
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        repository = await client.get_repository(
            "octocat",
            "CodeGuard",
        )

    assert isinstance(repository, GitHubRepository)
    assert repository.full_name == "octocat/CodeGuard"
    assert repository.default_branch == "main"
    assert repository.owner.login == "octocat"

    assert captured_requests[0].url.path == ("/repos/octocat/CodeGuard")


@pytest.mark.parametrize(
    ("owner", "repo"),
    [
        ("", "CodeGuard"),
        ("octocat", ""),
        ("octocat/team", "CodeGuard"),
        ("octocat", "CodeGuard/test"),
    ],
)
async def test_get_repository_rejects_invalid_identifiers(
    owner: str,
    repo: str,
) -> None:
    async with GitHubClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=REPOSITORY_RESPONSE,
                request=request,
            )
        )
    ) as client:
        with pytest.raises(ValueError):
            await client.get_repository(owner, repo)


async def test_get_repository_rejects_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"name": "Incomplete repository"},
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError) as exc_info:
            await client.get_repository(
                "octocat",
                "CodeGuard",
            )

    assert exc_info.value.status_code == 200
    assert "expected schema" in str(exc_info.value)
