import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import GitHubResponseError
from codeguard.github.schemas import GitHubPullRequestFile

PULL_REQUEST_FILES_RESPONSE = [
    {
        "sha": "a" * 40,
        "filename": "src/auth.py",
        "status": "modified",
        "additions": 5,
        "deletions": 2,
        "changes": 7,
        "patch": "@@ -1,2 +1,5 @@\n+new authentication logic",
    },
    {
        "sha": "b" * 40,
        "filename": "tests/test_auth.py",
        "status": "added",
        "additions": 20,
        "deletions": 0,
        "changes": 20,
    },
]


async def test_get_pull_request_files_returns_validated_models() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=PULL_REQUEST_FILES_RESPONSE,
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        files = await client.get_pull_request_files(
            "octocat",
            "CodeGuard",
            42,
        )

    assert len(files) == 2
    assert all(isinstance(file, GitHubPullRequestFile) for file in files)

    assert files[0].filename == "src/auth.py"
    assert files[0].additions == 5
    assert files[0].deletions == 2
    assert files[0].patch is not None

    assert files[1].filename == "tests/test_auth.py"
    assert files[1].patch is None

    request = captured_requests[0]

    assert request.url.path == ("/repos/octocat/CodeGuard/pulls/42/files")
    assert request.url.params["per_page"] == "100"
    assert request.url.params["page"] == "1"


async def test_get_pull_request_files_supports_pagination_parameters() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=[],
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        files = await client.get_pull_request_files(
            "octocat",
            "CodeGuard",
            42,
            per_page=25,
            page=2,
        )

    assert files == []
    assert captured_requests[0].url.params["per_page"] == "25"
    assert captured_requests[0].url.params["page"] == "2"


@pytest.mark.parametrize(
    ("per_page", "page"),
    [
        (0, 1),
        (101, 1),
        (True, 1),
        (100, 0),
        (100, -1),
        (100, True),
    ],
)
async def test_get_pull_request_files_rejects_invalid_pagination(
    per_page: int,
    page: int,
) -> None:
    async with GitHubClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[], request=request)
        )
    ) as client:
        with pytest.raises(ValueError):
            await client.get_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
                per_page=per_page,
                page=page,
            )


async def test_get_pull_request_files_rejects_non_list_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": "Unexpected response"},
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
            )


async def test_get_pull_request_files_rejects_invalid_file() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "filename": "src/auth.py",
                    "status": "invalid",
                    "additions": -1,
                }
            ],
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
            )
