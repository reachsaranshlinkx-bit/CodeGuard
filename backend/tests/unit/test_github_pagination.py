import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import GitHubResponseError


def make_file(filename: str) -> dict:
    return {
        "filename": filename,
        "status": "modified",
        "additions": 2,
        "deletions": 1,
        "changes": 3,
    }


def next_page_link(page: int) -> str:
    return (
        "<https://api.github.com/repos/octocat/CodeGuard/"
        f'pulls/42/files?per_page=100&page={page}>; rel="next"'
    )


async def test_pagination_retrieves_multiple_pages() -> None:
    requests: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        requests.append(page)

        if page == 1:
            return httpx.Response(
                200,
                json=[make_file("src/auth.py")],
                headers={"Link": next_page_link(2)},
                request=request,
            )

        return httpx.Response(
            200,
            json=[make_file("src/database.py")],
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        files = await client.get_all_pull_request_files(
            "octocat",
            "CodeGuard",
            42,
        )

    assert requests == [1, 2]
    assert len(files) == 2
    assert files[0].filename == "src/auth.py"
    assert files[1].filename == "src/database.py"


async def test_pagination_stops_without_next_link() -> None:
    requests: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(int(request.url.params["page"]))

        return httpx.Response(
            200,
            json=[make_file("src/auth.py")],
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        files = await client.get_all_pull_request_files(
            "octocat",
            "CodeGuard",
            42,
        )

    assert len(files) == 1
    assert requests == [1]


async def test_pagination_rejects_untrusted_next_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[make_file("src/auth.py")],
            headers={
                "Link": (
                    "<https://example.com/repos/octocat/CodeGuard/"
                    'pulls/42/files?per_page=100&page=2>; rel="next"'
                )
            },
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_all_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
            )


async def test_pagination_rejects_unexpected_next_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[make_file("src/auth.py")],
            headers={"Link": next_page_link(5)},
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_all_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
            )


async def test_pagination_raises_when_max_pages_reached() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[make_file("src/auth.py")],
            headers={"Link": next_page_link(2)},
            request=request,
        )

    async with GitHubClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(
            GitHubResponseError,
            match="Maximum pagination page limit",
        ):
            await client.get_all_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
                max_pages=1,
            )


@pytest.mark.parametrize("max_pages", [0, 31, True])
async def test_pagination_rejects_invalid_max_pages(
    max_pages: int,
) -> None:
    async with GitHubClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=[],
                request=request,
            )
        )
    ) as client:
        with pytest.raises(ValueError):
            await client.get_all_pull_request_files(
                "octocat",
                "CodeGuard",
                42,
                max_pages=max_pages,
            )
