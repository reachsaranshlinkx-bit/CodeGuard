import json

import httpx
import pytest

from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRequestError,
    GitHubResponseError,
    GitHubServerError,
)

COMMIT_SHA = "a" * 40

REVIEW_COMMENTS = [
    {
        "path": "src/auth.py",
        "line": 2,
        "side": "RIGHT",
        "body": "Avoid evaluating untrusted input.",
    }
]


def review_response() -> dict:
    return {
        "id": 123,
        "state": "COMMENTED",
        "commit_id": COMMIT_SHA,
        "body": "CodeGuard security review.",
        "html_url": (
            "https://github.com/octocat/CodeGuard/pull/42#pullrequestreview-123"
        ),
    }


async def test_create_pull_request_review_sends_comment_request() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=review_response(),
            request=request,
        )

    async with GitHubClient(
        token="test-token",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        result = await client.create_pull_request_review(
            owner="octocat",
            repo="CodeGuard",
            pull_number=42,
            commit_id=COMMIT_SHA,
            body="CodeGuard security review.",
            comments=REVIEW_COMMENTS,
        )

    assert result["id"] == 123
    assert result["state"] == "COMMENTED"

    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert request.method == "POST"
    assert request.url.path == ("/repos/octocat/CodeGuard/pulls/42/reviews")

    assert request.headers["Authorization"] == "Bearer test-token"

    payload = json.loads(request.content)

    assert payload["commit_id"] == COMMIT_SHA
    assert payload["body"] == "CodeGuard security review."
    assert payload["event"] == "COMMENT"
    assert payload["comments"] == REVIEW_COMMENTS


@pytest.mark.parametrize(
    ("owner", "repo", "pull_number"),
    [
        ("", "CodeGuard", 42),
        ("octocat", "", 42),
        ("octocat", "CodeGuard", 0),
        ("octocat", "CodeGuard", -1),
        ("octocat", "CodeGuard", True),
        ("octocat/team", "CodeGuard", 42),
        ("octocat", "CodeGuard/test", 42),
    ],
)
async def test_create_pull_request_review_rejects_invalid_identifiers(
    owner: str,
    repo: str,
    pull_number: int,
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=review_response(), request=request)

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(ValueError):
            await client.create_pull_request_review(
                owner=owner,
                repo=repo,
                pull_number=pull_number,
                commit_id=COMMIT_SHA,
                body="Security review.",
                comments=REVIEW_COMMENTS,
            )

    assert request_count == 0


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (401, GitHubAuthenticationError),
        (403, GitHubPermissionError),
        (404, GitHubNotFoundError),
        (422, GitHubAPIError),
        (500, GitHubServerError),
    ],
)
async def test_create_pull_request_review_maps_http_errors(
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
            await client.create_pull_request_review(
                owner="octocat",
                repo="CodeGuard",
                pull_number=42,
                commit_id=COMMIT_SHA,
                body="Security review.",
                comments=REVIEW_COMMENTS,
            )


async def test_create_review_does_not_retry_post_requests() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        raise httpx.ConnectError(
            "Connection failed",
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=2,
    ) as client:
        with pytest.raises(GitHubRequestError):
            await client.create_pull_request_review(
                owner="octocat",
                repo="CodeGuard",
                pull_number=42,
                commit_id=COMMIT_SHA,
                body="Security review.",
                comments=REVIEW_COMMENTS,
            )

    assert request_count == 1


async def test_create_review_rejects_invalid_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"invalid-json",
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError):
            await client.create_pull_request_review(
                owner="octocat",
                repo="CodeGuard",
                pull_number=42,
                commit_id=COMMIT_SHA,
                body="Security review.",
                comments=REVIEW_COMMENTS,
            )


async def test_create_review_rejects_non_object_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[],
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError):
            await client.create_pull_request_review(
                owner="octocat",
                repo="CodeGuard",
                pull_number=42,
                commit_id=COMMIT_SHA,
                body="Security review.",
                comments=REVIEW_COMMENTS,
            )


async def test_get_all_pull_request_reviews_returns_reviews() -> None:
    captured_requests: list[httpx.Request] = []

    expected_reviews = [
        {
            "id": 123,
            "state": "COMMENTED",
            "commit_id": COMMIT_SHA,
            "body": "CodeGuard security review.",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            200,
            json=expected_reviews,
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        reviews = await client.get_all_pull_request_reviews(
            "octocat",
            "CodeGuard",
            42,
        )

    assert reviews == expected_reviews
    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert request.method == "GET"
    assert request.url.path == ("/repos/octocat/CodeGuard/pulls/42/reviews")
    assert request.url.params["per_page"] == "100"
    assert request.url.params["page"] == "1"


async def test_get_all_pull_request_reviews_supports_pagination() -> None:
    requests: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        requests.append(page)

        review = {
            "id": page,
            "state": "COMMENTED",
            "commit_id": COMMIT_SHA,
            "body": f"Review page {page}",
        }

        if page == 1:
            return httpx.Response(
                200,
                json=[review],
                headers={
                    "Link": (
                        "<https://api.github.com/repos/octocat/"
                        "CodeGuard/pulls/42/reviews"
                        '?per_page=100&page=2>; rel="next"'
                    )
                },
                request=request,
            )

        return httpx.Response(
            200,
            json=[review],
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        reviews = await client.get_all_pull_request_reviews(
            "octocat",
            "CodeGuard",
            42,
        )

    assert requests == [1, 2]
    assert len(reviews) == 2
    assert reviews[0]["id"] == 1
    assert reviews[1]["id"] == 2


async def test_get_all_pull_request_reviews_rejects_non_list_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": "Unexpected response"},
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_all_pull_request_reviews(
                "octocat",
                "CodeGuard",
                42,
            )


async def test_get_all_pull_request_reviews_rejects_non_object_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=["invalid-review"],
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(GitHubResponseError):
            await client.get_all_pull_request_reviews(
                "octocat",
                "CodeGuard",
                42,
            )


@pytest.mark.parametrize(
    ("owner", "repo", "pull_number"),
    [
        ("", "CodeGuard", 42),
        ("octocat", "", 42),
        ("octocat", "CodeGuard", 0),
        ("octocat", "CodeGuard", True),
        ("octocat/team", "CodeGuard", 42),
    ],
)
async def test_get_all_pull_request_reviews_rejects_invalid_identifiers(
    owner: str,
    repo: str,
    pull_number: int,
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            200,
            json=[],
            request=request,
        )

    async with GitHubClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as client:
        with pytest.raises(ValueError):
            await client.get_all_pull_request_reviews(
                owner,
                repo,
                pull_number,
            )

    assert request_count == 0
