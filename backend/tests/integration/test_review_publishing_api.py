from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from codeguard.api.dependencies import get_github_client
from codeguard.core.config import Settings
from codeguard.github.exceptions import GitHubPermissionError
from codeguard.github.schemas import (
    GitHubPullRequest,
    GitHubPullRequestFile,
)
from codeguard.main import create_app

OWNER = "octocat"
REPO = "CodeGuard"
PULL_NUMBER = 42
HEAD_SHA = "b" * 40
BASE_SHA = "a" * 40

API_KEY = "test-codeguard-api-key-for-integration-tests"

API_HEADERS = {
    "X-CodeGuard-API-Key": API_KEY,
}

API_PAYLOAD = {
    "owner": OWNER,
    "repo": REPO,
    "pull_number": PULL_NUMBER,
}

VULNERABLE_SOURCE = "def authenticate(value):\n    return eval(value)\n"

VULNERABLE_PATCH = (
    "@@ -1,2 +1,2 @@\n"
    " def authenticate(value):\n"
    "-    return value\n"
    "+    return eval(value)\n"
)


def make_pull_request() -> GitHubPullRequest:
    return GitHubPullRequest.model_validate(
        {
            "url": (f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}"),
            "html_url": (f"https://github.com/{OWNER}/{REPO}/pull/{PULL_NUMBER}"),
            "number": PULL_NUMBER,
            "state": "open",
            "title": "Improve authentication",
            "base": {
                "ref": "main",
                "sha": BASE_SHA,
            },
            "head": {
                "ref": "feature/auth",
                "sha": HEAD_SHA,
            },
        }
    )


def make_file(
    *,
    patch: str | None = VULNERABLE_PATCH,
    additions: int = 1,
    deletions: int = 1,
) -> GitHubPullRequestFile:
    return GitHubPullRequestFile.model_validate(
        {
            "filename": "src/auth.py",
            "status": "modified",
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
            "patch": patch,
        }
    )


class FakeGitHubClient:
    """Fake GitHub client for publishing API integration tests."""

    def __init__(
        self,
        *,
        source: str = VULNERABLE_SOURCE,
        changed_file: GitHubPullRequestFile | None = None,
        existing_reviews: list[dict[str, Any]] | None = None,
        lookup_error: Exception | None = None,
    ) -> None:
        self.pull_request = make_pull_request()
        self.files = [changed_file or make_file()]
        self.source = source
        self.existing_reviews = existing_reviews if existing_reviews is not None else []
        self.lookup_error = lookup_error

        self.pull_request_calls = 0
        self.content_refs: list[str] = []
        self.review_lookup_calls = 0
        self.created_reviews: list[dict[str, Any]] = []

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
        self.pull_request_calls += 1
        return self.pull_request

    async def get_all_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[GitHubPullRequestFile]:
        return self.files

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        self.content_refs.append(ref)
        return self.source

    async def get_all_pull_request_reviews(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[dict[str, Any]]:
        self.review_lookup_calls += 1

        if self.lookup_error is not None:
            raise self.lookup_error

        return self.existing_reviews

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.created_reviews.append(
            {
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
                "commit_id": commit_id,
                "body": body,
                "comments": comments,
            }
        )

        return {
            "id": 123,
            "state": "COMMENTED",
        }


def create_test_client(
    github_client: FakeGitHubClient,
    *,
    publishing_enabled: bool = False,
) -> tuple[TestClient, Any]:
    settings = Settings(
        _env_file=None,
        api_key=SecretStr(API_KEY),
        enable_review_publishing=publishing_enabled,
    )

    app = create_app(settings=settings)

    app.dependency_overrides[get_github_client] = lambda: github_client

    return TestClient(app), app


def test_publish_endpoint_defaults_to_dry_run() -> None:
    github = FakeGitHubClient()

    client, app = create_test_client(github)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "dry_run"
        assert data["comment_count"] == 1

        assert data["comments"][0]["path"] == "src/auth.py"
        assert data["comments"][0]["line"] == 2

        assert github.review_lookup_calls == 0
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_requires_server_permission() -> None:
    github = FakeGitHubClient()

    client, app = create_test_client(
        github,
        publishing_enabled=False,
    )

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json={
                    **API_PAYLOAD,
                    "publish": True,
                },
                headers=API_HEADERS,
            )

        assert response.status_code == 403
        assert github.pull_request_calls == 0
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_publishes_when_explicitly_enabled() -> None:
    github = FakeGitHubClient()

    client, app = create_test_client(
        github,
        publishing_enabled=True,
    )

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json={
                    **API_PAYLOAD,
                    "publish": True,
                },
                headers=API_HEADERS,
            )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "published"
        assert data["comment_count"] == 1

        assert github.review_lookup_calls == 1
        assert len(github.created_reviews) == 1

        published = github.created_reviews[0]

        assert published["owner"] == OWNER
        assert published["repo"] == REPO
        assert published["pull_number"] == PULL_NUMBER
        assert published["commit_id"] == HEAD_SHA

        assert len(published["comments"]) == 1
        assert published["comments"][0]["path"] == "src/auth.py"
        assert published["comments"][0]["line"] == 2

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_detects_existing_duplicate() -> None:
    github = FakeGitHubClient()

    client, app = create_test_client(
        github,
        publishing_enabled=True,
    )

    try:
        with client:
            preview = client.post(
                "/api/v1/reviews/publish",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert preview.status_code == 200

        github.existing_reviews.append(
            {
                "commit_id": HEAD_SHA,
                "body": preview.json()["review_body"],
            }
        )

        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json={
                    **API_PAYLOAD,
                    "publish": True,
                },
                headers=API_HEADERS,
            )

        assert response.status_code == 200
        assert response.json()["status"] == "duplicate"

        assert github.review_lookup_calls == 1
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_skips_review_without_findings() -> None:
    github = FakeGitHubClient(
        source=("def authenticate(value):\n    return str(value)\n"),
        changed_file=make_file(
            patch=(
                "@@ -1,1 +1,2 @@\n def authenticate(value):\n+    return str(value)\n"
            ),
            additions=1,
            deletions=0,
        ),
    )

    client, app = create_test_client(
        github,
        publishing_enabled=True,
    )

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json={
                    **API_PAYLOAD,
                    "publish": True,
                },
                headers=API_HEADERS,
            )

        assert response.status_code == 200
        assert response.json()["status"] == "skipped"
        assert response.json()["comment_count"] == 0

        assert github.review_lookup_calls == 0
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_never_comments_without_patch_mapping() -> None:
    github = FakeGitHubClient(
        changed_file=make_file(
            patch=None,
            additions=1,
            deletions=0,
        ),
    )

    client, app = create_test_client(
        github,
        publishing_enabled=True,
    )

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json={
                    **API_PAYLOAD,
                    "publish": True,
                },
                headers=API_HEADERS,
            )

        assert response.status_code == 200
        assert response.json()["status"] == "skipped"
        assert response.json()["comment_count"] == 0

        assert github.review_lookup_calls == 0
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_rejects_missing_api_key() -> None:
    github = FakeGitHubClient()

    client, app = create_test_client(github)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json=API_PAYLOAD,
            )

        assert response.status_code == 401
        assert github.pull_request_calls == 0
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()


def test_publish_endpoint_fails_safely_on_review_lookup_error() -> None:
    github = FakeGitHubClient(
        lookup_error=GitHubPermissionError(
            "Review access denied.",
            status_code=403,
        ),
    )

    client, app = create_test_client(
        github,
        publishing_enabled=True,
    )

    try:
        with client:
            response = client.post(
                "/api/v1/reviews/publish",
                json={
                    **API_PAYLOAD,
                    "publish": True,
                },
                headers=API_HEADERS,
            )

        assert response.status_code == 403
        assert github.review_lookup_calls == 1
        assert github.created_reviews == []

    finally:
        app.dependency_overrides.clear()
