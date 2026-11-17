from __future__ import annotations

from typing import Any

import pytest

from codeguard.github.exceptions import GitHubPermissionError
from codeguard.github.schemas import (
    GitHubPullRequest,
    GitHubPullRequestFile,
)
from codeguard.review.report_builder import build_review_report
from codeguard.review.review_publisher import ReviewPublisher
from codeguard.services.pull_request_analyzer import (
    PullRequestAnalysisResult,
    PullRequestAnalyzer,
)

OWNER = "octocat"
REPO = "CodeGuard"
PULL_NUMBER = 42
HEAD_SHA = "b" * 40
BASE_SHA = "a" * 40

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
    """Fake GitHub client for end-to-end publishing tests."""

    def __init__(
        self,
        *,
        source: str = VULNERABLE_SOURCE,
        changed_file: GitHubPullRequestFile | None = None,
        existing_reviews: list[dict[str, Any]] | None = None,
        lookup_error: Exception | None = None,
    ) -> None:
        self.pull_request = make_pull_request()
        self.files = [
            changed_file or make_file(),
        ]
        self.source = source

        self.existing_reviews = existing_reviews or []
        self.lookup_error = lookup_error

        self.content_refs: list[str] = []
        self.review_lookup_calls = 0
        self.created_reviews: list[dict[str, Any]] = []

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
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


async def run_analysis(
    client: FakeGitHubClient,
) -> tuple[PullRequestAnalysisResult, Any]:
    analyzer = PullRequestAnalyzer(
        github_client=client,
    )

    analysis = await analyzer.analyze(
        OWNER,
        REPO,
        PULL_NUMBER,
    )

    report = build_review_report(analysis)

    return analysis, report


@pytest.mark.asyncio
async def test_end_to_end_pipeline_generates_dry_run_review() -> None:
    client = FakeGitHubClient()

    analysis, report = await run_analysis(client)

    publisher = ReviewPublisher(
        github_client=client,
    )

    result = await publisher.publish(
        report,
        analysis,
    )

    assert client.content_refs == [HEAD_SHA]

    assert report.total_findings == 1
    assert report.findings[0].rule_id == "PY001"

    assert result.status == "dry_run"
    assert result.comment_count == 1

    assert result.comments[0].path == "src/auth.py"
    assert result.comments[0].line == 2

    assert "<!-- codeguard-review:v1:" in result.review_body

    assert client.review_lookup_calls == 0
    assert client.created_reviews == []


@pytest.mark.asyncio
async def test_end_to_end_pipeline_publishes_when_explicitly_enabled() -> None:
    client = FakeGitHubClient()

    analysis, report = await run_analysis(client)

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        report,
        analysis,
    )

    assert result.status == "published"
    assert result.comment_count == 1

    assert client.review_lookup_calls == 1
    assert len(client.created_reviews) == 1

    published = client.created_reviews[0]

    assert published["owner"] == OWNER
    assert published["repo"] == REPO
    assert published["pull_number"] == PULL_NUMBER
    assert published["commit_id"] == HEAD_SHA

    assert len(published["comments"]) == 1
    assert published["comments"][0]["path"] == "src/auth.py"
    assert published["comments"][0]["line"] == 2

    assert "<!-- codeguard-review:v1:" in published["body"]


@pytest.mark.asyncio
async def test_end_to_end_pipeline_skips_existing_duplicate() -> None:
    client = FakeGitHubClient()

    analysis, report = await run_analysis(client)

    preview = await ReviewPublisher(
        github_client=client,
    ).publish(
        report,
        analysis,
    )

    client.existing_reviews = [
        {
            "commit_id": HEAD_SHA,
            "body": preview.review_body,
        }
    ]

    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        report,
        analysis,
    )

    assert result.status == "duplicate"
    assert client.review_lookup_calls == 1
    assert client.created_reviews == []


@pytest.mark.asyncio
async def test_end_to_end_pipeline_fails_safely_on_lookup_error() -> None:
    client = FakeGitHubClient(
        lookup_error=GitHubPermissionError(
            "Review access denied.",
            status_code=403,
        )
    )

    analysis, report = await run_analysis(client)

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    with pytest.raises(GitHubPermissionError):
        await publisher.publish(
            report,
            analysis,
        )

    assert client.review_lookup_calls == 1
    assert client.created_reviews == []


@pytest.mark.asyncio
async def test_end_to_end_pipeline_skips_review_without_findings() -> None:
    safe_source = "def authenticate(value):\n    return str(value)\n"

    safe_patch = "@@ -1,1 +1,2 @@\n def authenticate(value):\n+    return str(value)\n"

    client = FakeGitHubClient(
        source=safe_source,
        changed_file=make_file(
            patch=safe_patch,
            additions=1,
            deletions=0,
        ),
    )

    analysis, report = await run_analysis(client)

    assert report.total_findings == 0

    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        report,
        analysis,
    )

    assert result.status == "skipped"
    assert result.comment_count == 0

    assert client.review_lookup_calls == 0
    assert client.created_reviews == []


@pytest.mark.asyncio
async def test_end_to_end_pipeline_never_comments_without_patch_mapping() -> None:
    client = FakeGitHubClient(
        changed_file=make_file(
            patch=None,
            additions=1,
            deletions=0,
        ),
    )

    analysis, report = await run_analysis(client)

    # Full-file analysis still detects the vulnerability.
    assert report.total_findings == 1

    # But without reliable diff mapping, no inline comment is allowed.
    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        report,
        analysis,
    )

    assert result.status == "skipped"
    assert result.comment_count == 0

    assert client.review_lookup_calls == 0
    assert client.created_reviews == []
