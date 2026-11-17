from typing import Any

import pytest

from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.analyzers.python_risk import RiskFinding
from codeguard.review.finding_models import (
    ReviewFinding,
    ReviewReport,
)
from codeguard.review.review_publisher import ReviewPublisher
from codeguard.services.pull_request_analyzer import (
    FileAnalysisResult,
    PullRequestAnalysisResult,
)

COMMIT_SHA = "a" * 40


def make_analysis(
    *,
    include_finding: bool = True,
) -> PullRequestAnalysisResult:
    risk_finding = RiskFinding(
        rule_id="PY001",
        severity="high",
        message="eval() executes dynamically supplied code.",
        file_path="src/auth.py",
        lineno=12,
        cwe="CWE-95",
    )

    findings = (risk_finding,) if include_finding else ()

    file_result = FileAnalysisResult(
        file_path="src/auth.py",
        status="modified",
        additions=1 if include_finding else 0,
        deletions=0,
        changes=1 if include_finding else 0,
        previous_filename=None,
        ast_result=PythonAnalysisResult(
            file_path="src/auth.py",
            functions=(),
            classes=(),
            imports=(),
            calls=(),
        ),
        risk_findings=findings,
        added_line_numbers=(frozenset({12}) if include_finding else frozenset()),
        changed_line_findings=findings,
        line_mapping_error=None,
    )

    return PullRequestAnalysisResult(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        state="open",
        head_sha=COMMIT_SHA,
        base_sha="b" * 40,
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        files=(file_result,),
    )


def make_report(
    *,
    include_finding: bool = True,
) -> ReviewReport:
    findings = ()

    if include_finding:
        findings = (
            ReviewFinding(
                fingerprint="stable-fingerprint",
                rule_id="PY001",
                title="Dynamic Code Execution",
                severity="high",
                confidence="high",
                message="eval() executes dynamically supplied code.",
                description="Untrusted input may execute Python code.",
                file_path="src/auth.py",
                line_number=12,
                cwe="CWE-95",
                evidence="eval(username)",
                remediation="Avoid evaluating untrusted input.",
                references=("https://cwe.mitre.org/data/definitions/95.html",),
            ),
        )

    return ReviewReport(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        head_sha=COMMIT_SHA,
        findings=findings,
        files_analyzed=1,
    )


class FakeGitHubClient:
    """Fake GitHub client for publisher tests."""

    def __init__(
        self,
        error: Exception | None = None,
        existing_reviews: list[dict[str, Any]] | None = None,
        lookup_error: Exception | None = None,
    ) -> None:
        self.error = error
        self.lookup_error = lookup_error
        self.existing_reviews = existing_reviews or []
        self.review_lookup_calls = 0
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
                "commit_id": commit_id,
                "body": body,
                "comments": comments,
            }
        )

        if self.error is not None:
            raise self.error

        return {
            "id": 123,
            "state": "COMMENTED",
        }


async def test_publisher_defaults_to_dry_run() -> None:
    client = FakeGitHubClient()

    publisher = ReviewPublisher(github_client=client)

    result = await publisher.publish(
        make_report(),
        make_analysis(),
    )

    assert result.status == "dry_run"
    assert result.comment_count == 1
    assert result.review_body is not None
    assert "CodeGuard security review" in result.review_body
    assert client.calls == []


async def test_dry_run_never_calls_github() -> None:
    client = FakeGitHubClient()

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=True,
    )

    result = await publisher.publish(
        make_report(),
        make_analysis(),
    )

    assert result.status == "dry_run"
    assert client.calls == []


async def test_publish_sends_review_to_correct_pull_request() -> None:
    client = FakeGitHubClient()

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(),
        make_analysis(),
    )

    assert result.status == "published"
    assert result.comment_count == 1

    assert len(client.calls) == 1

    call = client.calls[0]

    assert "<!-- codeguard-review:v1:" in call["body"]

    assert call["owner"] == "octocat"
    assert call["repo"] == "CodeGuard"
    assert call["pull_number"] == 42
    assert call["commit_id"] == COMMIT_SHA

    assert call["comments"] == [
        {
            "path": "src/auth.py",
            "line": 12,
            "side": "RIGHT",
            "body": result.comments[0].body,
        }
    ]

    assert "Severity summary" in call["body"]
    assert "High: 1" in call["body"]


async def test_publisher_skips_github_when_no_comments_exist() -> None:
    client = FakeGitHubClient()

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(include_finding=False),
        make_analysis(include_finding=False),
    )

    assert result.status == "skipped"
    assert result.comment_count == 0
    assert client.calls == []


async def test_publisher_propagates_github_errors() -> None:
    from codeguard.github.exceptions import GitHubPermissionError

    client = FakeGitHubClient(
        error=GitHubPermissionError(
            "Insufficient permissions",
            status_code=403,
        )
    )

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    with pytest.raises(GitHubPermissionError):
        await publisher.publish(
            make_report(),
            make_analysis(),
        )

    assert len(client.calls) == 1


async def test_publisher_rejects_mismatched_analysis() -> None:
    client = FakeGitHubClient()

    publisher = ReviewPublisher(github_client=client)

    report = make_report()

    mismatched_analysis = PullRequestAnalysisResult(
        owner="another-owner",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        state="open",
        head_sha=COMMIT_SHA,
        base_sha="b" * 40,
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        files=make_analysis().files,
    )

    with pytest.raises(ValueError, match="does not match"):
        await publisher.publish(
            report,
            mismatched_analysis,
        )

    assert client.calls == []


async def test_publisher_skips_existing_duplicate_review() -> None:
    # First generate a dry-run preview.
    preview_client = FakeGitHubClient()

    preview_publisher = ReviewPublisher(
        github_client=preview_client,
    )

    preview = await preview_publisher.publish(
        make_report(),
        make_analysis(),
    )

    assert preview.status == "dry_run"

    # Simulate the same review already existing on GitHub.
    client = FakeGitHubClient(
        existing_reviews=[
            {
                "commit_id": COMMIT_SHA,
                "body": preview.review_body,
            }
        ]
    )

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(),
        make_analysis(),
    )

    assert result.status == "duplicate"
    assert result.comment_count == 1
    assert client.review_lookup_calls == 1
    assert client.calls == []


async def test_publisher_does_not_skip_review_from_another_commit() -> None:
    preview_client = FakeGitHubClient()

    preview = await ReviewPublisher(
        github_client=preview_client,
    ).publish(
        make_report(),
        make_analysis(),
    )

    client = FakeGitHubClient(
        existing_reviews=[
            {
                "commit_id": "b" * 40,
                "body": preview.review_body,
            }
        ]
    )

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(),
        make_analysis(),
    )

    assert result.status == "published"
    assert client.review_lookup_calls == 1
    assert len(client.calls) == 1


async def test_publisher_does_not_publish_when_review_lookup_fails() -> None:
    from codeguard.github.exceptions import GitHubPermissionError

    client = FakeGitHubClient(
        lookup_error=GitHubPermissionError(
            "Cannot retrieve existing reviews.",
            status_code=403,
        )
    )

    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    with pytest.raises(GitHubPermissionError):
        await publisher.publish(
            make_report(),
            make_analysis(),
        )

    assert client.review_lookup_calls == 1
    assert client.calls == []
