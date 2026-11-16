import pytest

from codeguard.github.schemas import (
    GitHubPullRequest,
    GitHubPullRequestFile,
)
from codeguard.review.report_builder import build_review_report
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer


def make_pull_request() -> GitHubPullRequest:
    return GitHubPullRequest.model_validate(
        {
            "url": "https://api.github.com/repos/octocat/CodeGuard/pulls/42",
            "html_url": "https://github.com/octocat/CodeGuard/pull/42",
            "number": 42,
            "state": "open",
            "title": "Improve authentication",
            "base": {
                "ref": "main",
                "sha": "a" * 40,
            },
            "head": {
                "ref": "feature/auth",
                "sha": "b" * 40,
            },
        }
    )


def make_file(
    patch: str | None,
    additions: int,
    deletions: int,
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
    """Fake GitHub client for end-to-end review pipeline tests."""

    def __init__(
        self,
        changed_file: GitHubPullRequestFile,
        source: str,
    ) -> None:
        self.pull_request = make_pull_request()
        self.files = [changed_file]
        self.source = source

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
        return self.source


@pytest.mark.asyncio
async def test_end_to_end_review_pipeline_detects_added_line_finding() -> None:
    source = "def authenticate(value):\n    return eval(value)\n"

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(value):\n"
        "-    return value\n"
        "+    return eval(value)\n"
    )

    github_client = FakeGitHubClient(
        changed_file=make_file(
            patch=patch,
            additions=1,
            deletions=1,
        ),
        source=source,
    )

    analyzer = PullRequestAnalyzer(github_client=github_client)

    analysis = await analyzer.analyze(
        "octocat",
        "CodeGuard",
        42,
    )

    report = build_review_report(analysis)

    assert report.owner == "octocat"
    assert report.repo == "CodeGuard"
    assert report.pull_number == 42
    assert report.files_analyzed == 1

    assert report.total_findings == 1

    finding = report.findings[0]

    assert finding.rule_id == "PY001"
    assert finding.severity == "high"
    assert finding.line_number == 2
    assert finding.file_path == "src/auth.py"
    assert finding.cwe == "CWE-95"
    assert finding.fingerprint

    assert report.files_with_unavailable_mapping == ()


@pytest.mark.asyncio
async def test_end_to_end_report_excludes_unchanged_line_findings() -> None:
    source = "def authenticate(value):\n    eval(value)\n    return str(value)\n"

    patch = (
        "@@ -1,2 +1,3 @@\n"
        " def authenticate(value):\n"
        "     eval(value)\n"
        "+    return str(value)\n"
    )

    github_client = FakeGitHubClient(
        changed_file=make_file(
            patch=patch,
            additions=1,
            deletions=0,
        ),
        source=source,
    )

    analyzer = PullRequestAnalyzer(github_client=github_client)

    analysis = await analyzer.analyze(
        "octocat",
        "CodeGuard",
        42,
    )

    report = build_review_report(analysis)

    assert report.files_analyzed == 1
    assert report.total_findings == 0
    assert report.severity_counts["high"] == 0


@pytest.mark.asyncio
async def test_end_to_end_report_preserves_findings_without_patch() -> None:
    source = "def authenticate(value):\n    return eval(value)\n"

    github_client = FakeGitHubClient(
        changed_file=make_file(
            patch=None,
            additions=1,
            deletions=0,
        ),
        source=source,
    )

    analyzer = PullRequestAnalyzer(github_client=github_client)

    analysis = await analyzer.analyze(
        "octocat",
        "CodeGuard",
        42,
    )

    report = build_review_report(analysis)

    assert report.files_analyzed == 1
    assert report.total_findings == 1
    assert report.findings[0].rule_id == "PY001"

    assert report.files_with_unavailable_mapping == ("src/auth.py",)
