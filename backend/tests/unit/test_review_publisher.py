from dataclasses import replace
from typing import Any

import pytest

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.engine import AIReviewFindingAssessment
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
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


def replace_analysis_file_result(
    analysis: PullRequestAnalysisResult,
    **updates: Any,
) -> PullRequestAnalysisResult:
    file_result = replace(analysis.files[0], **updates)
    return replace(analysis, files=(file_result,))


def make_ai_candidate(
    *,
    title: str = "Potential unsafe dynamic execution",
    severity: str = "high",
    path: str = "src/auth.py",
    line: int = 12,
    evidence: str = "eval(username)",
    suggested_fix: str = "Use a safe parser for the expected input format.",
) -> AIFindingCandidate:
    return AIFindingCandidate(
        title=title,
        severity=severity,
        confidence="medium",
        file_path=path,
        line_number=line,
        problem="The code executes dynamically supplied input.",
        evidence=evidence,
        why_it_matters="Untrusted input could execute arbitrary code.",
        suggested_fix=suggested_fix,
        recommended_validation="Add a test for untrusted input.",
    )


def make_ai_assessment(
    candidate: AIFindingCandidate | None = None,
    *,
    status: EvidenceMatchStatus = EvidenceMatchStatus.MATCHED,
    reason: EvidenceMatchReason = EvidenceMatchReason.EXACT_MATCH,
    evidence_path: str | None = None,
    evidence_line: int | None = None,
) -> AIReviewFindingAssessment:
    candidate = candidate or make_ai_candidate()

    return AIReviewFindingAssessment(
        candidate=candidate,
        evidence_match=EvidenceMatchResult(
            status=status,
            reason=reason,
            file_path=(candidate.file_path if evidence_path is None else evidence_path),
            line_number=(
                candidate.line_number if evidence_line is None else evidence_line
            ),
        ),
    )


def make_report(
    *,
    include_finding: bool = True,
    ai_assessments: tuple[AIReviewFindingAssessment, ...] = (),
    ai_review_summary: str | None = None,
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
        ai_review_summary=ai_review_summary,
        ai_assessments=ai_assessments,
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


async def test_ai_only_dry_run_includes_candidate_without_contacting_github() -> None:
    candidate = make_ai_candidate()
    assessment = make_ai_assessment(candidate)
    client = FakeGitHubClient()
    publisher = ReviewPublisher(github_client=client)

    result = await publisher.publish(
        make_report(
            include_finding=False,
            ai_assessments=(assessment,),
        ),
        make_analysis(),
    )

    assert result.status == "dry_run"
    assert result.comment_count == 1
    assert result.comments[0].path == candidate.file_path
    assert result.comments[0].line == candidate.line_number
    assert candidate.title in result.comments[0].body
    assert candidate.problem in result.comments[0].body
    assert candidate.evidence in result.comments[0].body
    assert candidate.suggested_fix in result.comments[0].body
    assert client.review_lookup_calls == 0
    assert client.calls == []


async def test_ai_only_publish_sends_candidate_comment_to_correct_pull_request() -> (
    None
):
    candidate = make_ai_candidate()
    assessment = make_ai_assessment(candidate)
    client = FakeGitHubClient()
    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(
            include_finding=False,
            ai_assessments=(assessment,),
        ),
        make_analysis(),
    )

    assert result.status == "published"
    assert result.comment_count == 1
    assert client.review_lookup_calls == 1
    assert len(client.calls) == 1

    call = client.calls[0]
    assert call["owner"] == "octocat"
    assert call["repo"] == "CodeGuard"
    assert call["pull_number"] == 42
    assert call["commit_id"] == COMMIT_SHA
    assert call["comments"] == [
        {
            "path": candidate.file_path,
            "line": candidate.line_number,
            "side": "RIGHT",
            "body": result.comments[0].body,
        }
    ]


async def test_mixed_static_and_ai_comments_publish_together() -> None:
    candidate = make_ai_candidate()
    assessment = make_ai_assessment(candidate)
    client = FakeGitHubClient()
    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(ai_assessments=(assessment,)),
        make_analysis(),
    )

    expected_static_body = (
        "**Dynamic Code Execution** (PY001)\n\n"
        "**Severity:** high\n\n"
        "eval() executes dynamically supplied code.\n\n"
        "**Evidence:**\n"
        "```python\neval(username)\n```\n\n"
        "**Recommended remediation:**\n"
        "Avoid evaluating untrusted input.\n\n"
        "**CWE:** CWE-95"
    )

    assert result.status == "published"
    assert result.comment_count == 2
    assert result.comments[0].body == expected_static_body
    assert candidate.title in result.comments[1].body
    assert len(client.calls) == 1
    assert client.calls[0]["comments"] == [
        {
            "path": "src/auth.py",
            "line": 12,
            "side": "RIGHT",
            "body": expected_static_body,
        },
        {
            "path": candidate.file_path,
            "line": candidate.line_number,
            "side": "RIGHT",
            "body": result.comments[1].body,
        },
    ]


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            EvidenceMatchStatus.MISMATCHED,
            EvidenceMatchReason.EXACT_MATCH,
        ),
        (
            EvidenceMatchStatus.UNVERIFIABLE,
            EvidenceMatchReason.LOCATION_UNVERIFIABLE,
        ),
    ],
)
async def test_unsafe_ai_evidence_is_not_published(
    status: EvidenceMatchStatus,
    reason: EvidenceMatchReason,
) -> None:
    assessment = make_ai_assessment(status=status, reason=reason)
    client = FakeGitHubClient()
    publisher = ReviewPublisher(
        github_client=client,
        dry_run=False,
    )

    result = await publisher.publish(
        make_report(
            include_finding=False,
            ai_assessments=(assessment,),
        ),
        make_analysis(),
    )

    assert result.status == "skipped"
    assert result.comment_count == 0
    assert client.review_lookup_calls == 0
    assert client.calls == []


@pytest.mark.parametrize(
    ("evidence_path", "evidence_line"),
    [
        ("src/other.py", 12),
        ("src/auth.py", 11),
    ],
)
async def test_inconsistent_ai_evidence_location_is_not_published(
    evidence_path: str,
    evidence_line: int,
) -> None:
    assessment = make_ai_assessment(
        evidence_path=evidence_path,
        evidence_line=evidence_line,
    )
    client = FakeGitHubClient()
    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        make_report(
            include_finding=False,
            ai_assessments=(assessment,),
        ),
        make_analysis(),
    )

    assert result.status == "skipped"
    assert client.review_lookup_calls == 0
    assert client.calls == []


async def test_unchanged_ai_line_is_not_published() -> None:
    candidate = make_ai_candidate(line=8)
    client = FakeGitHubClient()
    analysis = replace_analysis_file_result(
        make_analysis(),
        added_line_numbers=frozenset({12}),
    )
    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        make_report(
            include_finding=False,
            ai_assessments=(make_ai_assessment(candidate),),
        ),
        analysis,
    )

    assert result.status == "skipped"
    assert client.review_lookup_calls == 0
    assert client.calls == []


@pytest.mark.parametrize(
    "updates",
    [
        {"added_line_numbers": None},
        {"changed_line_findings": None},
        {"line_mapping_error": "patch_unavailable"},
    ],
)
async def test_missing_ai_diff_mapping_is_not_published(
    updates: dict[str, Any],
) -> None:
    client = FakeGitHubClient()
    analysis = replace_analysis_file_result(make_analysis(), **updates)
    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        make_report(
            include_finding=False,
            ai_assessments=(make_ai_assessment(),),
        ),
        analysis,
    )

    assert result.status == "skipped"
    assert client.review_lookup_calls == 0
    assert client.calls == []


async def test_ai_assessments_do_not_change_static_review_body_or_counts() -> None:
    candidate = make_ai_candidate(severity="critical")
    report = make_report(
        ai_assessments=(make_ai_assessment(candidate),),
        ai_review_summary="Do not add this summary to the GitHub review body.",
    )
    client = FakeGitHubClient()
    result = await ReviewPublisher(github_client=client).publish(
        report,
        make_analysis(),
    )
    static_preview = await ReviewPublisher(
        github_client=FakeGitHubClient(),
    ).publish(
        make_report(),
        make_analysis(),
    )

    review_body_without_marker = result.review_body.split(
        "\n\n<!-- codeguard-review:v1:",
        maxsplit=1,
    )[0]
    static_body_without_marker = static_preview.review_body.split(
        "\n\n<!-- codeguard-review:v1:",
        maxsplit=1,
    )[0]

    assert report.total_findings == 1
    assert report.severity_counts["high"] == 1
    assert report.severity_counts["critical"] == 0
    assert review_body_without_marker == static_body_without_marker
    assert report.ai_review_summary not in result.review_body
    assert candidate.title not in result.review_body


def _review_marker(review_body: str) -> str:
    return review_body.split("<!-- codeguard-review:v1:", maxsplit=1)[1].split(
        " -->",
        maxsplit=1,
    )[0]


async def test_identical_ai_comment_sets_have_the_same_marker_and_are_duplicates() -> (
    None
):
    report = make_report(
        include_finding=False,
        ai_assessments=(make_ai_assessment(),),
    )
    analysis = make_analysis()

    preview_one = await ReviewPublisher(
        github_client=FakeGitHubClient(),
    ).publish(report, analysis)
    preview_two = await ReviewPublisher(
        github_client=FakeGitHubClient(),
    ).publish(report, analysis)

    assert _review_marker(preview_one.review_body) == _review_marker(
        preview_two.review_body
    )

    client = FakeGitHubClient(
        existing_reviews=[
            {
                "commit_id": COMMIT_SHA,
                "body": preview_one.review_body,
            }
        ]
    )
    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(report, analysis)

    assert result.status == "duplicate"
    assert client.review_lookup_calls == 1
    assert client.calls == []


async def test_changed_ai_comment_content_has_a_different_fingerprint() -> None:
    analysis = make_analysis()
    original_report = make_report(
        include_finding=False,
        ai_assessments=(make_ai_assessment(),),
    )
    changed_candidate = make_ai_candidate(
        suggested_fix="Reject the input before it reaches dynamic execution.",
    )
    changed_report = make_report(
        include_finding=False,
        ai_assessments=(make_ai_assessment(changed_candidate),),
    )
    preview = await ReviewPublisher(
        github_client=FakeGitHubClient(),
    ).publish(original_report, analysis)
    client = FakeGitHubClient(
        existing_reviews=[
            {
                "commit_id": COMMIT_SHA,
                "body": preview.review_body,
            }
        ]
    )
    changed_result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(changed_report, analysis)

    assert _review_marker(changed_result.review_body) != _review_marker(
        preview.review_body
    )
    assert changed_result.status == "published"
    assert len(client.calls) == 1


async def test_ai_summary_without_eligible_comment_keeps_skip_behavior() -> None:
    client = FakeGitHubClient()
    analysis = replace_analysis_file_result(
        make_analysis(),
        added_line_numbers=None,
    )
    result = await ReviewPublisher(
        github_client=client,
        dry_run=False,
    ).publish(
        make_report(
            include_finding=False,
            ai_assessments=(make_ai_assessment(),),
            ai_review_summary="A candidate was assessed but not mapped.",
        ),
        analysis,
    )

    assert result.status == "skipped"
    assert result.comment_count == 0
    assert client.review_lookup_calls == 0
    assert client.calls == []
