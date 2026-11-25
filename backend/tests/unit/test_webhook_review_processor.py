from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

import codeguard.services.webhook_review_processor as processor_module
from codeguard.ai.contracts import AIFindingCandidate, AIReviewResult
from codeguard.ai.engine import (
    AIReviewExecutionResult,
    AIReviewFindingAssessment,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.github.client import GitHubClient
from codeguard.review.finding_models import ReviewReport
from codeguard.review.report_builder import build_review_report
from codeguard.review.review_publisher import ReviewPublishResult
from codeguard.services.pull_request_analyzer import (
    FileAnalysisResult,
    PullRequestAnalysisResult,
    PullRequestAnalyzer,
)
from codeguard.services.webhook_review_processor import (
    WebhookReviewProcessor,
)


@pytest.mark.parametrize(
    ("publishing_enabled", "expected_dry_run"),
    [
        (False, True),
        (True, False),
    ],
)
async def test_processor_respects_publishing_setting(
    monkeypatch: pytest.MonkeyPatch,
    publishing_enabled: bool,
    expected_dry_run: bool,
) -> None:
    analysis = cast(PullRequestAnalysisResult, object())
    report = cast(ReviewReport, object())
    github_client = cast(GitHubClient, object())

    analyzer = Mock(spec=PullRequestAnalyzer)
    analyzer.analyze = AsyncMock(return_value=analysis)

    expected_result = ReviewPublishResult(
        status="dry_run" if expected_dry_run else "published",
        comment_count=1,
        review_body="Test review",
        comments=(),
    )

    observed: dict[str, object] = {}

    def fake_build_report(
        received_analysis: PullRequestAnalysisResult,
    ) -> ReviewReport:
        assert received_analysis is analysis
        return report

    class FakePublisher:
        def __init__(
            self,
            *,
            github_client: GitHubClient,
            dry_run: bool,
        ) -> None:
            observed["github_client"] = github_client
            observed["dry_run"] = dry_run

        async def publish(
            self,
            report: ReviewReport,
            analysis: PullRequestAnalysisResult,
        ) -> ReviewPublishResult:
            observed["report"] = report
            observed["analysis"] = analysis
            return expected_result

    monkeypatch.setattr(
        processor_module,
        "build_review_report",
        fake_build_report,
    )
    monkeypatch.setattr(
        processor_module,
        "ReviewPublisher",
        FakePublisher,
    )

    processor = WebhookReviewProcessor(
        analyzer=cast(PullRequestAnalyzer, analyzer),
        github_client=github_client,
        publishing_enabled=publishing_enabled,
    )

    result = await processor.process(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    analyzer.analyze.assert_awaited_once_with(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    assert observed["github_client"] is github_client
    assert observed["dry_run"] is expected_dry_run
    assert observed["report"] is report
    assert observed["analysis"] is analysis
    assert result is expected_result


async def test_processor_propagates_analysis_failure() -> None:
    analyzer = Mock(spec=PullRequestAnalyzer)
    analyzer.analyze = AsyncMock(
        side_effect=RuntimeError("Simulated analysis failure"),
    )

    processor = WebhookReviewProcessor(
        analyzer=cast(PullRequestAnalyzer, analyzer),
        github_client=cast(GitHubClient, object()),
        publishing_enabled=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated analysis failure",
    ):
        await processor.process(
            owner="octocat",
            repo="CodeGuard",
            pull_number=42,
        )

    analyzer.analyze.assert_awaited_once()


async def test_processor_passes_ai_assessment_report_through_to_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = AIFindingCandidate(
        title="Potential unsafe value handling",
        severity="medium",
        confidence="medium",
        file_path="src/auth.py",
        line_number=1,
        problem="The returned value may not be validated.",
        evidence="return value",
        why_it_matters="Unexpected values may reach downstream code.",
        suggested_fix="Validate the value before returning it.",
        recommended_validation="Add tests for malformed values.",
    )
    assessment = AIReviewFindingAssessment(
        candidate=candidate,
        evidence_match=EvidenceMatchResult(
            status=EvidenceMatchStatus.MATCHED,
            reason=EvidenceMatchReason.EXACT_MATCH,
            file_path=candidate.file_path,
            line_number=candidate.line_number,
        ),
    )
    ai_review_result = AIReviewExecutionResult(
        provider_result=AIReviewResult(
            summary="One potential issue was assessed.",
            findings=(candidate,),
        ),
        assessments=(assessment,),
    )
    analysis = PullRequestAnalysisResult(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        state="open",
        head_sha="abc123",
        base_sha="def456",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        files=(
            FileAnalysisResult(
                file_path="src/auth.py",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
                previous_filename=None,
                ast_result=PythonAnalysisResult(
                    file_path="src/auth.py",
                    functions=(),
                    classes=(),
                    imports=(),
                    calls=(),
                ),
                risk_findings=(),
                added_line_numbers=frozenset({1}),
                changed_line_findings=(),
                line_mapping_error=None,
            ),
        ),
        ai_review_result=ai_review_result,
    )
    expected_report = build_review_report(analysis)
    analyzer = Mock(spec=PullRequestAnalyzer)
    analyzer.analyze = AsyncMock(return_value=analysis)
    github_client = cast(GitHubClient, object())
    expected_result = ReviewPublishResult(
        status="dry_run",
        comment_count=0,
        review_body="Test review",
        comments=(),
    )
    observed: dict[str, object] = {}

    class FakePublisher:
        def __init__(
            self,
            *,
            github_client: GitHubClient,
            dry_run: bool,
        ) -> None:
            observed["github_client"] = github_client
            observed["dry_run"] = dry_run

        async def publish(
            self,
            report: ReviewReport,
            analysis: PullRequestAnalysisResult,
        ) -> ReviewPublishResult:
            observed["report"] = report
            observed["analysis"] = analysis
            return expected_result

    monkeypatch.setattr(
        processor_module,
        "ReviewPublisher",
        FakePublisher,
    )

    processor = WebhookReviewProcessor(
        analyzer=cast(PullRequestAnalyzer, analyzer),
        github_client=github_client,
        publishing_enabled=False,
    )

    result = await processor.process(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    analyzer.analyze.assert_awaited_once_with(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    report = observed["report"]
    assert isinstance(report, ReviewReport)
    assert report == expected_report
    assert report.ai_review_summary == "One potential issue was assessed."
    assert report.ai_assessments == (assessment,)
    assert report.findings == ()
    assert report.total_findings == 0

    assert observed["github_client"] is github_client
    assert observed["dry_run"] is True
    assert observed["analysis"] is analysis
    assert result is expected_result
