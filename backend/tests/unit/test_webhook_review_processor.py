from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

import codeguard.services.webhook_review_processor as processor_module
from codeguard.github.client import GitHubClient
from codeguard.review.finding_models import ReviewReport
from codeguard.review.review_publisher import ReviewPublishResult
from codeguard.services.pull_request_analyzer import (
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
