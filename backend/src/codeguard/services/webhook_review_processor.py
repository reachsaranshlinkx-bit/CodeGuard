from codeguard.github.client import GitHubClient
from codeguard.review.report_builder import build_review_report
from codeguard.review.review_publisher import (
    ReviewPublisher,
    ReviewPublishResult,
)
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer


class WebhookReviewProcessor:
    """Orchestrate pull request analysis and review publishing."""

    def __init__(
        self,
        *,
        analyzer: PullRequestAnalyzer,
        github_client: GitHubClient,
        publishing_enabled: bool,
    ) -> None:
        self._analyzer = analyzer
        self._github_client = github_client
        self._publishing_enabled = publishing_enabled

    async def process(
        self,
        *,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> ReviewPublishResult:
        """Analyze a pull request and generate or publish its review."""

        # Reuse the existing CodeGuard analysis pipeline.
        analysis = await self._analyzer.analyze(
            owner=owner,
            repo=repo,
            pull_number=pull_number,
        )

        # Convert analysis results into the structured review report.
        report = build_review_report(analysis)

        # Publishing remains controlled by the server configuration.
        publisher = ReviewPublisher(
            github_client=self._github_client,
            dry_run=not self._publishing_enabled,
        )

        # Generate the review or publish it to GitHub.
        return await publisher.publish(
            report=report,
            analysis=analysis,
        )
