from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from codeguard.review.comment_mapper import (
    build_inline_review_comments,
)
from codeguard.review.comment_models import InlineReviewComment
from codeguard.review.finding_models import ReviewReport
from codeguard.review.review_fingerprint import (
    create_review_fingerprint,
    create_review_marker,
)
from codeguard.services.pull_request_analyzer import (
    PullRequestAnalysisResult,
)


class GitHubReviewClientProtocol(Protocol):
    """Interface required by the review publisher."""

    async def get_all_pull_request_reviews(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[dict[str, Any]]: ...

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ReviewPublishResult:
    """Result of generating or publishing a GitHub review."""

    status: Literal["dry_run", "skipped", "published", "duplicate"]
    comment_count: int
    review_body: str
    comments: tuple[InlineReviewComment, ...]
    github_response: dict[str, Any] | None = None


class ReviewPublisher:
    """Publish CodeGuard security findings as GitHub PR reviews."""

    def __init__(
        self,
        github_client: GitHubReviewClientProtocol,
        dry_run: bool = True,
    ) -> None:
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a boolean.")

        self._github_client = github_client
        self._dry_run = dry_run

    async def publish(
        self,
        report: ReviewReport,
        analysis: PullRequestAnalysisResult,
    ) -> ReviewPublishResult:
        """Generate a review and optionally publish it to GitHub."""

        comments = build_inline_review_comments(
            report,
            analysis,
        )

        review_body = self._build_review_body(report)

        if not comments:
            return ReviewPublishResult(
                status="skipped",
                comment_count=0,
                review_body=review_body,
                comments=(),
            )

        fingerprint = create_review_fingerprint(
            report,
            comments,
        )
        marker = create_review_marker(fingerprint)

        review_body = f"{review_body}\n\n{marker}"

        # Dry-run must not read or write GitHub reviews.
        if self._dry_run:
            return ReviewPublishResult(
                status="dry_run",
                comment_count=len(comments),
                review_body=review_body,
                comments=comments,
            )

        # Retrieve existing reviews before attempting publication.
        existing_reviews = await self._github_client.get_all_pull_request_reviews(
            report.owner,
            report.repo,
            report.pull_number,
        )

        duplicate_exists = any(
            isinstance(review.get("commit_id"), str)
            and review["commit_id"].lower() == report.head_sha.lower()
            and isinstance(review.get("body"), str)
            and marker in review["body"]
            for review in existing_reviews
        )

        if duplicate_exists:
            return ReviewPublishResult(
                status="duplicate",
                comment_count=len(comments),
                review_body=review_body,
                comments=comments,
            )

        github_comments = [comment.to_github_payload() for comment in comments]

        github_response = await self._github_client.create_pull_request_review(
            owner=report.owner,
            repo=report.repo,
            pull_number=report.pull_number,
            commit_id=report.head_sha,
            body=review_body,
            comments=github_comments,
        )

        return ReviewPublishResult(
            status="published",
            comment_count=len(comments),
            review_body=review_body,
            comments=comments,
            github_response=github_response,
        )

    @staticmethod
    def _build_review_body(report: ReviewReport) -> str:
        """Build a deterministic security review summary."""

        severity_counts = report.severity_counts

        lines = [
            "CodeGuard security review",
            "",
            "Severity summary",
            "",
        ]

        for severity in (
            "critical",
            "high",
            "medium",
            "low",
            "info",
        ):
            lines.append(f"- {severity.title()}: {severity_counts[severity]}")

        lines.extend(
            [
                "",
                f"Files analyzed: {report.files_analyzed}",
                f"Total findings: {report.total_findings}",
                "",
                "Inline comments contain the actionable security findings.",
                "",
                "Generated by CodeGuard.",
            ]
        )

        return "\n".join(lines)
