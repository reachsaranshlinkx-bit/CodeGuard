from dataclasses import asdict
from math import ceil
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from codeguard.api.dependencies import (
    get_github_client,
    get_pull_request_analyzer,
    require_api_key,
)
from codeguard.context.symbols import RepositorySymbolIndex
from codeguard.core.config import Settings, get_settings
from codeguard.github.client import GitHubClient
from codeguard.github.exceptions import (
    GitHubAuthenticationError,
    GitHubError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRateLimitError,
)
from codeguard.review.report_builder import build_review_report
from codeguard.review.review_publisher import ReviewPublisher
from codeguard.schemas.review import (
    RepositoryContextResponse,
    RepositoryFileFailureResponse,
    ReviewFindingResponse,
    ReviewPublishCommentResponse,
    ReviewPublishRequest,
    ReviewPublishResponse,
    ReviewRequest,
    ReviewResponse,
    SeverityCountsResponse,
)
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer

router = APIRouter(
    prefix="/api/v1",
    tags=["reviews"],
)


def build_repository_context_response(
    context: RepositorySymbolIndex | None,
) -> RepositoryContextResponse | None:
    """Convert repository context into a safe API response."""

    if context is None:
        return None

    failures = tuple(
        RepositoryFileFailureResponse(
            file_path=failure.file_path,
            stage=failure.stage,
            error_type=failure.error_type,
        )
        for failure in context.failures
    )

    return RepositoryContextResponse(
        commit_sha=context.commit_sha,
        is_complete=context.is_complete,
        indexed_file_count=len(context.files),
        tree_truncated=context.tree_truncated,
        file_limit_reached=context.file_limit_reached,
        failures=failures,
    )


@router.post(
    "/reviews",
    response_model=ReviewResponse,
    dependencies=[Depends(require_api_key)],
)
async def create_review(
    request: ReviewRequest,
    analyzer: Annotated[
        PullRequestAnalyzer,
        Depends(get_pull_request_analyzer),
    ],
) -> ReviewResponse:
    """Analyze a GitHub pull request and return a structured report."""

    try:
        analysis = await analyzer.analyze(
            owner=request.owner,
            repo=request.repo,
            pull_number=request.pull_number,
        )

        report = build_review_report(analysis)

        return ReviewResponse(
            owner=report.owner,
            repo=report.repo,
            pull_number=report.pull_number,
            title=report.title,
            html_url=report.html_url,
            head_sha=report.head_sha,
            findings=tuple(
                ReviewFindingResponse.model_validate(asdict(finding))
                for finding in report.findings
            ),
            files_analyzed=report.files_analyzed,
            files_with_unavailable_mapping=(report.files_with_unavailable_mapping),
            total_findings=report.total_findings,
            severity_counts=SeverityCountsResponse.model_validate(
                report.severity_counts
            ),
            repository_context=build_repository_context_response(
                analysis.repository_context
            ),
        )

    except GitHubNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GitHub repository or pull request not found.",
        ) from exc

    except GitHubPermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GitHub denied access to this repository or pull request.",
        ) from exc

    except GitHubAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub authentication failed. Check server configuration.",
        ) from exc

    except GitHubRateLimitError as exc:
        headers = {}

        if exc.retry_after is not None:
            headers["Retry-After"] = str(max(1, ceil(exc.retry_after)))

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="GitHub API rate limit exceeded. Try again later.",
            headers=headers,
        ) from exc

    except GitHubError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub API request failed.",
        ) from exc


@router.post(
    "/reviews/publish",
    response_model=ReviewPublishResponse,
    dependencies=[Depends(require_api_key)],
)
async def publish_review(
    request: ReviewPublishRequest,
    analyzer: Annotated[
        PullRequestAnalyzer,
        Depends(get_pull_request_analyzer),
    ],
    github_client: Annotated[
        GitHubClient,
        Depends(get_github_client),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> ReviewPublishResponse:
    """Generate a dry-run review or explicitly publish it to GitHub."""

    # Require administrator permission before doing any analysis.
    if request.publish and not settings.enable_review_publishing:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GitHub review publishing is disabled by the server.",
        )

    try:
        analysis = await analyzer.analyze(
            owner=request.owner,
            repo=request.repo,
            pull_number=request.pull_number,
        )

        report = build_review_report(analysis)

        publisher = ReviewPublisher(
            github_client=github_client,
            dry_run=not request.publish,
        )

        result = await publisher.publish(
            report,
            analysis,
        )

        return ReviewPublishResponse(
            owner=report.owner,
            repo=report.repo,
            pull_number=report.pull_number,
            title=report.title,
            html_url=report.html_url,
            head_sha=report.head_sha,
            status=result.status,
            comment_count=result.comment_count,
            review_body=result.review_body,
            repository_context=build_repository_context_response(
                analysis.repository_context
            ),
            comments=tuple(
                ReviewPublishCommentResponse.model_validate(comment.to_github_payload())
                for comment in result.comments
            ),
        )

    except GitHubNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GitHub repository or pull request not found.",
        ) from exc

    except GitHubPermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GitHub denied access to this repository or pull request.",
        ) from exc

    except GitHubAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub authentication failed. Check server configuration.",
        ) from exc

    except GitHubRateLimitError as exc:
        headers = {}

        if exc.retry_after is not None:
            headers["Retry-After"] = str(max(1, ceil(exc.retry_after)))

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="GitHub API rate limit exceeded. Try again later.",
            headers=headers,
        ) from exc

    except GitHubError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub API request failed.",
        ) from exc
