from dataclasses import asdict
from math import ceil
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from codeguard.api.dependencies import get_pull_request_analyzer
from codeguard.github.exceptions import (
    GitHubAuthenticationError,
    GitHubError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRateLimitError,
)
from codeguard.review.report_builder import build_review_report
from codeguard.schemas.review import (
    ReviewFindingResponse,
    ReviewRequest,
    ReviewResponse,
    SeverityCountsResponse,
)
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer

router = APIRouter(
    prefix="/api/v1",
    tags=["reviews"],
)


@router.post(
    "/reviews",
    response_model=ReviewResponse,
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
