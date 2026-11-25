from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from codeguard.ai.contracts import (
    AIConfidence,
    AIFindingCandidate,
    AISeverity,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchStatus,
)
from codeguard.ai.evidence_validation import EvidenceLocationReason
from codeguard.review.finding_models import Confidence, Severity


class ReviewRequest(BaseModel):
    """Request payload for a GitHub pull request review."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    pull_number: int = Field(gt=0, strict=True)


class ReviewFindingResponse(BaseModel):
    """API representation of one structured security finding."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    rule_id: str = Field(min_length=1)
    title: str = Field(min_length=1)

    severity: Severity
    confidence: Confidence

    message: str = Field(min_length=1)
    description: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    line_number: int = Field(gt=0, strict=True)
    cwe: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    remediation: str = Field(min_length=1)

    references: tuple[str, ...] = Field(min_length=1)


class AIFindingCandidateResponse(BaseModel):
    """API representation of one AI-proposed finding candidate."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str = Field(min_length=1, max_length=200)
    severity: AISeverity
    confidence: AIConfidence
    file_path: str = Field(min_length=1, max_length=500)
    line_number: int = Field(gt=0, strict=True)
    problem: str = Field(min_length=1, max_length=3000)
    evidence: Annotated[
        str,
        StringConstraints(strip_whitespace=False),
    ] = Field(min_length=1, max_length=4000)
    why_it_matters: str = Field(min_length=1, max_length=2000)
    suggested_fix: str = Field(min_length=1, max_length=3000)
    recommended_validation: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_candidate_contract(self) -> Self:
        """Apply the domain candidate's path and evidence validation."""

        AIFindingCandidate.model_validate(self.model_dump(mode="python"))
        return self


class EvidenceMatchResponse(BaseModel):
    """API representation of an AI candidate's evidence assessment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    status: EvidenceMatchStatus
    reason: EvidenceMatchReason
    file_path: str = Field(min_length=1)
    line_number: int = Field(gt=0, strict=True)
    location_reason: EvidenceLocationReason | None = None


class AIReviewAssessmentResponse(BaseModel):
    """API representation of an AI candidate and its evidence assessment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    candidate: AIFindingCandidateResponse
    evidence_match: EvidenceMatchResponse


class SeverityCountsResponse(BaseModel):
    """Finding counts grouped by severity."""

    model_config = ConfigDict(extra="forbid")

    critical: int = Field(ge=0, strict=True)
    high: int = Field(ge=0, strict=True)
    medium: int = Field(ge=0, strict=True)
    low: int = Field(ge=0, strict=True)
    info: int = Field(ge=0, strict=True)


class RepositoryFileFailureResponse(BaseModel):
    """Safe API representation of a repository file processing failure."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    file_path: str = Field(min_length=1)
    stage: Literal["retrieval", "analysis"]
    error_type: str = Field(min_length=1)


class RepositoryContextResponse(BaseModel):
    """API representation of repository context completeness."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    commit_sha: str = Field(min_length=1)
    is_complete: bool = Field(strict=True)
    indexed_file_count: int = Field(ge=0, strict=True)
    tree_truncated: bool = Field(strict=True)
    file_limit_reached: bool = Field(strict=True)
    failures: tuple[RepositoryFileFailureResponse, ...] = ()

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        """Ensure completeness agrees with repository processing metadata."""

        expected_complete = (
            not self.tree_truncated
            and not self.file_limit_reached
            and not self.failures
        )

        if self.is_complete != expected_complete:
            raise ValueError(
                "is_complete must match repository context completeness metadata."
            )

        return self


class ReviewResponse(BaseModel):
    """Structured security review API response."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    pull_number: int = Field(gt=0, strict=True)

    title: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    head_sha: str = Field(min_length=1)

    findings: tuple[ReviewFindingResponse, ...]
    files_analyzed: int = Field(ge=0, strict=True)

    files_with_unavailable_mapping: tuple[str, ...] = ()

    total_findings: int = Field(ge=0, strict=True)

    severity_counts: SeverityCountsResponse
    repository_context: RepositoryContextResponse | None = None
    ai_review_summary: str | None = None
    ai_assessments: tuple[AIReviewAssessmentResponse, ...] = ()

    @model_validator(mode="after")
    def validate_report_counts(self) -> Self:
        """Ensure report totals agree with the actual findings."""

        if self.total_findings != len(self.findings):
            raise ValueError("total_findings must equal the number of findings.")

        expected_counts = {
            severity: sum(finding.severity == severity for finding in self.findings)
            for severity in (
                "critical",
                "high",
                "medium",
                "low",
                "info",
            )
        }

        actual_counts = self.severity_counts.model_dump()

        if actual_counts != expected_counts:
            raise ValueError("severity_counts must match the actual findings.")

        return self


class ReviewPublishRequest(ReviewRequest):
    """Request to generate or publish a GitHub pull request review."""

    publish: bool = Field(default=False, strict=True)


class ReviewPublishCommentResponse(BaseModel):
    """API representation of an inline GitHub review comment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    path: str = Field(min_length=1)
    line: int = Field(gt=0, strict=True)
    side: Literal["RIGHT"] = "RIGHT"
    body: str = Field(min_length=1)


class ReviewPublishResponse(BaseModel):
    """Response returned after generating or publishing a review."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    pull_number: int = Field(gt=0, strict=True)

    title: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    head_sha: str = Field(min_length=1)

    status: Literal[
        "dry_run",
        "skipped",
        "published",
        "duplicate",
    ]

    comment_count: int = Field(ge=0, strict=True)
    review_body: str = Field(min_length=1)

    comments: tuple[ReviewPublishCommentResponse, ...]
    repository_context: RepositoryContextResponse | None = None
