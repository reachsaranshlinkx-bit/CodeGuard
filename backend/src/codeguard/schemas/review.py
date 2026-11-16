from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class SeverityCountsResponse(BaseModel):
    """Finding counts grouped by severity."""

    model_config = ConfigDict(extra="forbid")

    critical: int = Field(ge=0, strict=True)
    high: int = Field(ge=0, strict=True)
    medium: int = Field(ge=0, strict=True)
    low: int = Field(ge=0, strict=True)
    info: int = Field(ge=0, strict=True)


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
