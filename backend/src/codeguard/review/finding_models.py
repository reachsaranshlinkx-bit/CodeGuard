from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["critical", "high", "medium", "low", "info"]

Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """Metadata describing a CodeGuard security detection rule."""

    rule_id: str
    title: str
    description: str
    severity: Severity
    confidence: Confidence
    cwe: str
    remediation: str
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the rule definition."""

        required_fields = (
            self.rule_id,
            self.title,
            self.description,
            self.cwe,
            self.remediation,
        )

        if any(not value.strip() for value in required_fields):
            raise ValueError("Rule fields must not be empty.")

        if not self.references:
            raise ValueError("At least one reference is required.")

        if any(not reference.strip() for reference in self.references):
            raise ValueError("Rule references must not be empty.")


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """A fully enriched security finding for a code review."""

    fingerprint: str
    rule_id: str
    title: str
    severity: Severity
    confidence: Confidence
    message: str
    description: str
    file_path: str
    line_number: int
    cwe: str
    evidence: str
    remediation: str
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the review finding."""

        required_fields = (
            self.fingerprint,
            self.rule_id,
            self.title,
            self.message,
            self.description,
            self.file_path,
            self.cwe,
            self.evidence,
            self.remediation,
        )

        if any(not value.strip() for value in required_fields):
            raise ValueError("Review finding fields must not be empty.")

        if (
            isinstance(self.line_number, bool)
            or not isinstance(self.line_number, int)
            or self.line_number < 1
        ):
            raise ValueError("Line number must be a positive integer.")

        if not self.references:
            raise ValueError("At least one reference is required.")

        if self.severity not in {"critical", "high", "medium", "low", "info"}:
            raise ValueError("Invalid severity.")

        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("Invalid confidence.")


@dataclass(frozen=True, slots=True)
class ReviewReport:
    """Structured security review report for a GitHub pull request."""

    owner: str
    repo: str
    pull_number: int
    title: str
    html_url: str
    head_sha: str
    findings: tuple[ReviewFinding, ...]
    files_analyzed: int
    files_with_unavailable_mapping: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate report metadata."""

        if not self.owner.strip() or not self.repo.strip():
            raise ValueError("Repository owner and name are required.")

        if (
            isinstance(self.pull_number, bool)
            or not isinstance(self.pull_number, int)
            or self.pull_number < 1
        ):
            raise ValueError("Pull request number must be positive.")

        if not self.title.strip():
            raise ValueError("Report title must not be empty.")

        if not self.html_url.strip() or not self.head_sha.strip():
            raise ValueError("Pull request URL and head SHA are required.")

        if (
            isinstance(self.files_analyzed, bool)
            or not isinstance(self.files_analyzed, int)
            or self.files_analyzed < 0
        ):
            raise ValueError("Files analyzed cannot be negative.")

    @property
    def total_findings(self) -> int:
        """Return the total number of findings."""
        return len(self.findings)

    @property
    def severity_counts(self) -> dict[str, int]:
        """Return finding counts grouped by severity."""

        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        for finding in self.findings:
            counts[finding.severity] += 1

        return counts
