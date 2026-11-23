from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AISeverity = Literal["critical", "high", "medium", "low", "info"]
AIConfidence = Literal["low", "medium", "high"]


class AIReviewInput(BaseModel):
    """
    Bounded input supplied to an AI review provider.

    Repository content is untrusted input and must be treated
    as data rather than executable instructions.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    owner: str = Field(min_length=1, max_length=100)

    repo: str = Field(min_length=1, max_length=100)

    pull_number: int = Field(gt=0, strict=True)

    head_sha: str = Field(min_length=1, max_length=64)

    context: str = Field(
        min_length=1,
        max_length=50_000,
        description="Bounded repository and pull request context in characters.",
    )


class AIFindingCandidate(BaseModel):
    """
    An unverified finding proposed by an AI review provider.

    This model represents a candidate, not a confirmed security issue.
    Evidence validation must happen separately.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    title: str = Field(min_length=1, max_length=200)

    severity: AISeverity
    confidence: AIConfidence

    file_path: str = Field(
        min_length=1,
        max_length=500,
        description="Repository-relative POSIX file path.",
    )

    line_number: int = Field(
        gt=0,
        strict=True,
        description="AI-proposed source line; requires independent verification.",
    )

    problem: str = Field(min_length=1, max_length=3000)

    evidence: str = Field(
        min_length=1,
        max_length=4000,
        description="Exact source-code excerpt supporting the proposed finding.",
    )

    why_it_matters: str = Field(min_length=1, max_length=2000)

    suggested_fix: str = Field(min_length=1, max_length=3000)

    recommended_validation: str = Field(min_length=1, max_length=2000)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        """Reject absolute paths, traversal and non-POSIX separators."""

        invalid_path = (
            "\\" in value
            or value.startswith("/")
            or (len(value) > 1 and value[1] == ":")
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        )

        if invalid_path:
            raise ValueError("file_path must be a safe repository-relative POSIX path.")

        return value


class AIReviewResult(BaseModel):
    """
    Structured output returned by an AI review provider.

    Findings remain unverified until they pass independent validation.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    summary: str = Field(min_length=1, max_length=4000)

    findings: tuple[AIFindingCandidate, ...] = Field(
        default=(),
        max_length=50,
        description="Bounded collection of AI-proposed findings.",
    )
