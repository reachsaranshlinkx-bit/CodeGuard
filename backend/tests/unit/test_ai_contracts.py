import pytest
from pydantic import ValidationError

from codeguard.ai.contracts import (
    AIFindingCandidate,
    AIReviewInput,
    AIReviewResult,
)


def make_candidate_payload() -> dict[str, object]:
    return {
        "title": "Missing authentication check",
        "severity": "high",
        "confidence": "medium",
        "file_path": "src/auth.py",
        "line_number": 25,
        "problem": "The endpoint does not verify user authorization.",
        "evidence": "def get_account(account_id):",
        "why_it_matters": "Unauthorized users may access another account.",
        "suggested_fix": "Check account ownership before returning the account.",
        "recommended_validation": "Add tests for authorized and unauthorized users.",
    }


def test_ai_finding_candidate_accepts_valid_data() -> None:
    candidate = AIFindingCandidate.model_validate(make_candidate_payload())

    assert candidate.title == "Missing authentication check"
    assert candidate.severity == "high"
    assert candidate.line_number == 25


def test_ai_finding_candidate_is_immutable() -> None:
    candidate = AIFindingCandidate.model_validate(make_candidate_payload())

    with pytest.raises(ValidationError):
        candidate.severity = "low"  # type: ignore[misc]


@pytest.mark.parametrize(
    "file_path",
    [
        "/etc/passwd",
        "../secrets.py",
        "src/../secrets.py",
        r"src\auth.py",
        "C:/secrets.py",
        "src/auth.py\x00",
        "src/auth.py\nextra",
    ],
)
def test_ai_finding_candidate_rejects_unsafe_paths(file_path: str) -> None:
    payload = make_candidate_payload()
    payload["file_path"] = file_path

    with pytest.raises(ValidationError, match="repository-relative"):
        AIFindingCandidate.model_validate(payload)


def test_ai_finding_candidate_rejects_invalid_severity() -> None:
    payload = make_candidate_payload()
    payload["severity"] = "blocker"

    with pytest.raises(ValidationError):
        AIFindingCandidate.model_validate(payload)


def test_ai_finding_candidate_rejects_invalid_line_number() -> None:
    payload = make_candidate_payload()
    payload["line_number"] = 0

    with pytest.raises(ValidationError):
        AIFindingCandidate.model_validate(payload)


def test_ai_finding_candidate_rejects_extra_fields() -> None:
    payload = make_candidate_payload()
    payload["rule_id"] = "PY001"

    with pytest.raises(ValidationError, match="Extra inputs"):
        AIFindingCandidate.model_validate(payload)


def test_ai_review_result_accepts_valid_candidates() -> None:
    candidate = AIFindingCandidate.model_validate(make_candidate_payload())

    result = AIReviewResult(
        summary="One potential authorization issue was identified.",
        findings=(candidate,),
    )

    assert len(result.findings) == 1
    assert result.findings[0] == candidate


def test_ai_review_result_accepts_no_findings() -> None:
    result = AIReviewResult(
        summary="No additional potential issues were identified.",
    )

    assert result.findings == ()


def test_ai_review_result_rejects_more_than_fifty_findings() -> None:
    candidate = AIFindingCandidate.model_validate(make_candidate_payload())

    with pytest.raises(ValidationError):
        AIReviewResult(
            summary="Review completed.",
            findings=(candidate,) * 51,
        )


def make_review_input_payload() -> dict[str, object]:
    return {
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
        "head_sha": "abc123",
        "context": "Changed authentication function and related callers.",
    }


def test_ai_review_input_accepts_valid_data() -> None:

    review_input = AIReviewInput.model_validate(make_review_input_payload())

    assert review_input.owner == "octocat"
    assert review_input.repo == "CodeGuard"
    assert review_input.pull_number == 42


def test_ai_review_input_rejects_invalid_pull_number() -> None:

    payload = make_review_input_payload()
    payload["pull_number"] = 0

    with pytest.raises(ValidationError):
        AIReviewInput.model_validate(payload)


def test_ai_review_input_rejects_empty_context() -> None:

    payload = make_review_input_payload()
    payload["context"] = "   "

    with pytest.raises(ValidationError):
        AIReviewInput.model_validate(payload)


def test_ai_review_input_rejects_oversized_context() -> None:

    payload = make_review_input_payload()
    payload["context"] = "x" * 50_001

    with pytest.raises(ValidationError):
        AIReviewInput.model_validate(payload)
