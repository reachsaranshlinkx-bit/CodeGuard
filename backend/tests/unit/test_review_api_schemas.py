import pytest
from pydantic import ValidationError

from codeguard.schemas.review import (
    RepositoryContextResponse,
    ReviewFindingResponse,
    ReviewRequest,
    ReviewResponse,
    SeverityCountsResponse,
)


def make_finding_payload() -> dict:
    return {
        "fingerprint": "a" * 64,
        "rule_id": "PY001",
        "title": "Dynamic Code Execution",
        "severity": "high",
        "confidence": "medium",
        "message": "Potential unsafe dynamic execution.",
        "description": "Untrusted input may execute Python code.",
        "file_path": "src/auth.py",
        "line_number": 10,
        "cwe": "CWE-95",
        "evidence": "eval(user_input)",
        "remediation": "Avoid evaluating untrusted input.",
        "references": ["https://cwe.mitre.org/data/definitions/95.html"],
    }


def make_report_payload() -> dict:
    finding = make_finding_payload()

    return {
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
        "title": "Improve authentication",
        "html_url": "https://github.com/octocat/CodeGuard/pull/42",
        "head_sha": "abc123",
        "findings": [finding],
        "files_analyzed": 3,
        "files_with_unavailable_mapping": [],
        "total_findings": 1,
        "severity_counts": {
            "critical": 0,
            "high": 1,
            "medium": 0,
            "low": 0,
            "info": 0,
        },
    }


def test_review_request_accepts_valid_data() -> None:
    request = ReviewRequest.model_validate(
        {
            "owner": "octocat",
            "repo": "CodeGuard",
            "pull_number": 42,
        }
    )

    assert request.owner == "octocat"
    assert request.repo == "CodeGuard"
    assert request.pull_number == 42


def test_review_request_strips_whitespace() -> None:
    request = ReviewRequest.model_validate(
        {
            "owner": " octocat ",
            "repo": " CodeGuard ",
            "pull_number": 42,
        }
    )

    assert request.owner == "octocat"
    assert request.repo == "CodeGuard"


@pytest.mark.parametrize("owner", ["", "   "])
def test_review_request_rejects_empty_owner(owner: str) -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(
            owner=owner,
            repo="CodeGuard",
            pull_number=42,
        )


@pytest.mark.parametrize("repo", ["", "   "])
def test_review_request_rejects_empty_repo(repo: str) -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(
            owner="octocat",
            repo=repo,
            pull_number=42,
        )


@pytest.mark.parametrize("pull_number", [0, -1, True])
def test_review_request_rejects_invalid_pull_number(
    pull_number: int,
) -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(
            owner="octocat",
            repo="CodeGuard",
            pull_number=pull_number,
        )


def test_review_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(
            owner="octocat",
            repo="CodeGuard",
            pull_number=42,
            token="secret",
        )


def test_finding_response_accepts_valid_data() -> None:
    finding = ReviewFindingResponse.model_validate(make_finding_payload())

    assert finding.rule_id == "PY001"
    assert finding.severity == "high"
    assert finding.line_number == 10
    assert finding.cwe == "CWE-95"


def test_finding_response_rejects_invalid_line_number() -> None:
    payload = make_finding_payload()
    payload["line_number"] = 0

    with pytest.raises(ValidationError):
        ReviewFindingResponse.model_validate(payload)


def test_finding_response_rejects_invalid_severity() -> None:
    payload = make_finding_payload()
    payload["severity"] = "unknown"

    with pytest.raises(ValidationError):
        ReviewFindingResponse.model_validate(payload)


def test_severity_counts_reject_negative_values() -> None:
    with pytest.raises(ValidationError):
        SeverityCountsResponse(
            critical=-1,
            high=0,
            medium=0,
            low=0,
            info=0,
        )


def test_review_response_accepts_valid_report() -> None:
    report = ReviewResponse.model_validate(make_report_payload())

    assert report.owner == "octocat"
    assert report.total_findings == 1
    assert report.files_analyzed == 3
    assert report.severity_counts.high == 1


def test_review_response_serializes_to_json() -> None:
    report = ReviewResponse.model_validate(make_report_payload())

    data = report.model_dump(mode="json")

    assert data["owner"] == "octocat"
    assert data["findings"][0]["rule_id"] == "PY001"
    assert data["severity_counts"]["high"] == 1


def test_review_response_rejects_inconsistent_finding_count() -> None:
    payload = make_report_payload()
    payload["total_findings"] = 5

    with pytest.raises(
        ValidationError,
        match="total_findings",
    ):
        ReviewResponse.model_validate(payload)


def test_review_response_rejects_inconsistent_severity_counts() -> None:
    payload = make_report_payload()
    payload["severity_counts"]["high"] = 2

    with pytest.raises(
        ValidationError,
        match="severity_counts",
    ):
        ReviewResponse.model_validate(payload)


def test_repository_context_response_accepts_partial_context() -> None:
    context = RepositoryContextResponse.model_validate(
        {
            "commit_sha": "a" * 40,
            "is_complete": False,
            "indexed_file_count": 3,
            "tree_truncated": False,
            "file_limit_reached": False,
            "failures": [
                {
                    "file_path": "src/missing.py",
                    "stage": "retrieval",
                    "error_type": "GitHubNotFoundError",
                }
            ],
        }
    )

    assert context.is_complete is False
    assert context.indexed_file_count == 3
    assert len(context.failures) == 1
    assert context.failures[0].file_path == "src/missing.py"


def test_repository_context_response_rejects_inconsistent_completeness() -> None:
    with pytest.raises(ValidationError, match="is_complete"):
        RepositoryContextResponse.model_validate(
            {
                "commit_sha": "a" * 40,
                "is_complete": True,
                "indexed_file_count": 3,
                "tree_truncated": False,
                "file_limit_reached": False,
                "failures": [
                    {
                        "file_path": "src/missing.py",
                        "stage": "analysis",
                        "error_type": "SyntaxError",
                    }
                ],
            }
        )
