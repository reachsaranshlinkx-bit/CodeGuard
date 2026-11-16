from dataclasses import FrozenInstanceError

import pytest

from codeguard.review.finding_models import (
    ReviewFinding,
    ReviewReport,
)


def make_finding(
    severity: str = "high",
    fingerprint: str = "abc123",
    line_number: int = 10,
) -> ReviewFinding:
    return ReviewFinding(
        fingerprint=fingerprint,
        rule_id="PY001",
        title="Dynamic Code Execution",
        severity=severity,
        confidence="high",
        message="Potential unsafe dynamic execution.",
        description="Untrusted input may execute Python code.",
        file_path="src/auth.py",
        line_number=line_number,
        cwe="CWE-95",
        evidence="eval(user_input)",
        remediation="Avoid evaluating untrusted input.",
        references=("https://cwe.mitre.org/data/definitions/95.html",),
    )


def make_report(
    findings: tuple[ReviewFinding, ...] = (),
) -> ReviewReport:
    return ReviewReport(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        head_sha="abc123",
        findings=findings,
        files_analyzed=3,
    )


def test_review_finding_accepts_valid_data() -> None:
    finding = make_finding()

    assert finding.rule_id == "PY001"
    assert finding.severity == "high"
    assert finding.line_number == 10


def test_review_finding_is_immutable() -> None:
    finding = make_finding()

    with pytest.raises(FrozenInstanceError):
        finding.severity = "low"  # type: ignore[misc]


@pytest.mark.parametrize("line_number", [0, -1, True])
def test_review_finding_rejects_invalid_line_number(
    line_number: int,
) -> None:
    with pytest.raises(ValueError, match="Line number"):
        make_finding(line_number=line_number)


def test_review_finding_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError, match="severity"):
        make_finding(severity="unknown")


def test_review_report_accepts_valid_data() -> None:
    report = make_report((make_finding(),))

    assert report.owner == "octocat"
    assert report.pull_number == 42
    assert report.total_findings == 1


def test_review_report_counts_findings_by_severity() -> None:
    findings = (
        make_finding("high", "a"),
        make_finding("high", "b"),
        make_finding("medium", "c"),
        make_finding("low", "d"),
    )

    report = make_report(findings)

    assert report.severity_counts == {
        "critical": 0,
        "high": 2,
        "medium": 1,
        "low": 1,
        "info": 0,
    }


def test_empty_review_report_has_zero_findings() -> None:
    report = make_report()

    assert report.total_findings == 0
    assert sum(report.severity_counts.values()) == 0


def test_review_report_rejects_invalid_pull_number() -> None:
    with pytest.raises(ValueError, match="Pull request number"):
        ReviewReport(
            owner="octocat",
            repo="CodeGuard",
            pull_number=0,
            title="Test",
            html_url="https://example.com",
            head_sha="abc123",
            findings=(),
            files_analyzed=0,
        )


def test_review_report_is_immutable() -> None:
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.title = "Changed title"  # type: ignore[misc]
