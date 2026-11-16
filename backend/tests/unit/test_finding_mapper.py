from codeguard.analyzers.python_risk import RiskFinding
from codeguard.review.finding_mapper import (
    filter_findings_to_added_lines,
)


def make_finding(lineno: int, rule_id: str = "PY001") -> RiskFinding:
    return RiskFinding(
        rule_id=rule_id,
        severity="high",
        message="Potential security vulnerability.",
        file_path="src/auth.py",
        lineno=lineno,
        cwe="CWE-95",
    )


def test_keeps_only_findings_on_added_lines() -> None:
    findings = (
        make_finding(12),
        make_finding(25, "PY003"),
        make_finding(40),
        make_finding(45, "PY004"),
    )

    result = filter_findings_to_added_lines(
        findings,
        frozenset({25, 45}),
    )

    assert result == (findings[1], findings[3])


def test_returns_empty_tuple_when_no_lines_were_added() -> None:
    findings = (
        make_finding(12),
        make_finding(25),
    )

    result = filter_findings_to_added_lines(
        findings,
        frozenset(),
    )

    assert result == ()


def test_returns_empty_tuple_when_there_are_no_findings() -> None:
    result = filter_findings_to_added_lines(
        (),
        frozenset({12, 25}),
    )

    assert result == ()


def test_preserves_finding_order() -> None:
    findings = (
        make_finding(45, "PY004"),
        make_finding(25, "PY003"),
        make_finding(12, "PY001"),
    )

    result = filter_findings_to_added_lines(
        findings,
        frozenset({12, 25, 45}),
    )

    assert result == findings


def test_preserves_multiple_findings_on_same_added_line() -> None:
    findings = (
        make_finding(25, "PY001"),
        make_finding(25, "PY003"),
        make_finding(40, "PY004"),
    )

    result = filter_findings_to_added_lines(
        findings,
        frozenset({25}),
    )

    assert result == (findings[0], findings[1])


def test_does_not_modify_original_findings() -> None:
    findings = (
        make_finding(12),
        make_finding(25, "PY003"),
    )

    original = findings

    filter_findings_to_added_lines(
        findings,
        frozenset({25}),
    )

    assert findings == original
