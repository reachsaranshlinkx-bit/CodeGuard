import pytest

from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.analyzers.python_risk import RiskFinding
from codeguard.review.report_builder import build_review_report
from codeguard.services.pull_request_analyzer import (
    FileAnalysisResult,
    PullRequestAnalysisResult,
)


def make_finding(
    line_number: int = 10,
    rule_id: str = "PY001",
    file_path: str = "src/auth.py",
) -> RiskFinding:
    return RiskFinding(
        rule_id=rule_id,
        severity="high",
        message="eval() executes dynamically supplied code.",
        file_path=file_path,
        lineno=line_number,
        cwe="CWE-95",
    )


def make_file_result(
    *,
    file_path: str = "src/auth.py",
    risk_findings: tuple[RiskFinding, ...] = (),
    changed_line_findings: tuple[RiskFinding, ...] | None = (),
    line_mapping_error: str | None = None,
    analyzed: bool = True,
    skip_reason: str | None = None,
    error: str | None = None,
) -> FileAnalysisResult:
    ast_result = (
        PythonAnalysisResult(
            file_path=file_path,
            functions=(),
            classes=(),
            imports=(),
            calls=(),
        )
        if analyzed
        else None
    )

    return FileAnalysisResult(
        file_path=file_path,
        status="modified",
        additions=1,
        deletions=0,
        changes=1,
        previous_filename=None,
        ast_result=ast_result,
        risk_findings=risk_findings,
        added_line_numbers=frozenset(),
        changed_line_findings=changed_line_findings,
        line_mapping_error=line_mapping_error,
        skip_reason=skip_reason,
        error=error,
    )


def make_analysis(
    files: tuple[FileAnalysisResult, ...] = (),
) -> PullRequestAnalysisResult:
    return PullRequestAnalysisResult(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        state="open",
        head_sha="abc123",
        base_sha="def456",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        files=files,
    )


def test_build_report_enriches_findings_with_rule_metadata() -> None:
    finding = make_finding()

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(finding,),
                changed_line_findings=(finding,),
            ),
        )
    )

    report = build_review_report(analysis)

    assert report.total_findings == 1

    result = report.findings[0]

    assert result.rule_id == "PY001"
    assert result.title == "Dynamic Code Execution"
    assert result.severity == "high"
    assert result.confidence == "medium"
    assert result.cwe == "CWE-95"
    assert result.file_path == "src/auth.py"
    assert result.line_number == 10
    assert result.remediation.strip()
    assert result.references


def test_report_uses_only_changed_line_findings() -> None:
    unchanged = make_finding(5)
    changed = make_finding(10)

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(unchanged, changed),
                changed_line_findings=(changed,),
            ),
        )
    )

    report = build_review_report(analysis)

    assert report.total_findings == 1
    assert report.findings[0].line_number == 10


def test_report_preserves_findings_when_mapping_unavailable() -> None:
    finding = make_finding()

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(finding,),
                changed_line_findings=None,
                line_mapping_error="patch_unavailable",
            ),
        )
    )

    report = build_review_report(analysis)

    assert report.total_findings == 1
    assert report.findings[0].line_number == 10
    assert report.files_with_unavailable_mapping == ("src/auth.py",)


def test_valid_empty_mapping_does_not_include_full_file_findings() -> None:
    finding = make_finding()

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(finding,),
                changed_line_findings=(),
            ),
        )
    )

    report = build_review_report(analysis)

    assert report.total_findings == 0
    assert report.files_with_unavailable_mapping == ()


def test_fingerprints_are_stable() -> None:
    finding = make_finding()

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(finding,),
                changed_line_findings=(finding,),
            ),
        )
    )

    first = build_review_report(analysis)
    second = build_review_report(analysis)

    assert first.findings[0].fingerprint == (second.findings[0].fingerprint)


def test_different_locations_have_different_fingerprints() -> None:
    first = make_finding(line_number=10)
    second = make_finding(line_number=20)

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(first, second),
                changed_line_findings=(first, second),
            ),
        )
    )

    report = build_review_report(analysis)

    assert report.findings[0].fingerprint != report.findings[1].fingerprint


def test_report_counts_only_successfully_analyzed_files() -> None:
    finding = make_finding()

    analyzed_file = make_file_result(
        risk_findings=(finding,),
        changed_line_findings=(finding,),
    )

    skipped_file = make_file_result(
        file_path="README.md",
        analyzed=False,
        skip_reason="unsupported_file_type",
    )

    failed_file = make_file_result(
        file_path="src/broken.py",
        analyzed=False,
        error="GitHubNotFoundError",
    )

    analysis = make_analysis((analyzed_file, skipped_file, failed_file))

    report = build_review_report(analysis)

    assert report.files_analyzed == 1


def test_unknown_rule_id_raises_value_error() -> None:
    finding = make_finding(rule_id="PY999")

    analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(finding,),
                changed_line_findings=(finding,),
            ),
        )
    )

    with pytest.raises(ValueError, match="Unknown rule"):
        build_review_report(analysis)


def test_empty_pull_request_creates_empty_report() -> None:
    report = build_review_report(make_analysis())

    assert report.total_findings == 0
    assert report.files_analyzed == 0
    assert report.severity_counts["high"] == 0
