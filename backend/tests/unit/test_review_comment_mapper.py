import pytest

from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.analyzers.python_risk import RiskFinding
from codeguard.review.comment_mapper import (
    build_inline_review_comments,
)
from codeguard.review.finding_models import (
    ReviewFinding,
    ReviewReport,
)
from codeguard.services.pull_request_analyzer import (
    FileAnalysisResult,
    PullRequestAnalysisResult,
)


def make_risk_finding(
    line: int = 12,
    rule_id: str = "PY001",
    path: str = "src/auth.py",
) -> RiskFinding:
    return RiskFinding(
        rule_id=rule_id,
        severity="high",
        message="eval() executes dynamically supplied code.",
        file_path=path,
        lineno=line,
        cwe="CWE-95",
    )


def make_review_finding(
    line: int = 12,
    rule_id: str = "PY001",
    path: str = "src/auth.py",
) -> ReviewFinding:
    return ReviewFinding(
        fingerprint=f"{rule_id}:{path}:{line}",
        rule_id=rule_id,
        title="Dynamic Code Execution",
        severity="high",
        confidence="high",
        message="eval() executes dynamically supplied code.",
        description="Untrusted input can execute Python code.",
        file_path=path,
        line_number=line,
        cwe="CWE-95",
        evidence="eval(username)",
        remediation="Avoid evaluating untrusted input.",
        references=("https://cwe.mitre.org/data/definitions/95.html",),
    )


def make_file_result(
    *,
    path: str = "src/auth.py",
    findings: tuple[RiskFinding, ...] = (),
    added_lines: frozenset[int] | None = frozenset({12}),
    changed_findings: tuple[RiskFinding, ...] | None = (),
    mapping_error: str | None = None,
    analyzed: bool = True,
    previous_filename: str | None = None,
) -> FileAnalysisResult:
    ast_result = (
        PythonAnalysisResult(
            file_path=path,
            functions=(),
            classes=(),
            imports=(),
            calls=(),
        )
        if analyzed
        else None
    )

    return FileAnalysisResult(
        file_path=path,
        status="renamed" if previous_filename else "modified",
        additions=len(added_lines) if added_lines is not None else 0,
        deletions=0,
        changes=len(added_lines) if added_lines is not None else 0,
        previous_filename=previous_filename,
        ast_result=ast_result,
        risk_findings=findings,
        added_line_numbers=added_lines,
        changed_line_findings=changed_findings,
        line_mapping_error=mapping_error,
    )


def make_analysis(
    files: tuple[FileAnalysisResult, ...],
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


def make_report(
    findings: tuple[ReviewFinding, ...],
    unavailable: tuple[str, ...] = (),
) -> ReviewReport:
    return ReviewReport(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        head_sha="abc123",
        findings=findings,
        files_analyzed=1,
        files_with_unavailable_mapping=unavailable,
    )


def test_maps_finding_on_added_line() -> None:
    raw = make_risk_finding(line=12)
    finding = make_review_finding(line=12)

    analysis = make_analysis(
        (
            make_file_result(
                findings=(raw,),
                added_lines=frozenset({12}),
                changed_findings=(raw,),
            ),
        )
    )

    report = make_report((finding,))

    comments = build_inline_review_comments(report, analysis)

    assert len(comments) == 1

    comment = comments[0]

    assert comment.path == "src/auth.py"
    assert comment.line == 12
    assert comment.side == "RIGHT"
    assert "Dynamic Code Execution" in comment.body
    assert "Avoid evaluating untrusted input." in comment.body


def test_excludes_finding_on_unchanged_line() -> None:
    added = make_risk_finding(line=12)
    unchanged = make_risk_finding(line=8, rule_id="PY002")

    analysis = make_analysis(
        (
            make_file_result(
                findings=(added, unchanged),
                added_lines=frozenset({12}),
                changed_findings=(added,),
            ),
        )
    )

    report = make_report(
        (
            make_review_finding(line=12),
            make_review_finding(line=8, rule_id="PY002"),
        )
    )

    comments = build_inline_review_comments(report, analysis)

    assert len(comments) == 1
    assert comments[0].line == 12


def test_skips_findings_when_patch_mapping_is_unavailable() -> None:
    raw = make_risk_finding()
    finding = make_review_finding()

    analysis = make_analysis(
        (
            make_file_result(
                findings=(raw,),
                added_lines=None,
                changed_findings=None,
                mapping_error="patch_unavailable",
            ),
        )
    )

    report = make_report(
        (finding,),
        unavailable=("src/auth.py",),
    )

    assert build_inline_review_comments(report, analysis) == ()


def test_skips_files_with_empty_added_line_set() -> None:
    raw = make_risk_finding()

    analysis = make_analysis(
        (
            make_file_result(
                findings=(raw,),
                added_lines=frozenset(),
                changed_findings=(),
            ),
        )
    )

    report = make_report((make_review_finding(),))

    assert build_inline_review_comments(report, analysis) == ()


def test_skips_files_that_failed_analysis() -> None:
    raw = make_risk_finding()
    finding = make_review_finding()

    analysis = make_analysis(
        (
            make_file_result(
                findings=(raw,),
                added_lines=frozenset({12}),
                changed_findings=(raw,),
                analyzed=False,
            ),
        )
    )

    report = make_report((finding,))

    assert build_inline_review_comments(report, analysis) == ()


def test_uses_new_filename_for_renamed_file() -> None:
    new_path = "src/new_auth.py"
    raw = make_risk_finding(path=new_path)
    finding = make_review_finding(path=new_path)

    analysis = make_analysis(
        (
            make_file_result(
                path=new_path,
                previous_filename="src/old_auth.py",
                findings=(raw,),
                added_lines=frozenset({12}),
                changed_findings=(raw,),
            ),
        )
    )

    report = make_report((finding,))

    comments = build_inline_review_comments(report, analysis)

    assert len(comments) == 1
    assert comments[0].path == new_path


def test_returns_empty_tuple_when_report_has_no_findings() -> None:
    analysis = make_analysis((make_file_result(),))

    report = make_report(())

    assert build_inline_review_comments(report, analysis) == ()


def test_rejects_report_from_different_pull_request() -> None:
    analysis = make_analysis(())
    report = make_report(())

    mismatched_report = ReviewReport(
        owner="someone-else",
        repo="CodeGuard",
        pull_number=42,
        title=report.title,
        html_url=report.html_url,
        head_sha=report.head_sha,
        findings=(),
        files_analyzed=0,
    )

    with pytest.raises(ValueError, match="does not match"):
        build_inline_review_comments(mismatched_report, analysis)


def test_returns_comments_in_deterministic_order() -> None:
    first = make_risk_finding(
        line=20,
        path="src/z_auth.py",
    )
    second = make_risk_finding(
        line=5,
        path="src/a_auth.py",
    )

    first_review = make_review_finding(
        line=20,
        path="src/z_auth.py",
    )
    second_review = make_review_finding(
        line=5,
        path="src/a_auth.py",
    )

    analysis = make_analysis(
        (
            make_file_result(
                path="src/z_auth.py",
                findings=(first,),
                added_lines=frozenset({20}),
                changed_findings=(first,),
            ),
            make_file_result(
                path="src/a_auth.py",
                findings=(second,),
                added_lines=frozenset({5}),
                changed_findings=(second,),
            ),
        )
    )

    report = make_report((first_review, second_review))

    comments = build_inline_review_comments(report, analysis)

    assert [comment.path for comment in comments] == [
        "src/a_auth.py",
        "src/z_auth.py",
    ]
