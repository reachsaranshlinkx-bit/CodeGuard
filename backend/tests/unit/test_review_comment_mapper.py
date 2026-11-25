from dataclasses import replace

import pytest

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.engine import AIReviewFindingAssessment
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
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


def make_ai_candidate(
    *,
    title: str = "Potential unsafe dynamic execution",
    path: str = "src/auth.py",
    line: int = 12,
    problem: str = "The code executes dynamically supplied input.",
    evidence: str = "eval(username)",
    why_it_matters: str = "Untrusted input could execute arbitrary code.",
    suggested_fix: str = "Use a safe parser for the expected input format.",
    recommended_validation: str = "Add a test for untrusted input.",
) -> AIFindingCandidate:
    return AIFindingCandidate(
        title=title,
        severity="high",
        confidence="medium",
        file_path=path,
        line_number=line,
        problem=problem,
        evidence=evidence,
        why_it_matters=why_it_matters,
        suggested_fix=suggested_fix,
        recommended_validation=recommended_validation,
    )


def make_ai_assessment(
    candidate: AIFindingCandidate | None = None,
    *,
    status: EvidenceMatchStatus = EvidenceMatchStatus.MATCHED,
    reason: EvidenceMatchReason = EvidenceMatchReason.EXACT_MATCH,
    evidence_path: str | None = None,
    evidence_line: int | None = None,
) -> AIReviewFindingAssessment:
    candidate = candidate or make_ai_candidate()

    return AIReviewFindingAssessment(
        candidate=candidate,
        evidence_match=EvidenceMatchResult(
            status=status,
            reason=reason,
            file_path=evidence_path or candidate.file_path,
            line_number=(
                evidence_line if evidence_line is not None else candidate.line_number
            ),
        ),
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
    ai_assessments: tuple[AIReviewFindingAssessment, ...] = (),
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
        ai_assessments=ai_assessments,
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


def test_maps_matched_ai_assessment_on_an_added_line() -> None:
    assessment = make_ai_assessment()
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=frozenset({12}),
                changed_findings=(),
            ),
        )
    )

    comments = build_inline_review_comments(
        make_report((), ai_assessments=(assessment,)),
        analysis,
    )

    assert len(comments) == 1
    assert comments[0].path == "src/auth.py"
    assert comments[0].line == 12


def test_ai_comment_contains_actionable_candidate_information() -> None:
    candidate = make_ai_candidate(
        title="Unsafe dynamic code execution",
        problem="User input reaches eval().",
        evidence="eval(username)",
        why_it_matters="An attacker could execute arbitrary code.",
        suggested_fix="Replace eval() with a constrained parser.",
        recommended_validation="Test that executable input is rejected.",
    )
    assessment = make_ai_assessment(candidate)
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=frozenset({12}),
                changed_findings=(),
            ),
        )
    )

    comments = build_inline_review_comments(
        make_report((), ai_assessments=(assessment,)),
        analysis,
    )

    assert len(comments) == 1
    body = comments[0].body

    for detail in (
        candidate.title,
        candidate.severity,
        candidate.confidence,
        candidate.problem,
        candidate.why_it_matters,
        candidate.evidence,
        candidate.suggested_fix,
        candidate.recommended_validation,
    ):
        assert detail in body


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            EvidenceMatchStatus.MISMATCHED,
            EvidenceMatchReason.EXACT_MATCH,
        ),
        (
            EvidenceMatchStatus.UNVERIFIABLE,
            EvidenceMatchReason.LOCATION_UNVERIFIABLE,
        ),
    ],
)
def test_excludes_ai_assessments_without_matched_evidence(
    status: EvidenceMatchStatus,
    reason: EvidenceMatchReason,
) -> None:
    assessment = make_ai_assessment(status=status, reason=reason)
    analysis = make_analysis((make_file_result(changed_findings=()),))

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_assessment_when_evidence_reason_is_not_exact_match() -> None:
    assessment = make_ai_assessment(
        status=EvidenceMatchStatus.MATCHED,
        reason=EvidenceMatchReason.CONTENT_MISMATCH,
    )
    analysis = make_analysis((make_file_result(changed_findings=()),))

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_matched_ai_evidence_on_an_unchanged_line() -> None:
    candidate = make_ai_candidate(line=8)
    assessment = make_ai_assessment(candidate)
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=frozenset({12}),
                changed_findings=(),
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_assessment_when_added_line_mapping_is_missing() -> None:
    assessment = make_ai_assessment()
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=None,
                changed_findings=(),
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_assessment_when_changed_line_mapping_is_missing() -> None:
    assessment = make_ai_assessment()
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=frozenset({12}),
                changed_findings=None,
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_assessment_when_line_mapping_has_an_error() -> None:
    assessment = make_ai_assessment()
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=frozenset({12}),
                changed_findings=(),
                mapping_error="patch_unavailable",
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_assessment_when_ast_analysis_failed() -> None:
    assessment = make_ai_assessment()
    analysis = make_analysis(
        (
            make_file_result(
                added_lines=frozenset({12}),
                changed_findings=(),
                analyzed=False,
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_candidate_that_does_not_match_current_pr_file() -> None:
    candidate = make_ai_candidate(path="src/other.py")
    assessment = make_ai_assessment(candidate)
    analysis = make_analysis(
        (
            make_file_result(
                path="src/auth.py",
                added_lines=frozenset({12}),
                changed_findings=(),
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_excludes_ai_candidate_without_a_matching_pr_file_result() -> None:
    assessment = make_ai_assessment()
    analysis = make_analysis(
        (
            make_file_result(
                path="src/other.py",
                added_lines=frozenset({12}),
                changed_findings=(),
            ),
        )
    )

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("evidence_path", "evidence_line"),
    [
        ("src/other.py", 12),
        ("src/auth.py", 11),
    ],
)
def test_excludes_ai_assessment_when_evidence_location_disagrees_with_candidate(
    evidence_path: str,
    evidence_line: int,
) -> None:
    assessment = make_ai_assessment(
        evidence_path=evidence_path,
        evidence_line=evidence_line,
    )
    analysis = make_analysis((make_file_result(changed_findings=()),))

    assert (
        build_inline_review_comments(
            make_report((), ai_assessments=(assessment,)),
            analysis,
        )
        == ()
    )


def test_uses_current_filename_for_renamed_ai_candidate() -> None:
    new_path = "src/new_auth.py"
    candidate = make_ai_candidate(path=new_path)
    assessment = make_ai_assessment(candidate)
    analysis = make_analysis(
        (
            make_file_result(
                path=new_path,
                previous_filename="src/old_auth.py",
                added_lines=frozenset({12}),
                changed_findings=(),
            ),
        )
    )

    comments = build_inline_review_comments(
        make_report((), ai_assessments=(assessment,)),
        analysis,
    )

    assert len(comments) == 1
    assert comments[0].path == new_path


def test_processes_ai_assessments_deterministically() -> None:
    later_candidate = make_ai_candidate(
        title="Later path candidate",
        path="src/z_auth.py",
        line=20,
    )
    earlier_candidate = make_ai_candidate(
        title="Earlier path candidate",
        path="src/a_auth.py",
        line=5,
    )
    later_assessment = make_ai_assessment(later_candidate)
    earlier_assessment = make_ai_assessment(earlier_candidate)
    analysis = make_analysis(
        (
            make_file_result(
                path="src/z_auth.py",
                added_lines=frozenset({20}),
                changed_findings=(),
            ),
            make_file_result(
                path="src/a_auth.py",
                added_lines=frozenset({5}),
                changed_findings=(),
            ),
        )
    )

    first = build_inline_review_comments(
        make_report(
            (),
            ai_assessments=(later_assessment, earlier_assessment),
        ),
        analysis,
    )
    second = build_inline_review_comments(
        make_report(
            (),
            ai_assessments=(earlier_assessment, later_assessment),
        ),
        analysis,
    )

    assert first == second
    assert [comment.path for comment in first] == [
        "src/a_auth.py",
        "src/z_auth.py",
    ]


def test_ai_assessments_do_not_change_existing_static_comment() -> None:
    raw = make_risk_finding(line=12)
    static_finding = make_review_finding(line=12)
    candidate = make_ai_candidate(line=20)
    assessment = make_ai_assessment(candidate)
    analysis = make_analysis(
        (
            make_file_result(
                findings=(raw,),
                added_lines=frozenset({12, 20}),
                changed_findings=(raw,),
            ),
        )
    )
    static_comments = build_inline_review_comments(
        make_report((static_finding,)),
        analysis,
    )
    combined_comments = build_inline_review_comments(
        make_report((static_finding,), ai_assessments=(assessment,)),
        analysis,
    )

    matching_static_comments = tuple(
        comment
        for comment in combined_comments
        if comment.path == static_comments[0].path
        and comment.line == static_comments[0].line
    )

    assert matching_static_comments == static_comments


def test_static_and_ai_comments_can_coexist() -> None:
    raw = make_risk_finding(line=12)
    static_finding = make_review_finding(line=12)
    candidate = make_ai_candidate(line=20)
    assessment = make_ai_assessment(candidate)
    analysis = make_analysis(
        (
            make_file_result(
                findings=(raw,),
                added_lines=frozenset({12, 20}),
                changed_findings=(raw,),
            ),
        )
    )

    comments = build_inline_review_comments(
        make_report((static_finding,), ai_assessments=(assessment,)),
        analysis,
    )

    assert [(comment.path, comment.line) for comment in comments] == [
        ("src/auth.py", 12),
        ("src/auth.py", 20),
    ]
    assert "Dynamic Code Execution" in comments[0].body
    assert candidate.title in comments[1].body


@pytest.mark.parametrize(
    ("identity_field", "mismatched_value"),
    [
        ("repo", "DifferentRepo"),
        ("pull_number", 43),
        ("head_sha", "different-sha"),
    ],
)
def test_rejects_report_with_mismatched_analysis_identity(
    identity_field: str,
    mismatched_value: str | int,
) -> None:
    analysis = make_analysis(())
    report = make_report(())
    mismatched_report = replace(
        report,
        **{identity_field: mismatched_value},
    )

    with pytest.raises(ValueError, match="does not match"):
        build_inline_review_comments(mismatched_report, analysis)
