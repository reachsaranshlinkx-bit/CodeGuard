import pytest

from codeguard.ai.contracts import AIFindingCandidate, AIReviewResult
from codeguard.ai.engine import (
    AIReviewExecutionResult,
    AIReviewFindingAssessment,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
from codeguard.ai.evidence_validation import EvidenceLocationReason
from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.analyzers.python_risk import RiskFinding
from codeguard.review.finding_models import ReviewFinding
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
    *,
    ai_review_result: AIReviewExecutionResult | None = None,
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
        ai_review_result=ai_review_result,
    )


def make_ai_candidate(
    *,
    title: str = "Potential unsafe dynamic execution",
    file_path: str = "src/auth.py",
    line_number: int = 10,
    evidence: str = "eval(token)",
) -> AIFindingCandidate:
    return AIFindingCandidate(
        title=title,
        severity="high",
        confidence="medium",
        file_path=file_path,
        line_number=line_number,
        problem="The code executes dynamically supplied input.",
        evidence=evidence,
        why_it_matters="Untrusted input could execute arbitrary code.",
        suggested_fix="Use a safe parser for the expected input format.",
        recommended_validation="Add a test for untrusted input.",
    )


def make_ai_assessment(
    candidate: AIFindingCandidate,
    *,
    status: EvidenceMatchStatus,
    reason: EvidenceMatchReason,
    location_reason: EvidenceLocationReason | None = None,
) -> AIReviewFindingAssessment:
    return AIReviewFindingAssessment(
        candidate=candidate,
        evidence_match=EvidenceMatchResult(
            status=status,
            reason=reason,
            file_path=candidate.file_path,
            line_number=candidate.line_number,
            location_reason=location_reason,
        ),
    )


def make_ai_execution_result(
    summary: str,
    assessments: tuple[AIReviewFindingAssessment, ...] = (),
) -> AIReviewExecutionResult:
    return AIReviewExecutionResult(
        provider_result=AIReviewResult(
            summary=summary,
            findings=tuple(assessment.candidate for assessment in assessments),
        ),
        assessments=assessments,
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


def test_report_defaults_ai_fields_without_changing_static_findings() -> None:
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

    assert report.ai_review_summary is None
    assert report.ai_assessments == ()
    assert report.total_findings == 1
    assert report.findings[0].rule_id == "PY001"
    assert report.findings[0].file_path == "src/auth.py"


def test_report_preserves_ai_summary_when_review_has_no_findings() -> None:
    summary = "The AI review found no additional potential issues."
    analysis = make_analysis(
        ai_review_result=make_ai_execution_result(summary),
    )

    report = build_review_report(analysis)

    assert report.ai_review_summary == summary
    assert report.ai_assessments == ()
    assert report.total_findings == 0


def test_report_preserves_ai_candidates_and_evidence_assessment_order() -> None:
    first_candidate = make_ai_candidate(
        title="First proposed issue",
        file_path="src/first.py",
        line_number=12,
        evidence="unsafe_call(first)",
    )
    second_candidate = make_ai_candidate(
        title="Second proposed issue",
        file_path="src/second.py",
        line_number=20,
        evidence="unsafe_call(second)",
    )
    third_candidate = make_ai_candidate(
        title="Third proposed issue",
        file_path="src/third.py",
        line_number=30,
        evidence="unsafe_call(third)",
    )
    assessments = (
        make_ai_assessment(
            first_candidate,
            status=EvidenceMatchStatus.MATCHED,
            reason=EvidenceMatchReason.EXACT_MATCH,
        ),
        make_ai_assessment(
            second_candidate,
            status=EvidenceMatchStatus.MISMATCHED,
            reason=EvidenceMatchReason.CONTENT_MISMATCH,
            location_reason=EvidenceLocationReason.LOCATION_VALID,
        ),
        make_ai_assessment(
            third_candidate,
            status=EvidenceMatchStatus.UNVERIFIABLE,
            reason=EvidenceMatchReason.LOCATION_UNVERIFIABLE,
            location_reason=EvidenceLocationReason.INCOMPLETE_SNAPSHOT,
        ),
    )
    analysis = make_analysis(
        ai_review_result=make_ai_execution_result(
            "Three potential issues were assessed.",
            assessments,
        ),
    )

    report = build_review_report(analysis)

    assert report.ai_assessments == assessments
    assert tuple(item.candidate for item in report.ai_assessments) == (
        first_candidate,
        second_candidate,
        third_candidate,
    )
    assert tuple(item.evidence_match.status for item in report.ai_assessments) == (
        EvidenceMatchStatus.MATCHED,
        EvidenceMatchStatus.MISMATCHED,
        EvidenceMatchStatus.UNVERIFIABLE,
    )
    assert tuple(item.evidence_match.reason for item in report.ai_assessments) == (
        EvidenceMatchReason.EXACT_MATCH,
        EvidenceMatchReason.CONTENT_MISMATCH,
        EvidenceMatchReason.LOCATION_UNVERIFIABLE,
    )
    assert tuple(
        item.evidence_match.location_reason for item in report.ai_assessments
    ) == (
        None,
        EvidenceLocationReason.LOCATION_VALID,
        EvidenceLocationReason.INCOMPLETE_SNAPSHOT,
    )
    assert report.ai_assessments[1].candidate.title == "Second proposed issue"
    assert report.ai_assessments[1].candidate.evidence == "unsafe_call(second)"
    assert (
        report.ai_assessments[1].candidate.suggested_fix
        == "Use a safe parser for the expected input format."
    )


def test_ai_assessments_do_not_change_static_findings_or_severity_counts() -> None:
    static_finding = make_finding()
    static_analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(static_finding,),
                changed_line_findings=(static_finding,),
            ),
        )
    )
    static_report = build_review_report(static_analysis)

    candidate = make_ai_candidate()
    assessment = make_ai_assessment(
        candidate,
        status=EvidenceMatchStatus.UNVERIFIABLE,
        reason=EvidenceMatchReason.LOCATION_UNVERIFIABLE,
        location_reason=EvidenceLocationReason.SOURCE_OMITTED,
    )
    combined_report = build_review_report(
        make_analysis(
            static_analysis.files,
            ai_review_result=make_ai_execution_result(
                "A candidate could not be verified.",
                (assessment,),
            ),
        )
    )

    assert combined_report.findings == static_report.findings
    assert all(isinstance(item, ReviewFinding) for item in combined_report.findings)
    assert combined_report.total_findings == static_report.total_findings == 1
    assert combined_report.severity_counts == static_report.severity_counts
    assert combined_report.ai_assessments == (assessment,)
    assert combined_report.ai_assessments[0].evidence_match.status is (
        EvidenceMatchStatus.UNVERIFIABLE
    )


def test_combined_report_keeps_static_metadata_and_fingerprint_unchanged() -> None:
    finding = make_finding(line_number=17)
    static_analysis = make_analysis(
        (
            make_file_result(
                risk_findings=(finding,),
                changed_line_findings=(finding,),
            ),
        )
    )
    static_report = build_review_report(static_analysis)

    candidate = make_ai_candidate(
        title="Potential issue alongside static finding",
        file_path="src/other.py",
        line_number=8,
    )
    assessment = make_ai_assessment(
        candidate,
        status=EvidenceMatchStatus.MISMATCHED,
        reason=EvidenceMatchReason.CONTENT_MISMATCH,
        location_reason=EvidenceLocationReason.LOCATION_VALID,
    )
    combined_report = build_review_report(
        make_analysis(
            static_analysis.files,
            ai_review_result=make_ai_execution_result(
                "Static and AI assessments are both present.",
                (assessment,),
            ),
        )
    )

    assert combined_report.ai_review_summary == (
        "Static and AI assessments are both present."
    )
    assert combined_report.ai_assessments == (assessment,)
    assert combined_report.findings == static_report.findings
    assert (
        combined_report.findings[0].fingerprint == static_report.findings[0].fingerprint
    )
    assert combined_report.findings[0].rule_id == "PY001"
    assert combined_report.findings[0].title == "Dynamic Code Execution"
    assert combined_report.findings[0].severity == "high"
    assert combined_report.findings[0].cwe == "CWE-95"
