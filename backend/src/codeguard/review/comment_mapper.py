from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchStatus,
)
from codeguard.review.comment_models import InlineReviewComment
from codeguard.review.finding_models import (
    ReviewFinding,
    ReviewReport,
)
from codeguard.services.pull_request_analyzer import (
    PullRequestAnalysisResult,
)


def _build_comment_body(finding: ReviewFinding) -> str:
    """Build a useful and actionable GitHub security comment."""

    return (
        f"**{finding.title}** ({finding.rule_id})\n\n"
        f"**Severity:** {finding.severity}\n\n"
        f"{finding.message}\n\n"
        f"**Evidence:**\n"
        f"```python\n{finding.evidence}\n```\n\n"
        f"**Recommended remediation:**\n"
        f"{finding.remediation}\n\n"
        f"**CWE:** {finding.cwe}"
    )


def _build_ai_comment_body(candidate: AIFindingCandidate) -> str:
    """Build an actionable comment that keeps an AI candidate unconfirmed."""

    fence = "```"
    while fence in candidate.evidence:
        fence += "`"

    return (
        "**AI review candidate (not a confirmed vulnerability)**\n\n"
        f"**{candidate.title}**\n\n"
        f"**Severity:** {candidate.severity} | "
        f"**Confidence:** {candidate.confidence}\n\n"
        f"**Problem:**\n{candidate.problem}\n\n"
        f"**Why it matters:**\n{candidate.why_it_matters}\n\n"
        "**Evidence excerpt (matched to source; does not confirm the issue):**\n"
        f"{fence}python\n{candidate.evidence}\n{fence}\n\n"
        f"**Suggested fix:**\n{candidate.suggested_fix}\n\n"
        f"**Recommended validation:**\n{candidate.recommended_validation}"
    )


def build_inline_review_comments(
    report: ReviewReport,
    analysis: PullRequestAnalysisResult,
) -> tuple[InlineReviewComment, ...]:
    """Map valid added-line security findings to GitHub comments.

    Findings are published inline only when:
    - The report belongs to the same PR and head commit.
    - Python analysis completed successfully.
    - Diff-line mapping is available.
    - The finding belongs to a newly added line.
    """

    if (
        report.owner != analysis.owner
        or report.repo != analysis.repo
        or report.pull_number != analysis.pull_number
        or report.head_sha != analysis.head_sha
    ):
        raise ValueError("Review report does not match pull request analysis.")

    # Build a lookup of findings that are safe to publish inline.
    eligible_findings: dict[
        str,
        set[tuple[str, int]],
    ] = {}

    for file_result in analysis.files:
        if file_result.ast_result is None:
            continue

        if file_result.added_line_numbers is None:
            continue

        if file_result.changed_line_findings is None:
            continue

        if file_result.line_mapping_error is not None:
            continue

        added_lines = file_result.added_line_numbers

        locations = {
            (finding.rule_id, finding.lineno)
            for finding in file_result.changed_line_findings
            if (
                finding.file_path == file_result.file_path
                and finding.lineno in added_lines
            )
        }

        if locations:
            eligible_findings[file_result.file_path] = locations

    # Convert only validated report findings into inline comments.
    comments: list[tuple[str, int, tuple[str, ...], InlineReviewComment]] = []

    ordered_findings = sorted(
        report.findings,
        key=lambda finding: (
            finding.file_path,
            finding.line_number,
            finding.rule_id,
            finding.fingerprint,
        ),
    )

    for finding in ordered_findings:
        valid_locations = eligible_findings.get(
            finding.file_path,
            set(),
        )

        if (
            finding.rule_id,
            finding.line_number,
        ) not in valid_locations:
            continue

        comment = InlineReviewComment(
            path=finding.file_path,
            line=finding.line_number,
            body=_build_comment_body(finding),
        )
        comments.append(
            (
                finding.file_path,
                finding.line_number,
                ("0-static", finding.rule_id, finding.fingerprint),
                comment,
            )
        )

    for assessment in report.ai_assessments:
        candidate = assessment.candidate
        evidence_match = assessment.evidence_match

        if (
            evidence_match.status != EvidenceMatchStatus.MATCHED
            or evidence_match.reason != EvidenceMatchReason.EXACT_MATCH
            or evidence_match.file_path != candidate.file_path
            or evidence_match.line_number != candidate.line_number
        ):
            continue

        matching_file = None

        for file_result in analysis.files:
            if file_result.file_path != candidate.file_path:
                continue

            if (
                file_result.ast_result is None
                or file_result.added_line_numbers is None
                or file_result.changed_line_findings is None
                or file_result.line_mapping_error is not None
                or candidate.line_number not in file_result.added_line_numbers
            ):
                continue

            matching_file = file_result
            break

        if matching_file is None:
            continue

        comment = InlineReviewComment(
            path=matching_file.file_path,
            line=candidate.line_number,
            body=_build_ai_comment_body(candidate),
        )
        comments.append(
            (
                matching_file.file_path,
                candidate.line_number,
                (
                    "1-ai",
                    candidate.title,
                    candidate.severity,
                    candidate.confidence,
                    candidate.problem,
                    candidate.why_it_matters,
                    candidate.evidence,
                    candidate.suggested_fix,
                    candidate.recommended_validation,
                ),
                comment,
            )
        )

    ordered_comments = sorted(
        comments,
        key=lambda item: (item[0], item[1], item[2]),
    )

    return tuple(item[3] for item in ordered_comments)
