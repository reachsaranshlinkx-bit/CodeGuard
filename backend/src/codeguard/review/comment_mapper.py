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
    comments: list[InlineReviewComment] = []

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

        comments.append(
            InlineReviewComment(
                path=finding.file_path,
                line=finding.line_number,
                body=_build_comment_body(finding),
            )
        )

    return tuple(comments)
