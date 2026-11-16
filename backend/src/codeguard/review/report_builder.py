from __future__ import annotations

import hashlib

from codeguard.analyzers.python_risk import RiskFinding
from codeguard.review.finding_models import (
    ReviewFinding,
    ReviewReport,
)
from codeguard.review.rule_catalog import get_rule
from codeguard.services.pull_request_analyzer import (
    PullRequestAnalysisResult,
)


def _create_fingerprint(finding: RiskFinding) -> str:
    """Create a stable identifier for a security finding."""

    identity = "\0".join(
        (
            finding.rule_id,
            finding.file_path,
            str(finding.lineno),
        )
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _enrich_finding(finding: RiskFinding) -> ReviewFinding:
    """Enrich a raw security finding using the rule catalog."""

    rule = get_rule(finding.rule_id)

    if rule is None:
        raise ValueError(f"Unknown rule: {finding.rule_id}")

    return ReviewFinding(
        fingerprint=_create_fingerprint(finding),
        rule_id=rule.rule_id,
        title=rule.title,
        severity=rule.severity,
        confidence=rule.confidence,
        message=finding.message,
        description=rule.description,
        file_path=finding.file_path,
        line_number=finding.lineno,
        cwe=rule.cwe,
        evidence=finding.message,
        remediation=rule.remediation,
        references=rule.references,
    )


def build_review_report(
    analysis: PullRequestAnalysisResult,
) -> ReviewReport:
    """Build a structured security review report from PR analysis."""

    review_findings: list[ReviewFinding] = []
    unavailable_mapping: set[str] = set()
    files_analyzed = 0

    for file_result in analysis.files:
        # Only count files whose Python AST analysis succeeded.
        if file_result.ast_result is None:
            continue

        files_analyzed += 1

        # A missing mapping must not silently discard
        # complete-file security findings.
        if file_result.changed_line_findings is None:
            selected_findings = file_result.risk_findings
            unavailable_mapping.add(file_result.file_path)
        else:
            selected_findings = file_result.changed_line_findings

        for finding in selected_findings:
            review_findings.append(_enrich_finding(finding))

    # Make the output deterministic across repeated executions.
    review_findings.sort(
        key=lambda finding: (
            finding.file_path,
            finding.line_number,
            finding.rule_id,
            finding.fingerprint,
        )
    )

    return ReviewReport(
        owner=analysis.owner,
        repo=analysis.repo,
        pull_number=analysis.pull_number,
        title=analysis.title,
        html_url=analysis.html_url,
        head_sha=analysis.head_sha,
        findings=tuple(review_findings),
        files_analyzed=files_analyzed,
        files_with_unavailable_mapping=tuple(sorted(unavailable_mapping)),
    )
