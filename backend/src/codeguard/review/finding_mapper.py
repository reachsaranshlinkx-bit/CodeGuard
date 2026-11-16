from codeguard.analyzers.python_risk import RiskFinding


def filter_findings_to_added_lines(
    findings: tuple[RiskFinding, ...],
    added_line_numbers: frozenset[int],
) -> tuple[RiskFinding, ...]:
    """Keep only security findings located on newly added lines."""

    return tuple(
        finding for finding in findings if finding.lineno in added_line_numbers
    )
