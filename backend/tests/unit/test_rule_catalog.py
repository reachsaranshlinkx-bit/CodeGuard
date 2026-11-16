import pytest

from codeguard.review.finding_models import RuleDefinition
from codeguard.review.rule_catalog import get_rule, list_rules

EXPECTED_RULES = (
    ("PY001", "CWE-95"),
    ("PY002", "CWE-78"),
    ("PY003", "CWE-502"),
    ("PY004", "CWE-502"),
)


@pytest.mark.parametrize(
    ("rule_id", "expected_cwe"),
    EXPECTED_RULES,
)
def test_get_rule_returns_expected_definition(
    rule_id: str,
    expected_cwe: str,
) -> None:
    rule = get_rule(rule_id)

    assert rule is not None
    assert isinstance(rule, RuleDefinition)

    assert rule.rule_id == rule_id
    assert rule.cwe == expected_cwe


def test_rule_catalog_contains_all_existing_analyzer_rules() -> None:
    rules = list_rules()

    assert len(rules) == 4
    assert tuple(rule.rule_id for rule in rules) == (
        "PY001",
        "PY002",
        "PY003",
        "PY004",
    )


def test_rule_catalog_contains_unique_rule_ids() -> None:
    rules = list_rules()

    rule_ids = [rule.rule_id for rule in rules]

    assert len(rule_ids) == len(set(rule_ids))


def test_unknown_rule_returns_none() -> None:
    assert get_rule("PY999") is None


@pytest.mark.parametrize(
    ("rule_id", "expected_severity"),
    [
        ("PY001", "high"),
        ("PY002", "high"),
        ("PY003", "high"),
        ("PY004", "high"),
    ],
)
def test_rule_has_valid_severity_and_confidence(
    rule_id: str,
    expected_severity: str,
) -> None:
    rule = get_rule(rule_id)

    assert rule is not None
    assert rule.severity == expected_severity
    assert rule.confidence in {"low", "medium", "high"}


@pytest.mark.parametrize(
    "rule_id",
    ["PY001", "PY002", "PY003", "PY004"],
)
def test_rule_contains_remediation_and_references(
    rule_id: str,
) -> None:
    rule = get_rule(rule_id)

    assert rule is not None
    assert rule.title.strip()
    assert rule.description.strip()
    assert rule.remediation.strip()
    assert len(rule.references) >= 1
