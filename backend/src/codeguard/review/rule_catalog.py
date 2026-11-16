from types import MappingProxyType

from codeguard.review.finding_models import RuleDefinition

_RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        rule_id="PY001",
        title="Dynamic Code Execution",
        description=(
            "eval() and exec() execute dynamically supplied Python code. "
            "Untrusted input can lead to arbitrary code execution."
        ),
        severity="high",
        confidence="medium",
        cwe="CWE-95",
        remediation=(
            "Avoid eval() and exec() with untrusted input. "
            "Use ast.literal_eval() for supported Python literal data "
            "or implement a restricted parser."
        ),
        references=(
            "https://cwe.mitre.org/data/definitions/95.html",
            "https://docs.python.org/3/library/functions.html#eval",
        ),
    ),
    RuleDefinition(
        rule_id="PY002",
        title="Unsafe Command Execution",
        description=(
            "Shell command execution can introduce command injection "
            "when untrusted input influences the executed command."
        ),
        severity="high",
        confidence="medium",
        cwe="CWE-78",
        remediation=(
            "Prefer subprocess APIs with argument lists and shell=False. "
            "Validate untrusted arguments before execution."
        ),
        references=(
            "https://cwe.mitre.org/data/definitions/78.html",
            "https://docs.python.org/3/library/subprocess.html#security-considerations",
        ),
    ),
    RuleDefinition(
        rule_id="PY003",
        title="Unsafe Pickle Deserialization",
        description=(
            "pickle.load() and pickle.loads() can execute malicious "
            "code when processing untrusted serialized data."
        ),
        severity="high",
        confidence="medium",
        cwe="CWE-502",
        remediation=(
            "Never deserialize untrusted data using pickle. "
            "Prefer safer data formats such as JSON when appropriate."
        ),
        references=(
            "https://cwe.mitre.org/data/definitions/502.html",
            "https://docs.python.org/3/library/pickle.html",
        ),
    ),
    RuleDefinition(
        rule_id="PY004",
        title="Unsafe YAML Deserialization",
        description=(
            "yaml.load() without a recognized safe loader can "
            "deserialize unsafe Python objects."
        ),
        severity="high",
        confidence="medium",
        cwe="CWE-502",
        remediation=(
            "Use yaml.safe_load() or an explicitly configured safe loader "
            "when processing untrusted YAML."
        ),
        references=(
            "https://cwe.mitre.org/data/definitions/502.html",
            "https://pyyaml.org/wiki/PyYAMLDocumentation",
        ),
    ),
)


_RULES_BY_ID = MappingProxyType({rule.rule_id: rule for rule in _RULES})


def get_rule(rule_id: str) -> RuleDefinition | None:
    """Return a rule definition by its identifier."""

    return _RULES_BY_ID.get(rule_id)


def list_rules() -> tuple[RuleDefinition, ...]:
    """Return all registered security rules in stable order."""

    return _RULES
