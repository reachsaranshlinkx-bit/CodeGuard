import pytest

from codeguard.analyzers.python_risk import PythonRiskAnalyzer


def analyze(source: str):
    return PythonRiskAnalyzer().analyze(
        source,
        file_path="src/example.py",
    )


def test_risk_analyzer_handles_empty_source() -> None:
    findings = analyze("")

    assert findings == ()


def test_risk_analyzer_detects_eval_and_exec() -> None:
    source = "result = eval(user_input)\nexec(user_code)\n"

    findings = analyze(source)

    assert len(findings) == 2
    assert [finding.rule_id for finding in findings] == [
        "PY001",
        "PY001",
    ]
    assert [finding.lineno for finding in findings] == [1, 2]
    assert all(finding.severity == "high" for finding in findings)


def test_risk_analyzer_detects_os_system() -> None:
    source = "import os\nos.system(user_command)\n"

    findings = analyze(source)

    assert len(findings) == 1
    assert findings[0].rule_id == "PY002"
    assert findings[0].lineno == 2
    assert findings[0].severity == "high"


def test_risk_analyzer_detects_subprocess_shell_true() -> None:
    source = "import subprocess\nsubprocess.run(command, shell=True)\n"

    findings = analyze(source)

    assert len(findings) == 1
    assert findings[0].rule_id == "PY002"
    assert findings[0].lineno == 2


def test_risk_analyzer_allows_subprocess_shell_false() -> None:
    source = "import subprocess\nsubprocess.run(command, shell=False)\n"

    findings = analyze(source)

    assert findings == ()


def test_risk_analyzer_detects_pickle_deserialization() -> None:
    source = (
        "import pickle\n"
        "data = pickle.loads(payload)\n"
        "result = pickle.load(file_object)\n"
    )

    findings = analyze(source)

    assert len(findings) == 2
    assert all(finding.rule_id == "PY003" for finding in findings)
    assert [finding.lineno for finding in findings] == [2, 3]
    assert all(finding.severity == "high" for finding in findings)


def test_risk_analyzer_detects_unsafe_yaml_load() -> None:
    source = "import yaml\nconfig = yaml.load(data)\n"

    findings = analyze(source)

    assert len(findings) == 1
    assert findings[0].rule_id == "PY004"
    assert findings[0].lineno == 2


def test_risk_analyzer_allows_safe_yaml_loading() -> None:
    source = (
        "import yaml\n"
        "config = yaml.safe_load(data)\n"
        "config2 = yaml.load(data, Loader=yaml.SafeLoader)\n"
    )

    findings = analyze(source)

    assert findings == ()


def test_risk_analyzer_ignores_comments_and_strings() -> None:
    source = (
        "# eval(user_input)\n"
        'message = "exec(user_code)"\n'
        'description = "os.system(command)"\n'
    )

    findings = analyze(source)

    assert findings == ()


def test_risk_analyzer_preserves_file_path_and_line_numbers() -> None:
    source = "import os\n\nos.system(command)\n"

    findings = analyze(source)

    assert len(findings) == 1

    finding = findings[0]

    assert finding.file_path == "src/example.py"
    assert finding.lineno == 3


def test_risk_analyzer_does_not_execute_source() -> None:
    source = "raise RuntimeError('This code must not execute')\n"

    findings = analyze(source)

    assert findings == ()


def test_risk_analyzer_rejects_invalid_python_syntax() -> None:
    source = "eval(\n"

    with pytest.raises(SyntaxError):
        analyze(source)


def test_risk_analyzer_detects_aliased_module_imports() -> None:
    source = (
        "import os as operating_system\n"
        "import subprocess as process_runner\n"
        "operating_system.system(command)\n"
        "process_runner.run(command, shell=True)\n"
    )

    findings = analyze(source)

    assert len(findings) == 2
    assert [finding.rule_id for finding in findings] == [
        "PY002",
        "PY002",
    ]
    assert [finding.lineno for finding in findings] == [3, 4]


def test_risk_analyzer_detects_aliased_function_imports() -> None:
    source = (
        "from os import system as run_command\n"
        "from subprocess import Popen as spawn\n"
        "run_command(command)\n"
        "spawn(command, shell=True)\n"
    )

    findings = analyze(source)

    assert len(findings) == 2
    assert [finding.rule_id for finding in findings] == [
        "PY002",
        "PY002",
    ]
    assert [finding.lineno for finding in findings] == [3, 4]


def test_risk_analyzer_detects_aliased_pickle_and_yaml() -> None:
    source = (
        "import pickle as serializer\n"
        "import yaml as configuration\n"
        "serializer.loads(payload)\n"
        "configuration.load(payload)\n"
    )

    findings = analyze(source)

    assert len(findings) == 2
    assert [finding.rule_id for finding in findings] == [
        "PY003",
        "PY004",
    ]
    assert [finding.lineno for finding in findings] == [3, 4]


def test_risk_analyzer_detects_aliased_deserialization_functions() -> None:
    source = (
        "from pickle import loads as deserialize\n"
        "from yaml import load as parse_yaml\n"
        "deserialize(payload)\n"
        "parse_yaml(payload)\n"
    )

    findings = analyze(source)

    assert len(findings) == 2
    assert [finding.rule_id for finding in findings] == [
        "PY003",
        "PY004",
    ]


def test_risk_analyzer_detects_aliased_builtin_execution() -> None:
    source = "from builtins import eval as dynamic_eval\ndynamic_eval(user_input)\n"

    findings = analyze(source)

    assert len(findings) == 1
    assert findings[0].rule_id == "PY001"
    assert findings[0].lineno == 2


@pytest.mark.parametrize(
    "pattern",
    [
        "case operating_system:",
        "case [*operating_system]:",
        'case {"key": _, **operating_system}:',
    ],
)
def test_match_pattern_capture_shadows_imported_module_name(
    pattern: str,
) -> None:
    source = (
        "import os as operating_system\n"
        "\n"
        "def execute(value, command):\n"
        "    match value:\n"
        f"        {pattern}\n"
        "            pass\n"
        "\n"
        "    operating_system.system(command)\n"
    )

    findings = analyze(source)

    assert findings == ()
