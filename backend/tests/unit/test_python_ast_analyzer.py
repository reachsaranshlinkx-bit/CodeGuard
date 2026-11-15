import pytest

from codeguard.analyzers.python_ast import PythonASTAnalyzer


def test_analyzer_handles_empty_source() -> None:
    result = PythonASTAnalyzer().analyze(
        "",
        file_path="src/empty.py",
    )

    assert result.file_path == "src/empty.py"
    assert result.functions == ()
    assert result.classes == ()
    assert result.imports == ()
    assert result.calls == ()


def test_analyzer_extracts_top_level_functions() -> None:
    source = "def add(a, b):\n    return a + b\n\nasync def fetch():\n    return None\n"

    result = PythonASTAnalyzer().analyze(
        source,
        file_path="src/calculator.py",
    )

    assert len(result.functions) == 2

    add, fetch = result.functions

    assert add.name == "add"
    assert add.qualified_name == "add"
    assert add.lineno == 1
    assert add.end_lineno == 2
    assert add.is_async is False
    assert add.is_method is False

    assert fetch.name == "fetch"
    assert fetch.qualified_name == "fetch"
    assert fetch.lineno == 4
    assert fetch.end_lineno == 5
    assert fetch.is_async is True
    assert fetch.is_method is False


def test_analyzer_extracts_classes_and_methods() -> None:
    source = (
        "class Order:\n"
        "    def process(self):\n"
        "        self.save()\n"
        "\n"
        "    def save(self):\n"
        "        return True\n"
    )

    result = PythonASTAnalyzer().analyze(
        source,
        file_path="src/orders.py",
    )

    assert len(result.classes) == 1

    class_info = result.classes[0]

    assert class_info.name == "Order"
    assert class_info.qualified_name == "Order"
    assert class_info.lineno == 1
    assert class_info.end_lineno == 6
    assert class_info.methods == ("process", "save")

    assert len(result.functions) == 2

    process, save = result.functions

    assert process.qualified_name == "Order.process"
    assert process.is_method is True

    assert save.qualified_name == "Order.save"
    assert save.is_method is True


def test_analyzer_extracts_imports_and_aliases() -> None:
    source = (
        "import os\n"
        "import json as json_lib\n"
        "from pathlib import Path\n"
        "from .helpers import normalize as clean\n"
    )

    result = PythonASTAnalyzer().analyze(
        source,
        file_path="src/example.py",
    )

    assert len(result.imports) == 4

    assert result.imports[0].module is None
    assert result.imports[0].name == "os"
    assert result.imports[0].alias is None

    assert result.imports[1].module is None
    assert result.imports[1].name == "json"
    assert result.imports[1].alias == "json_lib"

    assert result.imports[2].module == "pathlib"
    assert result.imports[2].name == "Path"

    assert result.imports[3].module == "helpers"
    assert result.imports[3].name == "normalize"
    assert result.imports[3].alias == "clean"
    assert result.imports[3].level == 1


def test_analyzer_extracts_function_and_method_calls() -> None:
    source = 'print("hello")\nclient.get("/items")\nself.save()\n'

    result = PythonASTAnalyzer().analyze(
        source,
        file_path="src/example.py",
    )

    assert len(result.calls) == 3

    assert [call.callee for call in result.calls] == [
        "print",
        "client.get",
        "self.save",
    ]

    assert [call.lineno for call in result.calls] == [
        1,
        2,
        3,
    ]


def test_analyzer_preserves_nested_function_names() -> None:
    source = "def outer():\n    def inner():\n        return 1\n\n    return inner()\n"

    result = PythonASTAnalyzer().analyze(
        source,
        file_path="src/nested.py",
    )

    assert len(result.functions) == 2

    assert result.functions[0].qualified_name == "outer"
    assert result.functions[1].qualified_name == "outer.inner"

    assert result.functions[1].lineno == 2
    assert result.functions[1].end_lineno == 3

    assert result.calls[0].callee == "inner"


def test_analyzer_rejects_invalid_python_syntax() -> None:
    source = "def broken(:\n    return True\n"

    with pytest.raises(SyntaxError):
        PythonASTAnalyzer().analyze(
            source,
            file_path="src/broken.py",
        )


def test_analyzer_does_not_execute_repository_code() -> None:
    source = "raise RuntimeError('This must not execute')\n"

    result = PythonASTAnalyzer().analyze(
        source,
        file_path="src/untrusted.py",
    )

    assert result.file_path == "src/untrusted.py"
    assert result.calls[0].callee == "RuntimeError"
