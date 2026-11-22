import pytest

from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.file_indexer import FileSymbolIndexer


def make_discovered_file(
    path: str = "src/payments.py",
) -> DiscoveredPythonFile:
    return DiscoveredPythonFile(
        path=path,
        sha="a" * 40,
        size_bytes=128,
    )


def test_file_indexer_builds_symbol_index() -> None:
    discovered_file = make_discovered_file()

    source = (
        "import os\n"
        "\n"
        "class PaymentService:\n"
        "    def process(self):\n"
        "        validate_payment()\n"
    )

    result = FileSymbolIndexer().index(
        discovered_file,
        source,
    )

    assert result.discovered_file == discovered_file

    assert result.analysis.file_path == "src/payments.py"

    assert result.analysis.classes[0].name == "PaymentService"

    assert result.analysis.functions[0].qualified_name == ("PaymentService.process")

    assert result.analysis.imports[0].name == "os"

    assert result.analysis.calls[0].callee == "validate_payment"

    assert result.analysis.calls[0].caller_qualified_name == ("PaymentService.process")


def test_file_indexer_preserves_discovered_file_metadata() -> None:
    discovered_file = make_discovered_file(
        "src/orders.py",
    )

    result = FileSymbolIndexer().index(
        discovered_file,
        "def create_order():\n    pass\n",
    )

    assert result.discovered_file.path == "src/orders.py"
    assert result.discovered_file.sha == "a" * 40
    assert result.discovered_file.size_bytes == 128


def test_file_indexer_handles_empty_source() -> None:
    discovered_file = make_discovered_file()

    result = FileSymbolIndexer().index(
        discovered_file,
        "",
    )

    assert result.analysis.functions == ()
    assert result.analysis.classes == ()
    assert result.analysis.imports == ()
    assert result.analysis.calls == ()


def test_file_indexer_rejects_invalid_python_syntax() -> None:
    discovered_file = make_discovered_file()

    with pytest.raises(SyntaxError):
        FileSymbolIndexer().index(
            discovered_file,
            "def broken(:\n    pass\n",
        )
