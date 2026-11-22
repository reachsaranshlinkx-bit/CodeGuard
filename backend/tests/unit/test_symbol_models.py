import pytest

from codeguard.analyzers.python_ast import PythonASTAnalyzer
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.symbols import (
    FileSymbolIndex,
    RepositorySymbolIndex,
)


def make_file_symbol_index(
    path: str = "src/payments.py",
) -> FileSymbolIndex:
    discovered_file = DiscoveredPythonFile(
        path=path,
        sha="b" * 40,
        size_bytes=128,
    )

    analysis = PythonASTAnalyzer().analyze(
        "def process_payment():\n    validate_payment()\n",
        file_path=path,
    )

    return FileSymbolIndex(
        discovered_file=discovered_file,
        analysis=analysis,
    )


def test_file_symbol_index_preserves_file_and_analysis_metadata() -> None:
    result = make_file_symbol_index()

    assert result.discovered_file.path == "src/payments.py"
    assert result.discovered_file.sha == "b" * 40
    assert result.discovered_file.size_bytes == 128

    assert result.analysis.file_path == "src/payments.py"
    assert result.analysis.functions[0].qualified_name == "process_payment"
    assert result.analysis.calls[0].callee == "validate_payment"


def test_file_symbol_index_rejects_mismatched_file_path() -> None:
    file_index = make_file_symbol_index()

    mismatched_analysis = PythonASTAnalyzer().analyze(
        "def process_payment():\n    pass\n",
        file_path="src/orders.py",
    )

    with pytest.raises(ValueError, match="must match"):
        FileSymbolIndex(
            discovered_file=file_index.discovered_file,
            analysis=mismatched_analysis,
        )


def test_repository_symbol_index_preserves_file_indexes() -> None:
    payment_index = make_file_symbol_index("src/payments.py")
    order_index = make_file_symbol_index("src/orders.py")

    result = RepositorySymbolIndex(
        commit_sha="a" * 40,
        files=(payment_index, order_index),
    )

    assert result.commit_sha == "a" * 40
    assert len(result.files) == 2
    assert result.files[0].discovered_file.path == "src/payments.py"
    assert result.files[1].discovered_file.path == "src/orders.py"
    assert result.is_complete is True


def test_repository_symbol_index_detects_truncated_tree() -> None:
    result = RepositorySymbolIndex(
        commit_sha="a" * 40,
        files=(make_file_symbol_index(),),
        tree_truncated=True,
    )

    assert result.tree_truncated is True
    assert result.is_complete is False


def test_repository_symbol_index_detects_file_limit() -> None:
    result = RepositorySymbolIndex(
        commit_sha="a" * 40,
        files=(make_file_symbol_index(),),
        file_limit_reached=True,
    )

    assert result.file_limit_reached is True
    assert result.is_complete is False


def test_repository_symbol_index_rejects_empty_commit_sha() -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        RepositorySymbolIndex(
            commit_sha=" ",
            files=(),
        )


def test_repository_symbol_index_rejects_duplicate_file_paths() -> None:
    file_index = make_file_symbol_index()

    with pytest.raises(ValueError, match="duplicate"):
        RepositorySymbolIndex(
            commit_sha="a" * 40,
            files=(file_index, file_index),
        )
