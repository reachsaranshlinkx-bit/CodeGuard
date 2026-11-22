import pytest

from codeguard.context.file_discovery import (
    DiscoveredPythonFile,
    RepositoryFileDiscoveryResult,
)
from codeguard.context.file_indexer import FileSymbolIndexer
from codeguard.context.repository_indexer import RepositorySymbolIndexer
from codeguard.context.symbols import FileSymbolIndex


def make_discovered_file(
    path: str,
    sha_marker: str = "a",
    size_bytes: int = 128,
) -> DiscoveredPythonFile:
    return DiscoveredPythonFile(
        path=path,
        sha=sha_marker * 40,
        size_bytes=size_bytes,
    )


def make_file_index(
    discovered_file: DiscoveredPythonFile,
) -> FileSymbolIndex:
    source = "def process():\n    validate()\n"

    return FileSymbolIndexer().index(
        discovered_file,
        source,
    )


def make_discovery_result(
    files: tuple[DiscoveredPythonFile, ...],
    *,
    tree_truncated: bool = False,
    file_limit_reached: bool = False,
) -> RepositoryFileDiscoveryResult:
    return RepositoryFileDiscoveryResult(
        tree_sha="b" * 40,
        files=files,
        tree_truncated=tree_truncated,
        file_limit_reached=file_limit_reached,
        skipped_invalid_path_count=0,
        skipped_oversized_file_count=0,
        skipped_unknown_size_count=0,
    )


def test_repository_indexer_combines_file_indexes() -> None:
    payment_file = make_discovered_file("src/payments.py")
    order_file = make_discovered_file("src/orders.py", "c")

    discovery = make_discovery_result((payment_file, order_file))

    # Deliberately reverse the input order.
    file_indexes = (
        make_file_index(order_file),
        make_file_index(payment_file),
    )

    result = RepositorySymbolIndexer().index(
        commit_sha="d" * 40,
        discovery=discovery,
        file_indexes=file_indexes,
    )

    assert result.commit_sha == "d" * 40
    assert result.commit_sha != discovery.tree_sha

    assert len(result.files) == 2
    assert result.files[0].discovered_file.path == "src/payments.py"
    assert result.files[1].discovered_file.path == "src/orders.py"
    assert result.is_complete is True


def test_repository_indexer_preserves_discovery_metadata() -> None:
    discovered_file = make_discovered_file("src/payments.py")

    discovery = make_discovery_result(
        (discovered_file,),
        tree_truncated=True,
        file_limit_reached=True,
    )

    result = RepositorySymbolIndexer().index(
        commit_sha="d" * 40,
        discovery=discovery,
        file_indexes=(make_file_index(discovered_file),),
    )

    assert result.tree_truncated is True
    assert result.file_limit_reached is True
    assert result.is_complete is False


def test_repository_indexer_accepts_empty_repository() -> None:
    discovery = make_discovery_result(())

    result = RepositorySymbolIndexer().index(
        commit_sha="d" * 40,
        discovery=discovery,
        file_indexes=(),
    )

    assert result.commit_sha == "d" * 40
    assert result.files == ()
    assert result.is_complete is True


def test_repository_indexer_rejects_missing_file_indexes() -> None:
    payment_file = make_discovered_file("src/payments.py")
    order_file = make_discovered_file("src/orders.py", "c")

    discovery = make_discovery_result((payment_file, order_file))

    with pytest.raises(ValueError, match="Missing"):
        RepositorySymbolIndexer().index(
            commit_sha="d" * 40,
            discovery=discovery,
            file_indexes=(make_file_index(payment_file),),
        )


def test_repository_indexer_rejects_unexpected_file_indexes() -> None:
    discovery_file = make_discovered_file("src/payments.py")
    unexpected_file = make_discovered_file("src/orders.py", "c")

    discovery = make_discovery_result((discovery_file,))

    with pytest.raises(ValueError, match="not present"):
        RepositorySymbolIndexer().index(
            commit_sha="d" * 40,
            discovery=discovery,
            file_indexes=(make_file_index(unexpected_file),),
        )


def test_repository_indexer_rejects_mismatched_file_metadata() -> None:
    discovered_file = make_discovered_file("src/payments.py")

    discovery = make_discovery_result((discovered_file,))

    mismatched_file = make_discovered_file(
        "src/payments.py",
        sha_marker="c",
    )

    with pytest.raises(ValueError, match="metadata"):
        RepositorySymbolIndexer().index(
            commit_sha="d" * 40,
            discovery=discovery,
            file_indexes=(make_file_index(mismatched_file),),
        )


def test_repository_indexer_rejects_duplicate_file_indexes() -> None:
    discovered_file = make_discovered_file("src/payments.py")
    file_index = make_file_index(discovered_file)

    discovery = make_discovery_result((discovered_file,))

    with pytest.raises(ValueError, match="duplicate"):
        RepositorySymbolIndexer().index(
            commit_sha="d" * 40,
            discovery=discovery,
            file_indexes=(file_index, file_index),
        )


def test_repository_indexer_rejects_duplicate_discovery_paths() -> None:
    discovered_file = make_discovered_file("src/payments.py")

    discovery = make_discovery_result((discovered_file, discovered_file))

    with pytest.raises(ValueError, match="Discovery contains duplicate"):
        RepositorySymbolIndexer().index(
            commit_sha="d" * 40,
            discovery=discovery,
            file_indexes=(make_file_index(discovered_file),),
        )
