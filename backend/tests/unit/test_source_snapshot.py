import pytest

from codeguard.analyzers.python_ast import PythonASTAnalyzer
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositoryContextBuildResult,
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import (
    FileSymbolIndex,
    RepositoryFileFailure,
    RepositorySymbolIndex,
)


def make_source_file(
    path: str = "src/auth.py",
    source: str = "def authenticate():\n    return True\n",
) -> RepositorySourceFile:
    discovered_file = DiscoveredPythonFile(
        path=path,
        sha="a" * 40,
        size_bytes=len(source.encode("utf-8")),
    )

    return RepositorySourceFile(
        discovered_file=discovered_file,
        source=source,
    )


def make_snapshot(
    files: tuple[RepositorySourceFile, ...] | None = None,
    **kwargs: object,
) -> RepositorySourceSnapshot:
    return RepositorySourceSnapshot(
        commit_sha="b" * 40,
        files=files if files is not None else (make_source_file(),),
        **kwargs,  # type: ignore[arg-type]
    )


def test_source_file_preserves_original_source() -> None:
    source = "def authenticate():\n    return True\n"
    source_file = make_source_file(source=source)

    assert source_file.source == source
    assert source_file.path == "src/auth.py"
    assert source_file.blob_sha == "a" * 40
    assert source_file.source_size_bytes == len(source.encode("utf-8"))


def test_source_file_accepts_empty_source() -> None:
    source_file = make_source_file(source="")

    assert source_file.source == ""
    assert source_file.source_size_bytes == 0


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../secrets.py",
        "src/../secrets.py",
        r"src\auth.py",
        "C:/secrets.py",
        "src//auth.py",
        "src/auth.py\x00",
    ],
)
def test_source_file_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        make_source_file(path=path)


def test_snapshot_is_complete_without_coverage_gaps() -> None:
    snapshot = make_snapshot()

    assert snapshot.is_complete is True
    assert snapshot.total_source_bytes > 0


def test_snapshot_tracks_truncated_tree() -> None:
    snapshot = make_snapshot(tree_truncated=True)

    assert snapshot.is_complete is False


def test_snapshot_tracks_file_limit() -> None:
    snapshot = make_snapshot(file_limit_reached=True)

    assert snapshot.is_complete is False


def test_snapshot_tracks_retrieval_failures() -> None:
    failure = RepositoryFileFailure(
        file_path="src/missing.py",
        stage="retrieval",
        error_type="GitHubNotFoundError",
        message="File could not be retrieved.",
    )

    snapshot = make_snapshot(files=(), failures=(failure,))

    assert snapshot.is_complete is False
    assert snapshot.failures == (failure,)


def test_snapshot_tracks_omitted_source_files() -> None:
    snapshot = make_snapshot(omitted_source_paths=("src/large.py",))

    assert snapshot.is_complete is False
    assert snapshot.omitted_source_paths == ("src/large.py",)


@pytest.mark.parametrize(
    "metadata",
    [
        {"skipped_invalid_path_count": 1},
        {"skipped_oversized_file_count": 1},
        {"skipped_unknown_size_count": 1},
    ],
)
def test_snapshot_is_incomplete_when_discovery_skips_files(
    metadata: dict[str, int],
) -> None:
    snapshot = make_snapshot(**metadata)

    assert snapshot.is_complete is False


def test_snapshot_rejects_duplicate_source_paths() -> None:
    source_file = make_source_file()

    with pytest.raises(ValueError, match="duplicate file paths"):
        make_snapshot(files=(source_file, source_file))


def test_snapshot_rejects_source_and_failure_for_same_path() -> None:
    source_file = make_source_file()

    failure = RepositoryFileFailure(
        file_path="src/auth.py",
        stage="retrieval",
        error_type="GitHubError",
        message="Retrieval failure.",
    )

    with pytest.raises(ValueError, match="retrieval failure"):
        make_snapshot(
            files=(source_file,),
            failures=(failure,),
        )


def test_snapshot_rejects_duplicate_omitted_paths() -> None:
    with pytest.raises(ValueError, match="duplicate omitted paths"):
        make_snapshot(
            omitted_source_paths=("src/large.py", "src/large.py"),
        )


def test_snapshot_rejects_negative_skipped_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_snapshot(skipped_oversized_file_count=-1)


def test_snapshot_rejects_analysis_failure() -> None:
    failure = RepositoryFileFailure(
        file_path="src/auth.py",
        stage="analysis",
        error_type="SyntaxError",
        message="Invalid Python syntax.",
    )

    with pytest.raises(ValueError, match="retrieval failures"):
        make_snapshot(files=(), failures=(failure,))


def test_context_build_result_requires_matching_commit_sha() -> None:
    symbol_index = RepositorySymbolIndex(
        commit_sha="a" * 40,
        files=(),
    )

    snapshot = make_snapshot(files=())

    with pytest.raises(ValueError, match="same commit SHA"):
        RepositoryContextBuildResult(
            symbol_index=symbol_index,
            source_snapshot=snapshot,
        )


def test_context_build_result_accepts_matching_commit_sha() -> None:
    commit_sha = "b" * 40

    result = RepositoryContextBuildResult(
        symbol_index=RepositorySymbolIndex(
            commit_sha=commit_sha,
            files=(),
        ),
        source_snapshot=RepositorySourceSnapshot(
            commit_sha=commit_sha,
            files=(),
        ),
    )

    assert result.symbol_index.commit_sha == result.source_snapshot.commit_sha


def test_context_build_result_rejects_missing_source_coverage() -> None:
    commit_sha = "b" * 40
    source = "def authenticate():\n    return True\n"

    discovered_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="a" * 40,
        size_bytes=len(source.encode("utf-8")),
    )

    analysis = PythonASTAnalyzer().analyze(
        source,
        file_path="src/auth.py",
    )

    symbol_index = RepositorySymbolIndex(
        commit_sha=commit_sha,
        files=(
            FileSymbolIndex(
                discovered_file=discovered_file,
                analysis=analysis,
            ),
        ),
    )

    snapshot = RepositorySourceSnapshot(
        commit_sha=commit_sha,
        files=(),
    )

    with pytest.raises(ValueError, match="same discovered files"):
        RepositoryContextBuildResult(
            symbol_index=symbol_index,
            source_snapshot=snapshot,
        )


def test_context_build_result_rejects_mismatched_retrieval_failures() -> None:
    commit_sha = "b" * 40

    symbol_failure = RepositoryFileFailure(
        file_path="src/auth.py",
        stage="retrieval",
        error_type="RuntimeError",
        message="GitHub request failed.",
    )

    snapshot_failure = RepositoryFileFailure(
        file_path="src/auth.py",
        stage="retrieval",
        error_type="TimeoutError",
        message="Request timed out.",
    )

    symbol_index = RepositorySymbolIndex(
        commit_sha=commit_sha,
        files=(),
        failures=(symbol_failure,),
    )

    snapshot = RepositorySourceSnapshot(
        commit_sha=commit_sha,
        files=(),
        failures=(snapshot_failure,),
    )

    with pytest.raises(ValueError, match="retrieval failure records"):
        RepositoryContextBuildResult(
            symbol_index=symbol_index,
            source_snapshot=snapshot,
        )
