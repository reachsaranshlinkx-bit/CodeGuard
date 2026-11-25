from collections.abc import Callable

import pytest
from codeguard.ai.context_builder import AIReviewContextBuilder

from codeguard.ai.contracts import AIReviewInput
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import RepositoryFileFailure


def make_source_file(path: str, source: str) -> RepositorySourceFile:
    return RepositorySourceFile(
        discovered_file=DiscoveredPythonFile(
            path=path,
            sha="a" * 40,
            size_bytes=len(source.encode("utf-8")),
        ),
        source=source,
    )


def make_snapshot(
    files: tuple[RepositorySourceFile, ...],
    **metadata: object,
) -> RepositorySourceSnapshot:
    return RepositorySourceSnapshot(
        commit_sha="b" * 40,
        files=files,
        **metadata,  # type: ignore[arg-type]
    )


@pytest.fixture
def build_context() -> Callable[
    [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
    str | None,
]:
    builder = AIReviewContextBuilder()

    def build(
        changed_file_paths: tuple[str, ...],
        related_file_paths: tuple[str, ...],
        source_snapshot: RepositorySourceSnapshot,
    ) -> str | None:
        return builder.build(
            changed_file_paths=changed_file_paths,
            related_file_paths=related_file_paths,
            source_snapshot=source_snapshot,
        )

    return build


def test_build_includes_changed_and_related_source_but_not_unrelated_files(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    changed_source = "def changed():\n    return 'changed'\n"
    caller_source = "def caller():\n    return changed()\n"
    unrelated_source = "def unrelated():\n    return 'unrelated'\n"
    snapshot = make_snapshot(
        (
            make_source_file("src/changed.py", changed_source),
            make_source_file("src/caller.py", caller_source),
            make_source_file("src/unrelated.py", unrelated_source),
        )
    )

    context = build_context(
        ("src/changed.py",),
        ("src/caller.py",),
        snapshot,
    )

    assert context is not None
    assert changed_source in context
    assert caller_source in context
    assert unrelated_source not in context
    assert "src/unrelated.py" not in context


def test_build_prioritizes_changed_files_and_orders_paths_deterministically(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    sources = {
        "src/z_changed.py": "Z_CHANGED_SOURCE\n",
        "src/a_changed.py": "A_CHANGED_SOURCE\n",
        "src/z_caller.py": "Z_CALLER_SOURCE\n",
        "src/a_caller.py": "A_CALLER_SOURCE\n",
    }
    snapshot = make_snapshot(
        tuple(make_source_file(path, source) for path, source in sources.items())
    )

    context = build_context(
        ("src/z_changed.py", "src/a_changed.py"),
        ("src/z_caller.py", "src/a_caller.py"),
        snapshot,
    )
    reordered_context = build_context(
        ("src/a_changed.py", "src/z_changed.py"),
        ("src/a_caller.py", "src/z_caller.py"),
        make_snapshot(tuple(reversed(snapshot.files))),
    )

    assert context is not None
    assert reordered_context == context
    assert context.index("src/a_changed.py") < context.index("src/z_changed.py")
    assert context.index("src/z_changed.py") < context.index("src/a_caller.py")
    assert context.index("src/a_caller.py") < context.index("src/z_caller.py")


def test_build_deduplicates_paths(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    source = "def shared():\n    return True\n"
    snapshot = make_snapshot((make_source_file("src/shared.py", source),))

    context = build_context(
        ("src/shared.py", "src/shared.py"),
        ("src/shared.py",),
        snapshot,
    )

    assert context is not None
    assert context.count("src/shared.py") == 1
    assert context.count(source) == 1


def test_build_reports_unavailable_requested_files_without_losing_available_source(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    available_source = "def available():\n    return True\n"
    failure = RepositoryFileFailure(
        file_path="src/missing.py",
        stage="retrieval",
        error_type="FileNotFoundError",
        message="The source file could not be retrieved.",
    )
    snapshot = make_snapshot(
        (make_source_file("src/available.py", available_source),),
        failures=(failure,),
    )

    context = build_context(
        ("src/missing.py",),
        ("src/available.py",),
        snapshot,
    )

    assert context is not None
    assert available_source in context
    assert "src/missing.py" in context


def test_build_reports_files_omitted_to_fit_the_context_limit(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    changed_source = "CHANGED_START\n" + ("x" * 39_000) + "\nCHANGED_END\n"
    caller_source = "CALLER_START\n" + ("y" * 15_000) + "\nCALLER_END\n"
    snapshot = make_snapshot(
        (
            make_source_file("src/changed.py", changed_source),
            make_source_file("src/caller.py", caller_source),
        )
    )

    context = build_context(
        ("src/changed.py",),
        ("src/caller.py",),
        snapshot,
    )

    assert context is not None
    assert changed_source in context
    assert "src/changed.py" in context
    assert "src/caller.py" in context
    assert "omitted" in context.lower()
    assert "CALLER_START" not in context
    assert "CALLER_END" not in context
    AIReviewInput(
        owner="octocat",
        repo="CodeGuard",
        pull_number=1,
        head_sha="c" * 40,
        context=context,
    )


def test_build_does_not_truncate_a_file_that_does_not_fit(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    source = "TOO_LARGE_START\n" + ("z" * 50_000) + "\nTOO_LARGE_END\n"
    fallback_source = "def fallback():\n    return 'kept'\n"
    snapshot = make_snapshot(
        (
            make_source_file("src/too_large.py", source),
            make_source_file("src/fallback.py", fallback_source),
        )
    )

    context = build_context(
        ("src/too_large.py",),
        ("src/fallback.py",),
        snapshot,
    )

    assert context is not None
    assert fallback_source in context
    assert "src/too_large.py" in context
    assert "omitted" in context.lower()
    assert "TOO_LARGE_START" not in context
    assert "TOO_LARGE_END" not in context


def test_build_preserves_source_whitespace_and_indentation(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    source = "def example():\r\n\tvalue = '  spaced  '\r\n\treturn value  \r\n"
    snapshot = make_snapshot((make_source_file("src/whitespace.py", source),))

    context = build_context(("src/whitespace.py",), (), snapshot)

    assert context is not None
    assert source in context


def test_build_marks_incomplete_snapshot_coverage_and_preserves_its_reason(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    source = "def retained():\n    return True\n"
    snapshot = make_snapshot(
        (make_source_file("src/retained.py", source),),
        tree_truncated=True,
    )

    context = build_context(("src/retained.py",), (), snapshot)

    assert context is not None
    assert source in context
    assert "incomplete" in context.lower()
    assert "truncated" in context.lower()


@pytest.mark.parametrize(
    "coverage_metadata",
    [
        {"file_limit_reached": True},
        {
            "failures": (
                RepositoryFileFailure(
                    file_path="src/unavailable.py",
                    stage="retrieval",
                    error_type="TimeoutError",
                    message="Retrieval timed out.",
                ),
            )
        },
        {"omitted_source_paths": ("src/omitted.py",)},
        {"skipped_invalid_path_count": 1},
        {"skipped_oversized_file_count": 1},
        {"skipped_unknown_size_count": 1},
    ],
)
def test_build_marks_other_incomplete_snapshot_coverage(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
    coverage_metadata: dict[str, object],
) -> None:
    source = "def retained():\n    return True\n"
    snapshot = make_snapshot(
        (make_source_file("src/retained.py", source),),
        **coverage_metadata,
    )

    context = build_context(("src/retained.py",), (), snapshot)

    assert context is not None
    assert source in context
    assert "incomplete" in context.lower()


def test_build_returns_none_when_no_requested_source_is_available(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    snapshot = make_snapshot(
        (make_source_file("src/unrelated.py", "UNRELATED_SOURCE\n"),)
    )

    context = build_context(
        ("src/missing.py",),
        ("src/also_missing.py",),
        snapshot,
    )

    assert context is None


def test_build_returns_none_when_no_relevant_paths_are_requested(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    snapshot = make_snapshot(
        (make_source_file("src/available.py", "AVAILABLE_SOURCE\n"),)
    )

    assert build_context((), (), snapshot) is None


def test_build_does_not_mutate_source_snapshot(
    build_context: Callable[
        [tuple[str, ...], tuple[str, ...], RepositorySourceSnapshot],
        str | None,
    ],
) -> None:
    snapshot = make_snapshot(
        (
            make_source_file("src/changed.py", "CHANGED_SOURCE\n"),
            make_source_file("src/caller.py", "CALLER_SOURCE\n"),
        ),
        tree_truncated=True,
        omitted_source_paths=("src/other.py",),
    )
    original_snapshot = snapshot
    original_files = snapshot.files
    original_omitted_paths = snapshot.omitted_source_paths

    context = build_context(("src/changed.py",), ("src/caller.py",), snapshot)

    assert context is not None
    assert snapshot is original_snapshot
    assert snapshot.files is original_files
    assert snapshot.omitted_source_paths is original_omitted_paths
    assert snapshot.tree_truncated is True
