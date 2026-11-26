import re

import pytest

from codeguard.ai.contracts import AIReviewInput
from codeguard.ai.execution_fingerprint import (
    AI_EXECUTION_INPUT_FORMAT_VERSION,
    build_ai_execution_input_fingerprint,
)
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import RepositoryFileFailure


def make_review_input(**updates: object) -> AIReviewInput:
    values: dict[str, object] = {
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
        "head_sha": "b" * 40,
        "context": "AI review repository context\nChanged files:\nFile: src/auth.py",
    }
    values.update(updates)
    return AIReviewInput.model_validate(values)


def make_source_file(
    path: str = "src/auth.py",
    source: str = "def authenticate():\n    return True\n",
    *,
    blob_sha: str = "a" * 40,
    size_bytes: int | None = None,
) -> RepositorySourceFile:
    return RepositorySourceFile(
        discovered_file=DiscoveredPythonFile(
            path=path,
            sha=blob_sha,
            size_bytes=(
                len(source.encode("utf-8")) if size_bytes is None else size_bytes
            ),
        ),
        source=source,
    )


def make_failure(
    path: str = "src/missing.py",
    *,
    error_type: str = "TimeoutError",
    message: str = "Source retrieval timed out.",
) -> RepositoryFileFailure:
    return RepositoryFileFailure(
        file_path=path,
        stage="retrieval",
        error_type=error_type,
        message=message,
    )


def make_snapshot(
    *,
    commit_sha: str = "b" * 40,
    files: tuple[RepositorySourceFile, ...] | None = None,
    failures: tuple[RepositoryFileFailure, ...] = (),
    omitted_source_paths: tuple[str, ...] = (),
    tree_truncated: bool = False,
    file_limit_reached: bool = False,
    skipped_invalid_path_count: int = 0,
    skipped_oversized_file_count: int = 0,
    skipped_unknown_size_count: int = 0,
) -> RepositorySourceSnapshot:
    return RepositorySourceSnapshot(
        commit_sha=commit_sha,
        files=(make_source_file(),) if files is None else files,
        failures=failures,
        omitted_source_paths=omitted_source_paths,
        tree_truncated=tree_truncated,
        file_limit_reached=file_limit_reached,
        skipped_invalid_path_count=skipped_invalid_path_count,
        skipped_oversized_file_count=skipped_oversized_file_count,
        skipped_unknown_size_count=skipped_unknown_size_count,
    )


def fingerprint(
    *,
    base_sha: str = "c" * 40,
    review_input: AIReviewInput | None = None,
    source_snapshot: RepositorySourceSnapshot | None = None,
) -> str:
    return build_ai_execution_input_fingerprint(
        base_sha,
        make_review_input() if review_input is None else review_input,
        make_snapshot() if source_snapshot is None else source_snapshot,
    )


def test_identical_exact_inputs_produce_identical_fingerprints() -> None:
    review_input = make_review_input()
    snapshot = make_snapshot()

    first = build_ai_execution_input_fingerprint("c" * 40, review_input, snapshot)
    second = build_ai_execution_input_fingerprint("c" * 40, review_input, snapshot)

    assert first == second


def test_different_base_sha_changes_fingerprint() -> None:
    assert fingerprint(base_sha="c" * 40) != fingerprint(base_sha="d" * 40)


def test_different_validated_review_context_changes_fingerprint() -> None:
    first = make_review_input(context="Changed function:\n    return True")
    second = make_review_input(context="Changed function:\n    return False")

    assert fingerprint(review_input=first) != fingerprint(review_input=second)


def test_different_retained_source_content_changes_fingerprint() -> None:
    first = make_snapshot(files=(make_source_file(source="return True\n"),))
    second = make_snapshot(files=(make_source_file(source="return False\n"),))

    assert fingerprint(source_snapshot=first) != fingerprint(source_snapshot=second)


def test_different_snapshot_commit_sha_changes_fingerprint() -> None:
    first_input = make_review_input(head_sha="b" * 40)
    second_input = make_review_input(head_sha="e" * 40)
    first = make_snapshot(commit_sha=first_input.head_sha)
    second = make_snapshot(commit_sha=second_input.head_sha)

    assert fingerprint(
        review_input=first_input,
        source_snapshot=first,
    ) != fingerprint(review_input=second_input, source_snapshot=second)


def test_different_retrieval_failure_paths_change_fingerprint() -> None:
    first = make_snapshot(failures=(make_failure("src/first.py"),))
    second = make_snapshot(failures=(make_failure("src/second.py"),))

    assert fingerprint(source_snapshot=first) != fingerprint(source_snapshot=second)


def test_different_omitted_source_paths_change_fingerprint() -> None:
    first = make_snapshot(omitted_source_paths=("src/first.py",))
    second = make_snapshot(omitted_source_paths=("src/second.py",))

    assert fingerprint(source_snapshot=first) != fingerprint(source_snapshot=second)


@pytest.mark.parametrize(
    "metadata",
    [
        {"tree_truncated": True},
        {"file_limit_reached": True},
        {"skipped_invalid_path_count": 1},
        {"skipped_oversized_file_count": 1},
        {"skipped_unknown_size_count": 1},
    ],
)
def test_different_coverage_metadata_changes_fingerprint(
    metadata: dict[str, object],
) -> None:
    assert fingerprint() != fingerprint(source_snapshot=make_snapshot(**metadata))


def test_reordering_snapshot_files_does_not_change_fingerprint() -> None:
    first = make_snapshot(
        files=(
            make_source_file("src/z.py", "z = 1\n"),
            make_source_file("src/a.py", "a = 1\n"),
        )
    )
    reordered = make_snapshot(files=tuple(reversed(first.files)))

    assert fingerprint(source_snapshot=first) == fingerprint(source_snapshot=reordered)


def test_reordering_failure_and_omitted_path_collections_is_equivalent() -> None:
    first = make_snapshot(
        failures=(make_failure("src/z.py"), make_failure("src/a.py")),
        omitted_source_paths=("src/y.py", "src/b.py"),
    )
    reordered = make_snapshot(
        failures=tuple(reversed(first.failures)),
        omitted_source_paths=tuple(reversed(first.omitted_source_paths)),
    )

    assert fingerprint(source_snapshot=first) == fingerprint(source_snapshot=reordered)


def test_exact_source_and_context_whitespace_affects_fingerprint() -> None:
    source_with_whitespace = "def f():\r\n\treturn  1  \r\n"
    source_without_whitespace = "def f():\n    return 1\n"
    first = make_snapshot(files=(make_source_file(source=source_with_whitespace),))
    second = make_snapshot(files=(make_source_file(source=source_without_whitespace),))

    assert fingerprint(source_snapshot=first) != fingerprint(source_snapshot=second)
    assert fingerprint(
        review_input=make_review_input(context="File:\n\treturn  1  ")
    ) != fingerprint(review_input=make_review_input(context="File:\n    return 1"))


def test_fingerprint_does_not_return_raw_context_or_source() -> None:
    context = "PRIVATE_CONTEXT_MARKER\n    exact context text"
    source = "PRIVATE_SOURCE_MARKER\n    exact source text"
    result = fingerprint(
        review_input=make_review_input(context=context),
        source_snapshot=make_snapshot(files=(make_source_file(source=source),)),
    )

    assert context not in result
    assert source not in result
    assert "PRIVATE_CONTEXT_MARKER" not in result
    assert "PRIVATE_SOURCE_MARKER" not in result


def test_credentials_and_unrelated_snapshot_metadata_are_excluded() -> None:
    review_input = make_review_input()
    review_input_with_extra = review_input.model_copy(
        update={"api_key": "SECRET_CREDENTIAL", "delivery_id": "unrelated"}
    )
    ordinary_snapshot = make_snapshot(
        files=(make_source_file(blob_sha="a" * 40, size_bytes=100),),
        failures=(make_failure(error_type="TimeoutError", message="private error"),),
    )
    changed_unrelated_fields = make_snapshot(
        files=(make_source_file(blob_sha="f" * 40, size_bytes=900),),
        failures=(
            make_failure(error_type="OSError", message="different private error"),
        ),
    )

    assert fingerprint(review_input=review_input) == fingerprint(
        review_input=review_input_with_extra
    )
    assert fingerprint(source_snapshot=ordinary_snapshot) == fingerprint(
        source_snapshot=changed_unrelated_fields
    )
    assert "SECRET_CREDENTIAL" not in fingerprint(review_input=review_input_with_extra)


@pytest.mark.parametrize(
    "base_sha",
    ["", "  ", "not-a-sha", "A" * 40, "a" * 39, "a" * 64, None, 42],
)
def test_invalid_base_sha_is_rejected(base_sha: object) -> None:
    with pytest.raises(ValueError, match="base_sha"):
        build_ai_execution_input_fingerprint(
            base_sha,  # type: ignore[arg-type]
            make_review_input(),
            make_snapshot(),
        )


@pytest.mark.parametrize(
    "commit_sha",
    ["", "not-a-sha", "A" * 40, "b" * 39, f" {'b' * 40}"],
)
def test_invalid_snapshot_commit_sha_is_rejected(commit_sha: str) -> None:
    snapshot = make_snapshot()
    object.__setattr__(snapshot, "commit_sha", commit_sha)

    with pytest.raises(ValueError, match="source_snapshot.commit_sha"):
        fingerprint(source_snapshot=snapshot)


def test_snapshot_commit_sha_must_match_review_head_sha() -> None:
    snapshot = make_snapshot(commit_sha="c" * 40)

    with pytest.raises(ValueError, match="must match"):
        fingerprint(source_snapshot=snapshot)


def test_invalid_review_head_sha_is_rejected() -> None:
    review_input = make_review_input(head_sha="B" * 40)

    with pytest.raises(ValueError, match="review_input.head_sha"):
        fingerprint(review_input=review_input)


def test_noncanonical_constructed_review_head_sha_is_not_normalized() -> None:
    review_input = AIReviewInput.model_construct(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        head_sha=f" {'b' * 40} ",
        context="AI review repository context",
    )

    with pytest.raises(ValueError, match="review_input.head_sha"):
        fingerprint(review_input=review_input)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("tree_truncated", 1, "tree_truncated"),
        ("file_limit_reached", None, "file_limit_reached"),
        ("skipped_invalid_path_count", -1, "skipped-file counts"),
        ("skipped_oversized_file_count", -1, "skipped-file counts"),
        ("skipped_unknown_size_count", -1, "skipped-file counts"),
        ("skipped_invalid_path_count", True, "skipped-file counts"),
        ("skipped_oversized_file_count", True, "skipped-file counts"),
        ("skipped_unknown_size_count", False, "skipped-file counts"),
    ],
)
def test_invalid_snapshot_metadata_is_rejected(
    field_name: str,
    value: object,
    message: str,
) -> None:
    snapshot = make_snapshot()
    object.__setattr__(snapshot, field_name, value)

    with pytest.raises(ValueError, match=message):
        fingerprint(source_snapshot=snapshot)


def test_is_complete_must_be_a_real_boolean() -> None:
    class NonBooleanCompletenessSnapshot(RepositorySourceSnapshot):
        @property
        def is_complete(self) -> bool:
            return 1  # type: ignore[return-value]

    snapshot = make_snapshot()
    invalid_snapshot = NonBooleanCompletenessSnapshot(
        commit_sha=snapshot.commit_sha,
        files=snapshot.files,
        failures=snapshot.failures,
        omitted_source_paths=snapshot.omitted_source_paths,
        tree_truncated=snapshot.tree_truncated,
        file_limit_reached=snapshot.file_limit_reached,
        skipped_invalid_path_count=snapshot.skipped_invalid_path_count,
        skipped_oversized_file_count=snapshot.skipped_oversized_file_count,
        skipped_unknown_size_count=snapshot.skipped_unknown_size_count,
    )

    with pytest.raises(ValueError, match="is_complete"):
        fingerprint(source_snapshot=invalid_snapshot)


@pytest.mark.parametrize(
    "location",
    ["retained", "failure", "omitted"],
)
@pytest.mark.parametrize("path", ["../unsafe.py", r"src\unsafe.py", "/unsafe.py"])
def test_invalid_repository_paths_are_rejected(
    location: str,
    path: str,
) -> None:
    if location == "retained":
        snapshot = make_snapshot()
        object.__setattr__(snapshot.files[0].discovered_file, "path", path)
    elif location == "failure":
        snapshot = make_snapshot(failures=(make_failure(),))
        object.__setattr__(snapshot.failures[0], "file_path", path)
    else:
        snapshot = make_snapshot(omitted_source_paths=("src/omitted.py",))
        object.__setattr__(snapshot, "omitted_source_paths", (path,))

    with pytest.raises(ValueError, match="safe repository-relative POSIX path"):
        fingerprint(source_snapshot=snapshot)


def test_duplicate_retained_paths_are_rejected_at_fingerprint_time() -> None:
    first = make_source_file("src/first.py", "first = 1\n")
    second = make_source_file("src/second.py", "second = 2\n")
    snapshot = make_snapshot(files=(first, second))
    object.__setattr__(second.discovered_file, "path", first.path)

    with pytest.raises(ValueError, match="duplicate retained source paths"):
        fingerprint(source_snapshot=snapshot)


def test_valid_inputs_keep_the_pre_hardening_fingerprint() -> None:
    assert fingerprint() == (
        "ai-execution-input-v1:"
        "7b08be5ed9ee0b53fd9878cfc157c73fff6b67f974964c578390fcf9690bb732"
    )


def test_wrong_review_input_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="AIReviewInput"):
        build_ai_execution_input_fingerprint(
            "c" * 40,
            object(),  # type: ignore[arg-type]
            make_snapshot(),
        )


def test_invalid_constructed_review_input_is_rejected_without_echoing_context() -> None:
    invalid = AIReviewInput.model_construct(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        head_sha="b" * 40,
        context="PRIVATE_INVALID_CONTEXT",
    )
    object.__setattr__(invalid, "context", "")

    with pytest.raises(ValueError, match="valid AI review inputs") as error:
        fingerprint(review_input=invalid)

    assert "PRIVATE_INVALID_CONTEXT" not in str(error.value)


def test_wrong_source_snapshot_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="RepositorySourceSnapshot"):
        build_ai_execution_input_fingerprint(
            "c" * 40,
            make_review_input(),
            object(),  # type: ignore[arg-type]
        )


def test_invalid_utf8_source_is_rejected_without_echoing_source() -> None:
    snapshot = make_snapshot(
        files=(make_source_file(source="invalid \ud800", size_bytes=9),)
    )

    with pytest.raises(ValueError, match="cannot be serialized safely") as error:
        fingerprint(source_snapshot=snapshot)

    assert "invalid" not in str(error.value)


def test_fingerprint_uses_exact_versioned_lowercase_sha256_format() -> None:
    result = fingerprint()

    assert re.fullmatch(
        rf"{AI_EXECUTION_INPUT_FORMAT_VERSION}:[0-9a-f]{{64}}",
        result,
    )
