import pytest

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.evidence_validation import (
    EvidenceLocationReason,
    EvidenceLocationStatus,
    EvidenceLocationValidator,
)
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import RepositoryFileFailure

COMMIT_SHA = "a" * 40


def make_candidate(
    file_path: str = "src/auth.py",
    line_number: int = 2,
) -> AIFindingCandidate:
    return AIFindingCandidate(
        title="Missing authentication check",
        severity="high",
        confidence="medium",
        file_path=file_path,
        line_number=line_number,
        problem="The endpoint does not verify authorization.",
        evidence="return True",
        why_it_matters="Unauthorized access may occur.",
        suggested_fix="Add an authorization check.",
        recommended_validation="Test unauthorized requests.",
    )


def make_snapshot(
    source: str = "def authenticate():\n    return True\n",
    *,
    path: str = "src/auth.py",
    tree_truncated: bool = False,
    failures: tuple[RepositoryFileFailure, ...] = (),
    omitted_source_paths: tuple[str, ...] = (),
) -> RepositorySourceSnapshot:
    discovered_file = DiscoveredPythonFile(
        path=path,
        sha="b" * 40,
        size_bytes=len(source.encode("utf-8")),
    )

    return RepositorySourceSnapshot(
        commit_sha=COMMIT_SHA,
        files=(RepositorySourceFile(discovered_file, source),),
        tree_truncated=tree_truncated,
        failures=failures,
        omitted_source_paths=omitted_source_paths,
    )


def test_validator_accepts_existing_file_and_valid_line() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(line_number=2),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_VALID
    assert result.reason == EvidenceLocationReason.LOCATION_VALID
    assert result.source_line_count == 2


def test_validator_rejects_line_beyond_source_boundary() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(line_number=3),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_INVALID
    assert result.reason == EvidenceLocationReason.LINE_OUT_OF_RANGE
    assert result.source_line_count == 2


def test_validator_rejects_missing_file_in_complete_snapshot() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(file_path="src/missing.py"),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_INVALID
    assert result.reason == EvidenceLocationReason.FILE_NOT_FOUND


def test_validator_marks_missing_file_in_incomplete_snapshot_unverifiable() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(file_path="src/missing.py"),
        make_snapshot(tree_truncated=True),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.UNVERIFIABLE
    assert result.reason == EvidenceLocationReason.INCOMPLETE_SNAPSHOT


def test_validator_handles_retrieval_failure() -> None:
    failure = RepositoryFileFailure(
        file_path="src/missing.py",
        stage="retrieval",
        error_type="RuntimeError",
        message="Retrieval failed.",
    )

    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(file_path="src/missing.py"),
        make_snapshot(failures=(failure,)),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.UNVERIFIABLE
    assert result.reason == EvidenceLocationReason.SOURCE_RETRIEVAL_FAILED


def test_validator_handles_budget_omitted_file() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(file_path="src/large.py"),
        make_snapshot(omitted_source_paths=("src/large.py",)),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.UNVERIFIABLE
    assert result.reason == EvidenceLocationReason.SOURCE_OMITTED


def test_validator_does_not_reject_missing_file_in_partial_snapshot() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(file_path="src/other.py"),
        make_snapshot(tree_truncated=True),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.UNVERIFIABLE


def test_validator_does_not_claim_support_for_other_file_types() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(file_path="README.md"),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.UNVERIFIABLE
    assert result.reason == EvidenceLocationReason.UNSUPPORTED_FILE_TYPE


def test_validator_rejects_snapshot_from_different_commit() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(),
        make_snapshot(),
        expected_commit_sha="c" * 40,
    )

    assert result.status == EvidenceLocationStatus.UNVERIFIABLE
    assert result.reason == EvidenceLocationReason.COMMIT_MISMATCH


def test_validator_handles_empty_source_file() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(line_number=1),
        make_snapshot(source=""),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_INVALID
    assert result.reason == EvidenceLocationReason.LINE_OUT_OF_RANGE
    assert result.source_line_count == 0


def test_validator_returns_line_count_for_multiline_source() -> None:
    source = "first = 1\nsecond = 2\nthird = 3\n"

    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(line_number=3),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_VALID
    assert result.source_line_count == 3


def test_validator_requires_expected_commit_sha() -> None:
    validator = EvidenceLocationValidator()

    with pytest.raises(ValueError, match="Expected commit SHA"):
        validator.validate(
            make_candidate(),
            make_snapshot(),
            expected_commit_sha="",
        )


def test_validator_accepts_uppercase_python_extension() -> None:
    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(
            file_path="src/auth.PY",
            line_number=2,
        ),
        make_snapshot(path="src/auth.PY"),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_VALID
    assert result.reason == EvidenceLocationReason.LOCATION_VALID


def test_validator_counts_physical_lines_not_unicode_separators() -> None:
    source = 'message = "first\u2028second"\nnext = 1\n'

    validator = EvidenceLocationValidator()

    result = validator.validate(
        make_candidate(line_number=3),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceLocationStatus.LOCATION_INVALID
    assert result.reason == EvidenceLocationReason.LINE_OUT_OF_RANGE
    assert result.source_line_count == 2
