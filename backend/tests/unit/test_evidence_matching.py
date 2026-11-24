from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.evidence_matching import (
    EvidenceMatcher,
    EvidenceMatchReason,
    EvidenceMatchStatus,
)
from codeguard.ai.evidence_validation import EvidenceLocationReason
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import RepositoryFileFailure

COMMIT_SHA = "a" * 40


def make_candidate(
    *,
    file_path: str = "src/auth.py",
    line_number: int = 2,
    evidence: str = "    return True",
) -> AIFindingCandidate:
    return AIFindingCandidate(
        title="Missing authentication check",
        severity="high",
        confidence="high",
        file_path=file_path,
        line_number=line_number,
        problem="The endpoint does not verify authorization.",
        evidence=evidence,
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


def test_matcher_accepts_exact_single_line_evidence() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MATCHED
    assert result.reason == EvidenceMatchReason.EXACT_MATCH


def test_matcher_accepts_exact_multiline_evidence() -> None:
    source = (
        "def calculate():\n    first = 10\n    second = 20\n    return first + second\n"
    )

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(
            line_number=2,
            evidence="    first = 10\n    second = 20",
        ),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MATCHED
    assert result.reason == EvidenceMatchReason.EXACT_MATCH


def test_matcher_normalizes_crlf_source_line_endings() -> None:
    source = "def authenticate():\r\n    return True\r\n"

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(evidence="    return True"),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MATCHED


def test_matcher_rejects_different_source_evidence() -> None:
    source = "def authenticate():\n    return False\n"

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(evidence="    return True"),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.CONTENT_MISMATCH


def test_matcher_requires_exact_indentation() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(evidence="return True"),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.CONTENT_MISMATCH


def test_matcher_requires_evidence_at_reported_line() -> None:
    source = "def authenticate():\n    return False\n    return True\n"

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(line_number=2, evidence="    return True"),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.CONTENT_MISMATCH


def test_matcher_rejects_source_from_different_commit() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(),
        make_snapshot(),
        expected_commit_sha="c" * 40,
    )

    assert result.status == EvidenceMatchStatus.UNVERIFIABLE
    assert result.reason == EvidenceMatchReason.LOCATION_UNVERIFIABLE
    assert result.location_reason == EvidenceLocationReason.COMMIT_MISMATCH


def test_matcher_marks_retrieval_failure_unverifiable() -> None:
    failure = RepositoryFileFailure(
        file_path="src/missing.py",
        stage="retrieval",
        error_type="RuntimeError",
        message="Source retrieval failed.",
    )

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(file_path="src/missing.py"),
        make_snapshot(failures=(failure,)),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.UNVERIFIABLE
    assert result.reason == EvidenceMatchReason.LOCATION_UNVERIFIABLE
    assert result.location_reason == (EvidenceLocationReason.SOURCE_RETRIEVAL_FAILED)


def test_matcher_marks_incomplete_snapshot_unverifiable() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(file_path="src/missing.py"),
        make_snapshot(tree_truncated=True),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.UNVERIFIABLE
    assert result.reason == EvidenceMatchReason.LOCATION_UNVERIFIABLE
    assert result.location_reason == (EvidenceLocationReason.INCOMPLETE_SNAPSHOT)


def test_matcher_rejects_missing_file_in_complete_snapshot() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(file_path="src/missing.py"),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.LOCATION_INVALID
    assert result.location_reason == EvidenceLocationReason.FILE_NOT_FOUND


def test_matcher_marks_unsupported_file_type_unverifiable() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(file_path="README.md"),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )
    assert result.location_reason == (EvidenceLocationReason.UNSUPPORTED_FILE_TYPE)

    assert result.status == EvidenceMatchStatus.UNVERIFIABLE


def test_matcher_rejects_line_outside_source_boundary() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(line_number=3),
        make_snapshot(),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.LOCATION_INVALID
    assert result.location_reason == EvidenceLocationReason.LINE_OUT_OF_RANGE


def test_matcher_rejects_evidence_extending_beyond_source() -> None:
    source = "def authenticate():\n    return True"

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(
            evidence="    return True\n    raise ValueError()",
        ),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.CONTENT_MISMATCH


def test_matcher_marks_omitted_source_unverifiable() -> None:
    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(file_path="src/large.py"),
        make_snapshot(
            omitted_source_paths=("src/large.py",),
        ),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.UNVERIFIABLE
    assert result.reason == EvidenceMatchReason.LOCATION_UNVERIFIABLE
    assert result.location_reason == EvidenceLocationReason.SOURCE_OMITTED


def test_matcher_requires_trailing_newline_when_evidence_includes_it() -> None:
    source = "def authenticate():\n    return True"

    matcher = EvidenceMatcher()

    result = matcher.match(
        make_candidate(evidence="    return True\n"),
        make_snapshot(source=source),
        expected_commit_sha=COMMIT_SHA,
    )

    assert result.status == EvidenceMatchStatus.MISMATCHED
    assert result.reason == EvidenceMatchReason.CONTENT_MISMATCH
