from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.context.source_snapshot import RepositorySourceSnapshot


class EvidenceLocationStatus(StrEnum):
    """Result of independently checking an AI finding's location."""

    LOCATION_VALID = "location_valid"
    LOCATION_INVALID = "location_invalid"
    UNVERIFIABLE = "unverifiable"


class EvidenceLocationReason(StrEnum):
    """Machine-readable reason for a location validation result."""

    LOCATION_VALID = "location_valid"
    FILE_NOT_FOUND = "file_not_found"
    SOURCE_RETRIEVAL_FAILED = "source_retrieval_failed"
    SOURCE_OMITTED = "source_omitted"
    INCOMPLETE_SNAPSHOT = "incomplete_snapshot"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    LINE_OUT_OF_RANGE = "line_out_of_range"
    COMMIT_MISMATCH = "commit_mismatch"


@dataclass(frozen=True, slots=True)
class EvidenceLocationResult:
    """Immutable result of validating an AI finding's source location."""

    status: EvidenceLocationStatus
    reason: EvidenceLocationReason
    file_path: str
    line_number: int
    source_line_count: int | None = None


class EvidenceLocationValidator:
    """Verify AI finding file paths and line numbers against a source snapshot."""

    def validate(
        self,
        candidate: AIFindingCandidate,
        snapshot: RepositorySourceSnapshot,
        *,
        expected_commit_sha: str,
    ) -> EvidenceLocationResult:
        """Validate a candidate against the expected repository commit."""

        if not expected_commit_sha.strip():
            raise ValueError("Expected commit SHA must not be empty.")

        def result(
            status: EvidenceLocationStatus,
            reason: EvidenceLocationReason,
            line_count: int | None = None,
        ) -> EvidenceLocationResult:
            return EvidenceLocationResult(
                status=status,
                reason=reason,
                file_path=candidate.file_path,
                line_number=candidate.line_number,
                source_line_count=line_count,
            )

        # Never validate against a different repository revision.
        if snapshot.commit_sha != expected_commit_sha:
            return result(
                EvidenceLocationStatus.UNVERIFIABLE,
                EvidenceLocationReason.COMMIT_MISMATCH,
            )

        path = candidate.file_path

        # This validator currently supports Python source files only.
        if PurePosixPath(path).suffix.lower() != ".py":
            return result(
                EvidenceLocationStatus.UNVERIFIABLE,
                EvidenceLocationReason.UNSUPPORTED_FILE_TYPE,
            )

        source_files = {source_file.path: source_file for source_file in snapshot.files}

        source_file = source_files.get(path)

        # A missing file is not necessarily an invalid finding.
        if source_file is None:
            failure_paths = {failure.file_path for failure in snapshot.failures}

            if path in failure_paths:
                return result(
                    EvidenceLocationStatus.UNVERIFIABLE,
                    EvidenceLocationReason.SOURCE_RETRIEVAL_FAILED,
                )

            if path in snapshot.omitted_source_paths:
                return result(
                    EvidenceLocationStatus.UNVERIFIABLE,
                    EvidenceLocationReason.SOURCE_OMITTED,
                )

            if not snapshot.is_complete:
                return result(
                    EvidenceLocationStatus.UNVERIFIABLE,
                    EvidenceLocationReason.INCOMPLETE_SNAPSHOT,
                )

            return result(
                EvidenceLocationStatus.LOCATION_INVALID,
                EvidenceLocationReason.FILE_NOT_FOUND,
            )

        # Python source line numbers are 1-based.
        normalized_source = source_file.source.replace("\r\n", "\n").replace("\r", "\n")

        source_lines = normalized_source.split("\n")

        # A final newline terminates the last physical line;
        # it does not create another source line.
        if source_lines and source_lines[-1] == "":
            source_lines.pop()

        line_count = len(source_lines)

        if candidate.line_number > line_count:
            return result(
                EvidenceLocationStatus.LOCATION_INVALID,
                EvidenceLocationReason.LINE_OUT_OF_RANGE,
                line_count,
            )

        return result(
            EvidenceLocationStatus.LOCATION_VALID,
            EvidenceLocationReason.LOCATION_VALID,
            line_count,
        )
