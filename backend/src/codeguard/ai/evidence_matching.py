from dataclasses import dataclass
from enum import StrEnum

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.evidence_validation import (
    EvidenceLocationReason,
    EvidenceLocationStatus,
    EvidenceLocationValidator,
)
from codeguard.context.source_snapshot import RepositorySourceSnapshot


class EvidenceMatchStatus(StrEnum):
    """Overall outcome of independently matching AI-proposed evidence."""

    MATCHED = "matched"
    MISMATCHED = "mismatched"
    UNVERIFIABLE = "unverifiable"


class EvidenceMatchReason(StrEnum):
    """Machine-readable explanation of an evidence-matching outcome."""

    EXACT_MATCH = "exact_match"
    CONTENT_MISMATCH = "content_mismatch"
    LOCATION_INVALID = "location_invalid"
    LOCATION_UNVERIFIABLE = "location_unverifiable"


@dataclass(frozen=True, slots=True)
class EvidenceMatchResult:
    """Immutable result of matching evidence against repository source."""

    status: EvidenceMatchStatus
    reason: EvidenceMatchReason
    file_path: str
    line_number: int
    location_reason: EvidenceLocationReason | None = None


class EvidenceMatcher:
    """Verify AI-proposed source evidence against a repository snapshot."""

    def __init__(
        self,
        location_validator: EvidenceLocationValidator | None = None,
    ) -> None:
        self._location_validator = location_validator or EvidenceLocationValidator()

    def match(
        self,
        candidate: AIFindingCandidate,
        snapshot: RepositorySourceSnapshot,
        *,
        expected_commit_sha: str,
    ) -> EvidenceMatchResult:
        """Match candidate evidence at its reported repository location."""

        def result(
            status: EvidenceMatchStatus,
            reason: EvidenceMatchReason,
            location_reason: EvidenceLocationReason | None = None,
        ) -> EvidenceMatchResult:
            return EvidenceMatchResult(
                status=status,
                reason=reason,
                file_path=candidate.file_path,
                line_number=candidate.line_number,
                location_reason=location_reason,
            )

        # First independently validate the repository, file and line.
        location = self._location_validator.validate(
            candidate,
            snapshot,
            expected_commit_sha=expected_commit_sha,
        )

        if location.status == EvidenceLocationStatus.UNVERIFIABLE:
            return result(
                EvidenceMatchStatus.UNVERIFIABLE,
                EvidenceMatchReason.LOCATION_UNVERIFIABLE,
                location.reason,
            )

        if location.status == EvidenceLocationStatus.LOCATION_INVALID:
            return result(
                EvidenceMatchStatus.MISMATCHED,
                EvidenceMatchReason.LOCATION_INVALID,
                location.reason,
            )

        # A valid location must correspond to a retained source file.
        source_file = next(
            (item for item in snapshot.files if item.path == candidate.file_path),
            None,
        )

        if source_file is None:
            return result(
                EvidenceMatchStatus.UNVERIFIABLE,
                EvidenceMatchReason.LOCATION_UNVERIFIABLE,
            )

        normalized_source = source_file.source.replace("\r\n", "\n").replace("\r", "\n")

        normalized_evidence = candidate.evidence.replace("\r\n", "\n").replace(
            "\r", "\n"
        )

        source_lines = self._physical_lines(normalized_source)
        evidence_lines = self._physical_lines(normalized_evidence)

        start_index = candidate.line_number - 1
        end_index = start_index + len(evidence_lines)

        actual_lines = source_lines[start_index:end_index]

        # An evidence excerpt ending in a newline must have a corresponding
        # source line terminator. An excerpt without one may match either.
        evidence_requires_terminator = normalized_evidence.endswith("\n")

        source_has_terminator_after_match = end_index < len(
            source_lines
        ) or normalized_source.endswith("\n")

        if (
            evidence_lines
            and actual_lines == evidence_lines
            and (not evidence_requires_terminator or source_has_terminator_after_match)
        ):
            return result(
                EvidenceMatchStatus.MATCHED,
                EvidenceMatchReason.EXACT_MATCH,
            )

        return result(
            EvidenceMatchStatus.MISMATCHED,
            EvidenceMatchReason.CONTENT_MISMATCH,
        )

    @staticmethod
    def _physical_lines(value: str) -> list[str]:
        """Split physical lines while preserving meaningful source whitespace."""

        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")

        # A final newline terminates the last physical line.
        # It does not introduce another empty source line.
        if lines and lines[-1] == "":
            lines.pop()

        return lines
