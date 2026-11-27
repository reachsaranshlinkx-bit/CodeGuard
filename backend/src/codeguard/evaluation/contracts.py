from __future__ import annotations

from datetime import datetime
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.ai.evidence_matching import (
    EvidenceMatcher,
    EvidenceMatchStatus,
)
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.review.changed_lines import parse_added_line_numbers

EVALUATION_SCHEMA_VERSION = "1.0"
_CASE_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_COMMIT_SHA_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"

CaseIdentifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100, pattern=_CASE_ID_PATTERN),
]
FindingIdentifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100, pattern=_CASE_ID_PATTERN),
]
NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class EvaluationSourceFile(_EvaluationModel):
    """Source content for one file in a synthetic repository snapshot."""

    path: str = Field(min_length=1, max_length=500)
    content: str

    @model_validator(mode="after")
    def validate_path(self) -> EvaluationSourceFile:
        _validate_python_path(self.path)
        return self


class EvaluationSourceSnapshot(_EvaluationModel):
    """Immutable base or head source tree for an evaluation pull request."""

    commit_sha: str = Field(pattern=_COMMIT_SHA_PATTERN)
    files: tuple[EvaluationSourceFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_paths(self) -> EvaluationSourceSnapshot:
        paths = tuple(source_file.path for source_file in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("A source snapshot cannot contain duplicate file paths.")
        return self


class EvaluationPullRequest(_EvaluationModel):
    """Synthetic pull request metadata supplied with a dataset case."""

    title: NonBlankText = Field(max_length=300)
    description: NonBlankText = Field(max_length=10_000)


class EvaluationChangedFile(_EvaluationModel):
    """Changed-file metadata and unified diff for one synthetic PR file."""

    path: str = Field(min_length=1, max_length=500)
    status: Literal["added", "modified", "removed"]
    additions: int = Field(ge=0, strict=True)
    deletions: int = Field(ge=0, strict=True)
    patch: str = Field(min_length=1, max_length=50_000)

    @model_validator(mode="after")
    def validate_change(self) -> EvaluationChangedFile:
        _validate_python_path(self.path)
        if self.additions + self.deletions == 0:
            raise ValueError("A changed file must contain at least one changed line.")
        if "@@" not in self.patch:
            raise ValueError("A changed-file patch must contain a unified diff hunk.")
        return self


class ExpectedFinding(_EvaluationModel):
    """Human-authored ground truth reusing the production finding contract."""

    finding_id: FindingIdentifier
    finding: AIFindingCandidate
    rationale: NonBlankText = Field(max_length=2_000)


class EvaluationCase(_EvaluationModel):
    """One labelled synthetic pull request and its source snapshots."""

    case_id: CaseIdentifier
    category: Literal[
        "security",
        "reliability",
        "error_handling",
        "edge_case",
        "safe_pattern",
        "negative",
    ]
    case_type: Literal["positive", "negative"]
    description: NonBlankText = Field(max_length=2_000)
    pull_request: EvaluationPullRequest
    base_snapshot: EvaluationSourceSnapshot
    head_snapshot: EvaluationSourceSnapshot
    changed_files: tuple[EvaluationChangedFile, ...] = Field(min_length=1)
    expected_findings: tuple[ExpectedFinding, ...] = Field(max_length=50)
    recommended_test_scenarios: tuple[NonBlankText, ...] = Field(
        min_length=1,
        max_length=20,
    )
    analysis_limitations: tuple[NonBlankText, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_case_references(self) -> EvaluationCase:
        if self.base_snapshot.commit_sha == self.head_snapshot.commit_sha:
            raise ValueError("Base and head snapshots must have distinct commit SHAs.")

        base_sources = {
            source_file.path: source_file.content
            for source_file in self.base_snapshot.files
        }
        head_sources = {
            source_file.path: source_file.content
            for source_file in self.head_snapshot.files
        }
        changes_by_path = {
            changed_file.path: changed_file for changed_file in self.changed_files
        }
        if len(changes_by_path) != len(self.changed_files):
            raise ValueError("A case cannot contain duplicate changed-file paths.")

        for changed_file in self.changed_files:
            in_base = changed_file.path in base_sources
            in_head = changed_file.path in head_sources
            if changed_file.status == "added" and (in_base or not in_head):
                raise ValueError(
                    f"Added file {changed_file.path!r} must exist only in the "
                    "head snapshot."
                )
            if changed_file.status == "modified" and not (in_base and in_head):
                raise ValueError(
                    f"Modified file {changed_file.path!r} must exist in both snapshots."
                )
            if changed_file.status == "removed" and (not in_base or in_head):
                raise ValueError(
                    f"Removed file {changed_file.path!r} must exist only in the "
                    "base snapshot."
                )

        finding_ids = tuple(expected.finding_id for expected in self.expected_findings)
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("Ground-truth finding IDs must be unique within a case.")

        if self.case_type == "negative" and self.expected_findings:
            raise ValueError("Negative cases cannot contain labelled findings.")
        if self.case_type == "positive" and not self.expected_findings:
            raise ValueError(
                "Positive cases must contain at least one labelled finding."
            )

        source_snapshot = _repository_snapshot(self.head_snapshot)
        matcher = EvidenceMatcher()
        for expected in self.expected_findings:
            finding = expected.finding
            finding_changed_file = changes_by_path.get(finding.file_path)
            if finding_changed_file is None or finding_changed_file.status == "removed":
                raise ValueError(
                    f"Ground-truth finding {expected.finding_id!r} must reference "
                    "an added or modified file."
                )
            evidence_match = matcher.match(
                finding,
                source_snapshot,
                expected_commit_sha=self.head_snapshot.commit_sha,
            )
            if evidence_match.status is not EvidenceMatchStatus.MATCHED:
                raise ValueError(
                    f"Ground-truth finding {expected.finding_id!r} has invalid "
                    "source evidence or line reference "
                    f"({evidence_match.reason.value})."
                )
            if finding.line_number not in parse_added_line_numbers(
                finding_changed_file.patch
            ):
                raise ValueError(
                    f"Ground-truth finding {expected.finding_id!r} must reference "
                    "an added source line in the file diff."
                )

        return self


class EvaluationDataset(_EvaluationModel):
    """Versioned golden dataset with explicitly bounded evaluation scope."""

    schema_version: Literal["1.0"]
    dataset_id: CaseIdentifier
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: NonBlankText = Field(max_length=200)
    description: NonBlankText = Field(max_length=2_000)
    scope_statement: NonBlankText = Field(max_length=2_000)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> EvaluationDataset:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation dataset case IDs must be unique.")
        return self


class FindingMatch(_EvaluationModel):
    """Explicit one-to-one association between one prediction and one label."""

    predicted_finding_index: int = Field(ge=0, strict=True)
    ground_truth_finding_id: FindingIdentifier


class EvaluationResult(_EvaluationModel):
    """Per-case predictions and explicit prediction-to-label associations.

    Predictions and labels absent from ``finding_matches`` are false positives
    and missed findings, respectively.
    """

    case_id: CaseIdentifier
    predicted_findings: tuple[AIFindingCandidate, ...] = Field(max_length=50)
    finding_matches: tuple[FindingMatch, ...] = ()

    @model_validator(mode="after")
    def validate_unique_finding_matches(self) -> EvaluationResult:
        predicted_indexes = tuple(
            match.predicted_finding_index for match in self.finding_matches
        )
        ground_truth_ids = tuple(
            match.ground_truth_finding_id for match in self.finding_matches
        )
        if len(predicted_indexes) != len(set(predicted_indexes)):
            raise ValueError("A predicted finding cannot match multiple labels.")
        if len(ground_truth_ids) != len(set(ground_truth_ids)):
            raise ValueError(
                "A ground-truth finding cannot match multiple predictions."
            )
        return self


class EvaluationRun(_EvaluationModel):
    """Provider-neutral results tied to an exact versioned dataset snapshot."""

    run_id: CaseIdentifier
    dataset: EvaluationDataset
    started_at: datetime
    system_name: NonBlankText = Field(max_length=200)
    system_version: NonBlankText = Field(max_length=100)
    is_complete: bool = True
    results: tuple[EvaluationResult, ...]

    @model_validator(mode="after")
    def validate_result_references(self) -> EvaluationRun:
        cases_by_id = {case.case_id: case for case in self.dataset.cases}
        result_case_ids = tuple(result.case_id for result in self.results)
        if len(result_case_ids) != len(set(result_case_ids)):
            raise ValueError("An evaluation run cannot contain duplicate case results.")

        for result in self.results:
            case = cases_by_id.get(result.case_id)
            if case is None:
                raise ValueError(
                    f"Evaluation result references unknown case {result.case_id!r}."
                )

            expected_ids = {finding.finding_id for finding in case.expected_findings}
            for match in result.finding_matches:
                if match.ground_truth_finding_id not in expected_ids:
                    raise ValueError(
                        f"Evaluation result for {result.case_id!r} references an "
                        "unknown ground-truth finding."
                    )
                if match.predicted_finding_index >= len(result.predicted_findings):
                    raise ValueError(
                        "Finding match indexes must reference a predicted finding."
                    )

        if self.is_complete:
            missing_case_ids = set(cases_by_id) - set(result_case_ids)
            if missing_case_ids:
                raise ValueError(
                    "A complete evaluation run must include every dataset case."
                )

        return self


def load_evaluation_dataset(path: Path | None = None) -> EvaluationDataset:
    """Load a versioned dataset file or the packaged Python PR golden dataset."""

    if path is None:
        dataset_text = (
            files("codeguard.evaluation")
            .joinpath("datasets", "python_pr_v1.json")
            .read_text(encoding="utf-8")
        )
    else:
        dataset_text = path.read_text(encoding="utf-8")
    return EvaluationDataset.model_validate_json(dataset_text)


def _validate_python_path(path: str) -> None:
    invalid = (
        not path.strip()
        or path != path.strip()
        or "\\" in path
        or "\x00" in path
        or path.startswith("/")
        or (len(path) > 1 and path[1] == ":")
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or PurePosixPath(path).suffix.lower() != ".py"
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    )
    if invalid:
        raise ValueError(
            "Evaluation paths must be safe repository-relative Python paths."
        )


def _repository_snapshot(
    snapshot: EvaluationSourceSnapshot,
) -> RepositorySourceSnapshot:
    files_for_snapshot = tuple(
        RepositorySourceFile(
            discovered_file=DiscoveredPythonFile(
                path=source_file.path,
                sha="0" * 40,
                size_bytes=len(source_file.content.encode("utf-8")),
            ),
            source=source_file.content,
        )
        for source_file in snapshot.files
    )
    return RepositorySourceSnapshot(
        commit_sha=snapshot.commit_sha,
        files=files_for_snapshot,
    )
