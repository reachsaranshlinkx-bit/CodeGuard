from dataclasses import dataclass
from typing import Literal

from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.context.file_discovery import DiscoveredPythonFile

FailureStage = Literal["retrieval", "analysis"]


@dataclass(frozen=True, slots=True)
class RepositoryFileFailure:
    """Failure encountered while processing a repository file."""

    file_path: str
    stage: FailureStage
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class FileSymbolIndex:
    """Symbol analysis associated with one discovered repository file."""

    discovered_file: DiscoveredPythonFile
    analysis: PythonAnalysisResult

    def __post_init__(self) -> None:
        """Ensure the analysis belongs to the discovered file."""

        if self.analysis.file_path != self.discovered_file.path:
            raise ValueError("Analysis file path must match the discovered file path.")


@dataclass(frozen=True, slots=True)
class RepositorySymbolIndex:
    """Combined symbol indexes for a specific repository commit."""

    commit_sha: str
    files: tuple[FileSymbolIndex, ...]
    tree_truncated: bool = False
    file_limit_reached: bool = False
    failures: tuple[RepositoryFileFailure, ...] = ()

    def __post_init__(self) -> None:
        """Validate repository identity and file uniqueness."""

        if not self.commit_sha.strip():
            raise ValueError("Repository commit SHA must not be empty.")

        file_paths = tuple(file_index.discovered_file.path for file_index in self.files)

        if len(file_paths) != len(set(file_paths)):
            raise ValueError("Repository symbol index contains duplicate file paths.")

        failure_paths = tuple(failure.file_path for failure in self.failures)

        if len(failure_paths) != len(set(failure_paths)):
            raise ValueError(
                "Repository symbol index contains duplicate file failures."
            )

        overlapping_paths = set(file_paths) & set(failure_paths)

        if overlapping_paths:
            raise ValueError(
                "A file cannot have both a successful index and a failure."
            )

    @property
    def is_complete(self) -> bool:
        """Indicate whether every discovered file was successfully indexed."""

        return (
            not self.tree_truncated
            and not self.file_limit_reached
            and not self.failures
        )
