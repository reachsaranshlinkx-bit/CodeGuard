from dataclasses import dataclass

from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.symbols import (
    RepositoryFileFailure,
    RepositorySymbolIndex,
)


def _validate_repository_path(path: str) -> None:
    """Ensure a path is safe and repository-relative."""

    invalid = (
        not path.strip()
        or "\\" in path
        or "\x00" in path
        or path.startswith("/")
        or (len(path) > 1 and path[1] == ":")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    )

    if invalid:
        raise ValueError("Path must be a safe repository-relative POSIX path.")


@dataclass(frozen=True, slots=True)
class RepositorySourceFile:
    """Original source contents associated with a discovered repository file."""

    discovered_file: DiscoveredPythonFile
    source: str

    def __post_init__(self) -> None:
        """Validate the retained source file."""

        _validate_repository_path(self.discovered_file.path)

        if not isinstance(self.source, str):
            raise TypeError("Repository source must be a string.")

    @property
    def path(self) -> str:
        """Return the repository-relative file path."""

        return self.discovered_file.path

    @property
    def blob_sha(self) -> str:
        """Return the GitHub blob SHA."""

        return self.discovered_file.sha

    @property
    def source_size_bytes(self) -> int:
        """Return the UTF-8 size of the retained source."""

        return len(self.source.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class RepositorySourceSnapshot:
    """Bounded source files retrieved from one repository commit."""

    commit_sha: str
    files: tuple[RepositorySourceFile, ...]

    failures: tuple[RepositoryFileFailure, ...] = ()
    omitted_source_paths: tuple[str, ...] = ()

    tree_truncated: bool = False
    file_limit_reached: bool = False

    skipped_invalid_path_count: int = 0
    skipped_oversized_file_count: int = 0
    skipped_unknown_size_count: int = 0

    def __post_init__(self) -> None:
        """Validate snapshot identity, paths and coverage metadata."""

        if not isinstance(self.commit_sha, str) or not self.commit_sha.strip():
            raise ValueError("Repository source snapshot commit SHA must not be empty.")

        counts = (
            self.skipped_invalid_path_count,
            self.skipped_oversized_file_count,
            self.skipped_unknown_size_count,
        )

        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ):
            raise ValueError("Skipped file counts must be non-negative integers.")

        file_paths = tuple(source_file.path for source_file in self.files)
        failure_paths = tuple(failure.file_path for failure in self.failures)
        omitted_paths = self.omitted_source_paths

        for path in (*failure_paths, *omitted_paths):
            _validate_repository_path(path)

        if len(file_paths) != len(set(file_paths)):
            raise ValueError("Source snapshot contains duplicate file paths.")

        if len(failure_paths) != len(set(failure_paths)):
            raise ValueError("Source snapshot contains duplicate failures.")

        if len(omitted_paths) != len(set(omitted_paths)):
            raise ValueError("Source snapshot contains duplicate omitted paths.")

        if set(file_paths) & set(failure_paths):
            raise ValueError("A source file cannot also have a retrieval failure.")

        if set(file_paths) & set(omitted_paths):
            raise ValueError("A retained source file cannot also be omitted.")

        if set(failure_paths) & set(omitted_paths):
            raise ValueError("A failed source file cannot also be omitted.")

        if any(failure.stage != "retrieval" for failure in self.failures):
            raise ValueError("Source snapshot failures must be retrieval failures.")

    @property
    def total_source_bytes(self) -> int:
        """Return the total number of retained UTF-8 source bytes."""

        return sum(source_file.source_size_bytes for source_file in self.files)

    @property
    def is_complete(self) -> bool:
        """Indicate whether the supported source snapshot has full coverage."""

        return (
            not self.tree_truncated
            and not self.file_limit_reached
            and not self.failures
            and not self.omitted_source_paths
            and self.skipped_invalid_path_count == 0
            and self.skipped_oversized_file_count == 0
            and self.skipped_unknown_size_count == 0
        )


@dataclass(frozen=True, slots=True)
class RepositoryContextBuildResult:
    """Combined source and symbol context for one repository commit."""

    symbol_index: RepositorySymbolIndex
    source_snapshot: RepositorySourceSnapshot

    def __post_init__(self) -> None:
        """Validate commit identity and source/index consistency."""

        if self.symbol_index.commit_sha != self.source_snapshot.commit_sha:
            raise ValueError(
                "Symbol index and source snapshot must use the same commit SHA."
            )

        symbol_files = {
            item.discovered_file.path: item for item in self.symbol_index.files
        }
        symbol_failures = {
            failure.file_path: failure for failure in self.symbol_index.failures
        }

        source_files = {item.path: item for item in self.source_snapshot.files}
        source_failures = {
            failure.file_path: failure for failure in self.source_snapshot.failures
        }
        omitted_paths = set(self.source_snapshot.omitted_source_paths)

        symbol_paths = set(symbol_files) | set(symbol_failures)
        snapshot_paths = set(source_files) | set(source_failures) | omitted_paths

        if symbol_paths != snapshot_paths:
            raise ValueError(
                "Source snapshot and symbol index must account for "
                "the same discovered files."
            )

        # Retrieval failures must be represented identically in both results.
        for path, failure in source_failures.items():
            if symbol_failures.get(path) != failure:
                raise ValueError(
                    "Source snapshot retrieval failure records must match "
                    "symbol index failures."
                )

        for path, failure in symbol_failures.items():
            if failure.stage == "retrieval":
                if source_failures.get(path) != failure:
                    raise ValueError(
                        "Symbol index retrieval failures must appear "
                        "in the source snapshot."
                    )

            elif path not in source_files and path not in omitted_paths:
                raise ValueError(
                    "Analysis failures must have retained or omitted source."
                )

        # For successfully indexed files, both representations must refer
        # to the same discovered GitHub file metadata.
        for path, source_file in source_files.items():
            symbol_file = symbol_files.get(path)

            if (
                symbol_file is not None
                and symbol_file.discovered_file != source_file.discovered_file
            ):
                raise ValueError(
                    f"Source and symbol file metadata must match for {path!r}."
                )
