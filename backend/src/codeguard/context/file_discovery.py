from dataclasses import dataclass
from pathlib import PurePosixPath

from codeguard.github.schemas import GitHubRepositoryTree

EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        "site-packages",
        "dist",
        "build",
    }
)


@dataclass(frozen=True, slots=True)
class DiscoveredPythonFile:
    """A Python file selected from a repository tree."""

    path: str
    sha: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RepositoryFileDiscoveryResult:
    """Result of discovering relevant Python repository files."""

    tree_sha: str
    files: tuple[DiscoveredPythonFile, ...]
    tree_truncated: bool
    file_limit_reached: bool
    skipped_invalid_path_count: int
    skipped_oversized_file_count: int
    skipped_unknown_size_count: int

    @property
    def is_complete(self) -> bool:
        """Indicate whether discovery covered the complete tree."""

        return not self.tree_truncated and not self.file_limit_reached


class RepositoryFileDiscovery:
    """Discover relevant Python files from a GitHub repository tree."""

    def __init__(
        self,
        max_files: int = 500,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        if (
            isinstance(max_files, bool)
            or not isinstance(max_files, int)
            or max_files < 1
        ):
            raise ValueError("max_files must be a positive integer.")

        if (
            isinstance(max_file_size_bytes, bool)
            or not isinstance(max_file_size_bytes, int)
            or max_file_size_bytes < 1
        ):
            raise ValueError("max_file_size_bytes must be a positive integer.")

        self._max_files = max_files
        self._max_file_size_bytes = max_file_size_bytes

    @staticmethod
    def _is_safe_relative_path(path: str) -> bool:
        """Reject absolute paths and unsafe path components."""

        if not path or "\\" in path or "\x00" in path:
            return False

        if path.startswith("/"):
            return False

        parts = path.split("/")

        if any(part in {"", ".", ".."} for part in parts):
            return False

        return True

    @staticmethod
    def _is_excluded(path: str) -> bool:
        """Check whether any path component is excluded."""

        parts = PurePosixPath(path).parts

        return any(part in EXCLUDED_DIRECTORIES for part in parts)

    def discover(
        self,
        repository_tree: GitHubRepositoryTree,
    ) -> RepositoryFileDiscoveryResult:
        """Select safe, eligible Python files from a repository tree."""

        candidates: list[DiscoveredPythonFile] = []

        skipped_invalid_paths = 0
        skipped_oversized = 0
        skipped_unknown_size = 0

        for entry in repository_tree.tree:
            # Only regular, non-executable or executable Git files
            # are eligible. Exclude symbolic links and other objects.
            if entry.type != "blob" or entry.mode not in {
                "100644",
                "100755",
            }:
                continue

            if not self._is_safe_relative_path(entry.path):
                skipped_invalid_paths += 1
                continue

            if PurePosixPath(entry.path).suffix.lower() != ".py":
                continue

            if self._is_excluded(entry.path):
                continue

            if entry.size is None:
                skipped_unknown_size += 1
                continue

            if entry.size > self._max_file_size_bytes:
                skipped_oversized += 1
                continue

            candidates.append(
                DiscoveredPythonFile(
                    path=entry.path,
                    sha=entry.sha,
                    size_bytes=entry.size,
                )
            )

        # Stable ordering makes the output deterministic.
        candidates.sort(key=lambda item: item.path)

        file_limit_reached = len(candidates) > self._max_files

        selected_files = tuple(candidates[: self._max_files])

        return RepositoryFileDiscoveryResult(
            tree_sha=repository_tree.sha,
            files=selected_files,
            tree_truncated=repository_tree.truncated,
            file_limit_reached=file_limit_reached,
            skipped_invalid_path_count=skipped_invalid_paths,
            skipped_oversized_file_count=skipped_oversized,
            skipped_unknown_size_count=skipped_unknown_size,
        )
