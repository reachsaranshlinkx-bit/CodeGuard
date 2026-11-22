from codeguard.context.file_discovery import RepositoryFileDiscoveryResult
from codeguard.context.symbols import (
    FileSymbolIndex,
    RepositorySymbolIndex,
)


class RepositorySymbolIndexer:
    """Combine per-file symbol indexes into a repository-level index."""

    def index(
        self,
        commit_sha: str,
        discovery: RepositoryFileDiscoveryResult,
        file_indexes: tuple[FileSymbolIndex, ...],
    ) -> RepositorySymbolIndex:
        """Build a validated repository symbol index."""

        discovered_by_path = {}

        for discovered_file in discovery.files:
            path = discovered_file.path

            if path in discovered_by_path:
                raise ValueError(f"Discovery contains duplicate file paths: {path!r}.")

            discovered_by_path[path] = discovered_file

        indexes_by_path: dict[str, FileSymbolIndex] = {}

        for file_index in file_indexes:
            discovered_file = file_index.discovered_file
            path = discovered_file.path

            if path in indexes_by_path:
                raise ValueError(
                    f"Repository symbol index contains duplicate file indexes: "
                    f"{path!r}."
                )

            expected_file = discovered_by_path.get(path)

            if expected_file is None:
                raise ValueError(
                    f"File index path {path!r} is not present in repository discovery."
                )

            if discovered_file != expected_file:
                raise ValueError(
                    f"Discovered file metadata does not match for {path!r}."
                )

            indexes_by_path[path] = file_index

        missing_paths = [
            path for path in discovered_by_path if path not in indexes_by_path
        ]

        if missing_paths:
            raise ValueError(
                "Missing file indexes for discovered files: " + ", ".join(missing_paths)
            )

        # Preserve the deterministic order established by file discovery.
        ordered_indexes = tuple(
            indexes_by_path[discovered_file.path] for discovered_file in discovery.files
        )

        return RepositorySymbolIndex(
            commit_sha=commit_sha,
            files=ordered_indexes,
            tree_truncated=discovery.tree_truncated,
            file_limit_reached=discovery.file_limit_reached,
        )
