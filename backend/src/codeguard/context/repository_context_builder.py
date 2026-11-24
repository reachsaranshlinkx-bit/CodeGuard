from typing import Protocol

from codeguard.context.file_content_retriever import (
    RepositoryFileContentRetriever,
)
from codeguard.context.file_discovery import RepositoryFileDiscovery
from codeguard.context.file_indexer import FileSymbolIndexer
from codeguard.context.repository_indexer import RepositorySymbolIndexer
from codeguard.context.source_snapshot import (
    RepositoryContextBuildResult,
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import (
    FileSymbolIndex,
    RepositoryFileFailure,
    RepositorySymbolIndex,
)
from codeguard.github.schemas import (
    GitHubGitCommit,
    GitHubRepositoryTree,
)


class RepositoryContextGitHubClient(Protocol):
    """GitHub operations required to build repository context."""

    async def get_git_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> GitHubGitCommit:
        """Retrieve a commit and its root tree reference."""
        ...

    async def get_repository_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        *,
        recursive: bool = True,
    ) -> GitHubRepositoryTree:
        """Retrieve a repository tree."""
        ...

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        """Retrieve file content at a specific Git reference."""
        ...


class RepositoryContextBuilder:
    """Build repository symbol indexes and bounded source snapshots."""

    def __init__(
        self,
        github_client: RepositoryContextGitHubClient,
        *,
        max_concurrency: int = 5,
        max_files: int = 500,
        max_file_size_bytes: int = 1_000_000,
        max_source_snapshot_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(max_source_snapshot_bytes, bool)
            or not isinstance(max_source_snapshot_bytes, int)
            or max_source_snapshot_bytes < 1
        ):
            raise ValueError("max_source_snapshot_bytes must be a positive integer.")

        self._github_client = github_client
        self._max_source_snapshot_bytes = max_source_snapshot_bytes

        self._file_discovery = RepositoryFileDiscovery(
            max_files=max_files,
            max_file_size_bytes=max_file_size_bytes,
        )

        self._file_content_retriever = RepositoryFileContentRetriever(
            github_client,
            max_concurrency=max_concurrency,
        )

        self._file_indexer = FileSymbolIndexer()
        self._repository_indexer = RepositorySymbolIndexer()

    async def build(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> RepositorySymbolIndex:
        """Build a symbol index, preserving the existing public API."""

        result = await self.build_with_source(
            owner=owner,
            repo=repo,
            commit_sha=commit_sha,
        )

        return result.symbol_index

    async def build_with_source(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> RepositoryContextBuildResult:
        """Build a symbol index and bounded source snapshot."""

        # 1. Resolve the requested commit.
        commit = await self._github_client.get_git_commit(
            owner,
            repo,
            commit_sha,
        )

        # 2. Retrieve the tree belonging to that commit.
        repository_tree = await self._github_client.get_repository_tree(
            owner,
            repo,
            commit.tree.sha,
            recursive=True,
        )

        # 3. Discover eligible Python files.
        discovery = self._file_discovery.discover(repository_tree)

        # 4. Retrieve files independently.
        retrieval_result = await self._file_content_retriever.retrieve_partial(
            owner=owner,
            repo=repo,
            commit_sha=commit.sha,
            discovery=discovery,
        )

        file_indexes: list[FileSymbolIndex] = []
        failures: list[RepositoryFileFailure] = list(retrieval_result.failures)

        source_files: list[RepositorySourceFile] = []
        omitted_source_paths: list[str] = []

        remaining_bytes = self._max_source_snapshot_bytes

        # 5. Preserve retrieved source deterministically within the budget.
        # Discovery and retrieval preserve their deterministic file ordering.
        for retrieved_file in retrieval_result.retrieved_files:
            discovered_file = retrieved_file.discovered_file
            source_bytes = len(retrieved_file.source.encode("utf-8"))

            if source_bytes <= remaining_bytes:
                source_files.append(
                    RepositorySourceFile(
                        discovered_file=discovered_file,
                        source=retrieved_file.source,
                    )
                )
                remaining_bytes -= source_bytes
            else:
                omitted_source_paths.append(discovered_file.path)

            # 6. Analyze every retrieved file, even when its source was
            # omitted from the retained snapshot.
            # The source snapshot and AST index have separate responsibilities.
            try:
                file_index = self._file_indexer.index(
                    discovered_file,
                    retrieved_file.source,
                )
            except SyntaxError as exc:
                failures.append(
                    RepositoryFileFailure(
                        file_path=discovered_file.path,
                        stage="analysis",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            file_indexes.append(file_index)

        # 7. Build the original symbol index.
        symbol_index = self._repository_indexer.index(
            commit_sha=commit.sha,
            discovery=discovery,
            file_indexes=tuple(file_indexes),
            failures=tuple(failures),
        )

        # 8. Build the independent source snapshot.
        source_snapshot = RepositorySourceSnapshot(
            commit_sha=commit.sha,
            files=tuple(source_files),
            failures=retrieval_result.failures,
            omitted_source_paths=tuple(omitted_source_paths),
            tree_truncated=discovery.tree_truncated,
            file_limit_reached=discovery.file_limit_reached,
            skipped_invalid_path_count=discovery.skipped_invalid_path_count,
            skipped_oversized_file_count=discovery.skipped_oversized_file_count,
            skipped_unknown_size_count=discovery.skipped_unknown_size_count,
        )

        return RepositoryContextBuildResult(
            symbol_index=symbol_index,
            source_snapshot=source_snapshot,
        )
