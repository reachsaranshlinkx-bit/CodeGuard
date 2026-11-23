from typing import Protocol

from codeguard.context.file_content_retriever import (
    RepositoryFileContentRetriever,
)
from codeguard.context.file_discovery import RepositoryFileDiscovery
from codeguard.context.file_indexer import FileSymbolIndexer
from codeguard.context.repository_indexer import RepositorySymbolIndexer
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
    """Build a repository-level Python symbol index from GitHub."""

    def __init__(
        self,
        github_client: RepositoryContextGitHubClient,
        *,
        max_concurrency: int = 5,
        max_files: int = 500,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        self._github_client = github_client

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
        """Build a repository index while preserving file-level failures."""

        # 1. Resolve the requested commit.
        commit = await self._github_client.get_git_commit(
            owner,
            repo,
            commit_sha,
        )

        # 2. Retrieve the root tree belonging to that commit.
        repository_tree = await self._github_client.get_repository_tree(
            owner,
            repo,
            commit.tree.sha,
            recursive=True,
        )

        # 3. Discover eligible Python files.
        discovery = self._file_discovery.discover(repository_tree)

        # 4. Retrieve files independently, preserving successful results.
        retrieval_result = await self._file_content_retriever.retrieve_partial(
            owner=owner,
            repo=repo,
            commit_sha=commit.sha,
            discovery=discovery,
        )

        file_indexes: list[FileSymbolIndex] = []
        failures: list[RepositoryFileFailure] = list(retrieval_result.failures)

        # 5. Analyze each successfully retrieved file independently.
        for retrieved_file in retrieval_result.retrieved_files:
            discovered_file = retrieved_file.discovered_file

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

        # 6. Combine successful indexes and all file failures.
        return self._repository_indexer.index(
            commit_sha=commit.sha,
            discovery=discovery,
            file_indexes=tuple(file_indexes),
            failures=tuple(failures),
        )
