from typing import Protocol

from codeguard.context.file_content_retriever import (
    RepositoryFileContentRetriever,
)
from codeguard.context.file_discovery import RepositoryFileDiscovery
from codeguard.context.file_indexer import FileSymbolIndexer
from codeguard.context.repository_indexer import RepositorySymbolIndexer
from codeguard.context.symbols import RepositorySymbolIndex
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
        """Build a symbol index for a specific GitHub commit."""

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

        # 4. Retrieve discovered files at the exact resolved commit.
        retrieved_files = await self._file_content_retriever.retrieve(
            owner=owner,
            repo=repo,
            commit_sha=commit.sha,
            discovery=discovery,
        )

        # 5. Build an AST symbol index for each retrieved file.
        file_indexes = tuple(
            self._file_indexer.index(
                retrieved_file.discovered_file,
                retrieved_file.source,
            )
            for retrieved_file in retrieved_files
        )

        # 6. Combine individual indexes into one repository index.
        return self._repository_indexer.index(
            commit_sha=commit.sha,
            discovery=discovery,
            file_indexes=file_indexes,
        )
