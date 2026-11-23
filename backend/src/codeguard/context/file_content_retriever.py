import asyncio
from dataclasses import dataclass
from typing import Protocol

from codeguard.context.file_discovery import (
    DiscoveredPythonFile,
    RepositoryFileDiscoveryResult,
)
from codeguard.context.symbols import RepositoryFileFailure


class FileContentClient(Protocol):
    """Interface required for retrieving GitHub file contents."""

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        """Retrieve the contents of a file at a specific Git reference."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievedPythonFile:
    """Discovered Python file together with its retrieved source."""

    discovered_file: DiscoveredPythonFile
    source: str


@dataclass(frozen=True, slots=True)
class FileContentRetrievalResult:
    """Results of retrieving discovered repository files."""

    retrieved_files: tuple[RetrievedPythonFile, ...]
    failures: tuple[RepositoryFileFailure, ...]

    @property
    def is_complete(self) -> bool:
        """Indicate whether every requested file was retrieved."""

        return not self.failures


class RepositoryFileContentRetriever:
    """Retrieve discovered Python files with bounded concurrency."""

    def __init__(
        self,
        client: FileContentClient,
        max_concurrency: int = 5,
    ) -> None:
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer.")

        self._client = client
        self._max_concurrency = max_concurrency

    async def retrieve(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        discovery: RepositoryFileDiscoveryResult,
    ) -> tuple[RetrievedPythonFile, ...]:
        """Retrieve all discovered files at the specified commit."""

        if not discovery.files:
            return ()

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def retrieve_file(
            discovered_file: DiscoveredPythonFile,
        ) -> RetrievedPythonFile:
            async with semaphore:
                source = await self._client.get_file_content(
                    owner=owner,
                    repo=repo,
                    path=discovered_file.path,
                    ref=commit_sha,
                )

            return RetrievedPythonFile(
                discovered_file=discovered_file,
                source=source,
            )

        tasks = [
            asyncio.create_task(retrieve_file(discovered_file))
            for discovered_file in discovery.files
        ]

        try:
            # asyncio.gather preserves the order of its input awaitables.
            results = await asyncio.gather(*tasks)
            return tuple(results)

        except BaseException:
            # Cancel remaining requests and collect their results/exceptions.
            # This also cleans up child tasks if the caller is cancelled.
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def retrieve_partial(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        discovery: RepositoryFileDiscoveryResult,
    ) -> FileContentRetrievalResult:
        """Retrieve files while preserving individual retrieval failures."""

        if not discovery.files:
            return FileContentRetrievalResult(
                retrieved_files=(),
                failures=(),
            )

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def retrieve_file(
            discovered_file: DiscoveredPythonFile,
        ) -> RetrievedPythonFile:
            async with semaphore:
                source = await self._client.get_file_content(
                    owner=owner,
                    repo=repo,
                    path=discovered_file.path,
                    ref=commit_sha,
                )

            return RetrievedPythonFile(
                discovered_file=discovered_file,
                source=source,
            )

        tasks = [
            asyncio.create_task(retrieve_file(discovered_file))
            for discovered_file in discovery.files
        ]

        try:
            # return_exceptions=True allows other files to finish
            # even when an individual retrieval fails.
            outcomes = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )
        except BaseException:
            # Caller cancellation must still cancel outstanding requests.
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        retrieved_files: list[RetrievedPythonFile] = []
        failures: list[RepositoryFileFailure] = []

        for discovered_file, outcome in zip(
            discovery.files,
            outcomes,
            strict=True,
        ):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome

            if isinstance(outcome, Exception):
                failures.append(
                    RepositoryFileFailure(
                        file_path=discovered_file.path,
                        stage="retrieval",
                        error_type=type(outcome).__name__,
                        message=str(outcome),
                    )
                )
                continue

            if isinstance(outcome, BaseException):
                raise outcome

            retrieved_files.append(outcome)

        return FileContentRetrievalResult(
            retrieved_files=tuple(retrieved_files),
            failures=tuple(failures),
        )
