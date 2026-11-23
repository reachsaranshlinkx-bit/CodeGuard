import asyncio

import pytest

from codeguard.context.file_content_retriever import (
    FileContentRetrievalResult,
    RepositoryFileContentRetriever,
    RetrievedPythonFile,
)
from codeguard.context.file_discovery import (
    DiscoveredPythonFile,
    RepositoryFileDiscoveryResult,
)
from codeguard.context.symbols import RepositoryFileFailure

COMMIT_SHA = "a" * 40


class FakeGitHubClient:
    """Fake GitHub client for testing bounded file retrieval."""

    def __init__(
        self,
        sources: dict[str, str],
        *,
        delays: dict[str, float] | None = None,
        fail_path: str | None = None,
    ) -> None:
        self.sources = sources
        self.delays = delays or {}
        self.fail_path = fail_path

        self.calls: list[tuple[str, str, str, str]] = []
        self.active_requests = 0
        self.max_active_requests = 0

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        self.calls.append((owner, repo, path, ref))

        self.active_requests += 1
        self.max_active_requests = max(
            self.max_active_requests,
            self.active_requests,
        )

        try:
            await asyncio.sleep(self.delays.get(path, 0.01))

            if path == self.fail_path:
                raise RuntimeError(f"Failed to retrieve {path}")

            return self.sources[path]
        finally:
            self.active_requests -= 1


def make_discovered_file(path: str) -> DiscoveredPythonFile:
    return DiscoveredPythonFile(
        path=path,
        sha="b" * 40,
        size_bytes=128,
    )


def make_discovery(
    files: tuple[DiscoveredPythonFile, ...],
) -> RepositoryFileDiscoveryResult:
    return RepositoryFileDiscoveryResult(
        tree_sha="c" * 40,
        files=files,
        tree_truncated=False,
        file_limit_reached=False,
        skipped_invalid_path_count=0,
        skipped_oversized_file_count=0,
        skipped_unknown_size_count=0,
    )


async def test_retriever_preserves_discovery_order() -> None:
    files = (
        make_discovered_file("src/payments.py"),
        make_discovered_file("src/orders.py"),
        make_discovered_file("src/users.py"),
    )

    discovery = make_discovery(files)

    client = FakeGitHubClient(
        {
            "src/payments.py": "def process_payment(): pass",
            "src/orders.py": "def create_order(): pass",
            "src/users.py": "def create_user(): pass",
        },
        delays={
            "src/payments.py": 0.03,
            "src/orders.py": 0.01,
            "src/users.py": 0.02,
        },
    )

    retriever = RepositoryFileContentRetriever(
        client,
        max_concurrency=2,
    )

    result = await retriever.retrieve(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=discovery,
    )

    assert [item.discovered_file.path for item in result] == [
        "src/payments.py",
        "src/orders.py",
        "src/users.py",
    ]

    assert [item.source for item in result] == [
        "def process_payment(): pass",
        "def create_order(): pass",
        "def create_user(): pass",
    ]

    assert all(isinstance(item, RetrievedPythonFile) for item in result)


async def test_retriever_respects_maximum_concurrency() -> None:
    files = tuple(make_discovered_file(f"src/module_{index}.py") for index in range(6))

    sources = {item.path: f"# Source for {item.path}" for item in files}

    client = FakeGitHubClient(sources)

    retriever = RepositoryFileContentRetriever(
        client,
        max_concurrency=2,
    )

    result = await retriever.retrieve(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=make_discovery(files),
    )

    assert len(result) == 6
    assert client.max_active_requests == 2


async def test_retriever_uses_exact_commit_sha() -> None:
    discovered_file = make_discovered_file("src/payments.py")

    client = FakeGitHubClient({"src/payments.py": "print('hello')"})

    retriever = RepositoryFileContentRetriever(client)

    await retriever.retrieve(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=make_discovery((discovered_file,)),
    )

    assert client.calls == [("octocat", "CodeGuard", "src/payments.py", COMMIT_SHA)]


async def test_retriever_returns_empty_tuple_for_empty_discovery() -> None:
    client = FakeGitHubClient({})

    retriever = RepositoryFileContentRetriever(client)

    result = await retriever.retrieve(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=make_discovery(()),
    )

    assert result == ()
    assert client.calls == []


async def test_retriever_propagates_file_retrieval_failure() -> None:
    files = (
        make_discovered_file("src/payments.py"),
        make_discovered_file("src/orders.py"),
    )

    client = FakeGitHubClient(
        {
            "src/payments.py": "print('payment')",
            "src/orders.py": "print('order')",
        },
        fail_path="src/orders.py",
    )

    retriever = RepositoryFileContentRetriever(client)

    with pytest.raises(RuntimeError, match="src/orders.py"):
        await retriever.retrieve(
            owner="octocat",
            repo="CodeGuard",
            commit_sha=COMMIT_SHA,
            discovery=make_discovery(files),
        )


@pytest.mark.parametrize(
    "max_concurrency",
    [0, -1, True, 1.5],
)
def test_retriever_rejects_invalid_concurrency(
    max_concurrency: object,
) -> None:
    client = FakeGitHubClient({})

    with pytest.raises(ValueError):
        RepositoryFileContentRetriever(
            client,
            max_concurrency=max_concurrency,
        )


async def test_retriever_cancels_remaining_requests_after_failure() -> None:
    files = (
        make_discovered_file("src/payments.py"),
        make_discovered_file("src/orders.py"),
        make_discovered_file("src/users.py"),
    )

    client = FakeGitHubClient(
        {
            "src/payments.py": "print('payment')",
            "src/orders.py": "print('order')",
            "src/users.py": "print('user')",
        },
        delays={
            "src/payments.py": 30,
            "src/orders.py": 0.01,
        },
        fail_path="src/orders.py",
    )

    retriever = RepositoryFileContentRetriever(
        client,
        max_concurrency=2,
    )

    with pytest.raises(RuntimeError, match="src/orders.py"):
        await retriever.retrieve(
            owner="octocat",
            repo="CodeGuard",
            commit_sha=COMMIT_SHA,
            discovery=make_discovery(files),
        )

    # No in-flight retrieval should remain after failure.
    assert client.active_requests == 0


async def test_retriever_partial_preserves_successes_and_records_failures() -> None:
    files = (
        make_discovered_file("src/payments.py"),
        make_discovered_file("src/orders.py"),
        make_discovered_file("src/users.py"),
    )

    client = FakeGitHubClient(
        {
            "src/payments.py": "def process_payment(): pass",
            "src/orders.py": "def create_order(): pass",
            "src/users.py": "def create_user(): pass",
        },
        fail_path="src/orders.py",
    )

    retriever = RepositoryFileContentRetriever(client)

    result = await retriever.retrieve_partial(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=make_discovery(files),
    )

    assert isinstance(result, FileContentRetrievalResult)

    assert [item.discovered_file.path for item in result.retrieved_files] == [
        "src/payments.py",
        "src/users.py",
    ]

    assert result.failures == (
        RepositoryFileFailure(
            file_path="src/orders.py",
            stage="retrieval",
            error_type="RuntimeError",
            message="Failed to retrieve src/orders.py",
        ),
    )

    assert result.is_complete is False


async def test_retriever_partial_handles_all_files_failing() -> None:
    files = (
        make_discovered_file("src/payments.py"),
        make_discovered_file("src/orders.py"),
    )

    client = FakeGitHubClient({})

    retriever = RepositoryFileContentRetriever(client)

    result = await retriever.retrieve_partial(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=make_discovery(files),
    )

    assert result.retrieved_files == ()
    assert len(result.failures) == 2
    assert all(failure.stage == "retrieval" for failure in result.failures)
    assert result.is_complete is False


async def test_retriever_partial_handles_empty_discovery() -> None:
    client = FakeGitHubClient({})

    retriever = RepositoryFileContentRetriever(client)

    result = await retriever.retrieve_partial(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
        discovery=make_discovery(()),
    )

    assert result.retrieved_files == ()
    assert result.failures == ()
    assert result.is_complete is True
    assert client.calls == []


async def test_retriever_partial_propagates_cancellation() -> None:
    files = (
        make_discovered_file("src/payments.py"),
        make_discovered_file("src/orders.py"),
    )

    client = FakeGitHubClient(
        {
            "src/payments.py": "print('payment')",
            "src/orders.py": "print('order')",
        },
        delays={
            "src/payments.py": 30,
            "src/orders.py": 30,
        },
    )

    retriever = RepositoryFileContentRetriever(
        client,
        max_concurrency=2,
    )

    retrieval_task = asyncio.create_task(
        retriever.retrieve_partial(
            owner="octocat",
            repo="CodeGuard",
            commit_sha=COMMIT_SHA,
            discovery=make_discovery(files),
        )
    )

    await asyncio.sleep(0.01)
    retrieval_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await retrieval_task

    assert client.active_requests == 0
