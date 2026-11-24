import pytest

from codeguard.context.repository_context_builder import RepositoryContextBuilder
from codeguard.context.source_snapshot import RepositoryContextBuildResult
from codeguard.context.symbols import (
    RepositoryFileFailure,
)
from codeguard.github.schemas import GitHubGitCommit, GitHubRepositoryTree

COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40


def make_tree_entry(
    path: str,
    sha_marker: str,
) -> dict[str, object]:
    return {
        "path": path,
        "mode": "100644",
        "type": "blob",
        "sha": sha_marker * 40,
        "size": 128,
    }


def make_repository_tree(
    paths: tuple[str, ...] = (),
    *,
    truncated: bool = False,
) -> GitHubRepositoryTree:
    entries = [
        make_tree_entry(path, chr(ord("c") + index)) for index, path in enumerate(paths)
    ]

    return GitHubRepositoryTree.model_validate(
        {
            "sha": TREE_SHA,
            "truncated": truncated,
            "tree": entries,
        }
    )


class FakeGitHubClient:
    """Fake GitHub client for repository context integration tests."""

    def __init__(
        self,
        tree: GitHubRepositoryTree,
        sources: dict[str, str],
    ) -> None:
        self.tree = tree
        self.sources = sources

        self.commit_requests: list[tuple[str, str, str]] = []
        self.tree_requests: list[tuple[str, str, str, bool]] = []
        self.file_requests: list[tuple[str, str, str, str]] = []

        self.commit_error: Exception | None = None
        self.tree_error: Exception | None = None
        self.file_error_path: str | None = None

        self.commit = GitHubGitCommit.model_validate(
            {
                "sha": COMMIT_SHA,
                "tree": {
                    "sha": TREE_SHA,
                    "url": (
                        "https://api.github.com/repos/"
                        "octocat/CodeGuard/git/trees/"
                        f"{TREE_SHA}"
                    ),
                },
            }
        )

    async def get_git_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> GitHubGitCommit:
        self.commit_requests.append((owner, repo, commit_sha))

        if self.commit_error is not None:
            raise self.commit_error

        return self.commit

    async def get_repository_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        *,
        recursive: bool = True,
    ) -> GitHubRepositoryTree:
        self.tree_requests.append((owner, repo, tree_sha, recursive))

        if self.tree_error is not None:
            raise self.tree_error

        return self.tree

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        self.file_requests.append((owner, repo, path, ref))

        if path == self.file_error_path:
            raise RuntimeError(f"Failed to retrieve {path}")

        return self.sources[path]


async def test_repository_context_builder_builds_complete_index() -> None:
    # Tree order is deliberately different from alphabetical file order.
    tree = make_repository_tree(
        (
            "src/payments.py",
            "src/orders.py",
            "README.md",
        )
    )

    client = FakeGitHubClient(
        tree,
        {
            "src/payments.py": ("def process_payment():\n    validate_payment()\n"),
            "src/orders.py": ("def create_order():\n    save_order()\n"),
        },
    )

    builder = RepositoryContextBuilder(
        github_client=client,
        max_concurrency=2,
    )

    result = await builder.build(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert result.commit_sha == COMMIT_SHA
    assert result.is_complete is True
    assert len(result.files) == 2

    assert [item.discovered_file.path for item in result.files] == [
        "src/orders.py",
        "src/payments.py",
    ]

    assert result.files[0].analysis.functions[0].qualified_name == ("create_order")

    assert result.files[1].analysis.functions[0].qualified_name == ("process_payment")

    assert result.files[0].analysis.calls[0].caller_qualified_name == ("create_order")

    assert client.commit_requests == [("octocat", "CodeGuard", COMMIT_SHA)]

    assert client.tree_requests == [("octocat", "CodeGuard", TREE_SHA, True)]

    assert {(path, ref) for _, _, path, ref in client.file_requests} == {
        ("src/orders.py", COMMIT_SHA),
        ("src/payments.py", COMMIT_SHA),
    }


async def test_repository_context_builder_handles_empty_repository() -> None:
    client = FakeGitHubClient(
        make_repository_tree(),
        {},
    )

    builder = RepositoryContextBuilder(client)

    result = await builder.build(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert result.commit_sha == COMMIT_SHA
    assert result.files == ()
    assert result.is_complete is True
    assert client.file_requests == []


async def test_repository_context_builder_preserves_truncated_tree() -> None:
    tree = make_repository_tree(
        ("src/payments.py",),
        truncated=True,
    )

    client = FakeGitHubClient(
        tree,
        {"src/payments.py": "def process_payment(): pass\n"},
    )

    builder = RepositoryContextBuilder(client)

    result = await builder.build(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert result.tree_truncated is True
    assert result.is_complete is False
    assert len(result.files) == 1


async def test_repository_context_builder_preserves_file_limit_metadata() -> None:
    tree = make_repository_tree(
        (
            "src/payments.py",
            "src/orders.py",
        )
    )

    client = FakeGitHubClient(
        tree,
        {"src/orders.py": "def create_order(): pass\n"},
    )

    builder = RepositoryContextBuilder(
        github_client=client,
        max_files=1,
    )

    result = await builder.build(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert result.file_limit_reached is True
    assert result.is_complete is False

    assert [item.discovered_file.path for item in result.files] == ["src/orders.py"]


async def test_repository_context_builder_propagates_commit_error() -> None:
    client = FakeGitHubClient(
        make_repository_tree(),
        {},
    )
    client.commit_error = RuntimeError("Commit retrieval failed")

    builder = RepositoryContextBuilder(client)

    with pytest.raises(RuntimeError, match="Commit retrieval failed"):
        await builder.build(
            owner="octocat",
            repo="CodeGuard",
            commit_sha=COMMIT_SHA,
        )

    assert client.tree_requests == []


async def test_repository_context_builder_propagates_tree_error() -> None:
    client = FakeGitHubClient(
        make_repository_tree(),
        {},
    )
    client.tree_error = RuntimeError("Tree retrieval failed")

    builder = RepositoryContextBuilder(client)

    with pytest.raises(RuntimeError, match="Tree retrieval failed"):
        await builder.build(
            owner="octocat",
            repo="CodeGuard",
            commit_sha=COMMIT_SHA,
        )

    assert client.file_requests == []


async def test_builder_keeps_success_when_file_retrieval_fails() -> None:
    tree = make_repository_tree(
        (
            "src/payments.py",
            "src/orders.py",
        )
    )

    client = FakeGitHubClient(
        tree,
        {
            "src/payments.py": "def process_payment(): pass\n",
            "src/orders.py": "def create_order(): pass\n",
        },
    )
    client.file_error_path = "src/orders.py"

    builder = RepositoryContextBuilder(client)

    result = await builder.build(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert len(result.files) == 1
    assert result.files[0].discovered_file.path == "src/payments.py"

    assert result.failures == (
        RepositoryFileFailure(
            file_path="src/orders.py",
            stage="retrieval",
            error_type="RuntimeError",
            message="Failed to retrieve src/orders.py",
        ),
    )

    assert result.is_complete is False


async def test_builder_keeps_valid_files_when_syntax_is_invalid() -> None:
    tree = make_repository_tree(
        (
            "src/payments.py",
            "src/orders.py",
        )
    )

    client = FakeGitHubClient(
        tree,
        {
            "src/payments.py": "def process_payment(): pass\n",
            "src/orders.py": "def broken(:\n    pass\n",
        },
    )

    builder = RepositoryContextBuilder(client)

    result = await builder.build(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert len(result.files) == 1
    assert result.files[0].discovered_file.path == "src/payments.py"

    assert len(result.failures) == 1

    failure = result.failures[0]

    assert failure.file_path == "src/orders.py"
    assert failure.stage == "analysis"
    assert failure.error_type == "SyntaxError"
    assert failure.message
    assert result.is_complete is False


async def test_build_with_source_preserves_source_and_symbol_index() -> None:
    source = "def authenticate():\n    return True\n"

    client = FakeGitHubClient(
        make_repository_tree(("src/auth.py",)),
        {"src/auth.py": source},
    )

    builder = RepositoryContextBuilder(client)

    result = await builder.build_with_source(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert isinstance(result, RepositoryContextBuildResult)
    assert result.symbol_index.commit_sha == COMMIT_SHA
    assert result.source_snapshot.commit_sha == COMMIT_SHA

    assert len(result.symbol_index.files) == 1
    assert result.source_snapshot.files[0].source == source
    assert result.source_snapshot.files[0].path == "src/auth.py"

    assert result.source_snapshot.total_source_bytes == len(source.encode("utf-8"))
    assert result.source_snapshot.is_complete is True


async def test_build_with_source_preserves_source_when_ast_fails() -> None:
    valid_source = "def process_payment(): pass\n"
    invalid_source = "def broken(:\n    pass\n"

    client = FakeGitHubClient(
        make_repository_tree(("src/payments.py", "src/orders.py")),
        {
            "src/payments.py": valid_source,
            "src/orders.py": invalid_source,
        },
    )

    builder = RepositoryContextBuilder(client)

    result = await builder.build_with_source(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    snapshot_by_path = {
        source_file.path: source_file for source_file in result.source_snapshot.files
    }

    assert snapshot_by_path["src/orders.py"].source == invalid_source
    assert snapshot_by_path["src/payments.py"].source == valid_source

    assert len(result.symbol_index.files) == 1
    assert len(result.symbol_index.failures) == 1
    assert result.symbol_index.failures[0].stage == "analysis"

    # Source coverage is complete even though AST analysis failed.
    assert result.source_snapshot.is_complete is True
    assert result.symbol_index.is_complete is False


async def test_build_with_source_tracks_retrieval_failure() -> None:
    client = FakeGitHubClient(
        make_repository_tree(("src/auth.py", "src/orders.py")),
        {
            "src/auth.py": "def authenticate(): pass\n",
            "src/orders.py": "def create_order(): pass\n",
        },
    )
    client.file_error_path = "src/orders.py"

    builder = RepositoryContextBuilder(client)

    result = await builder.build_with_source(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert [item.path for item in result.source_snapshot.files] == ["src/auth.py"]

    assert len(result.source_snapshot.failures) == 1
    assert result.source_snapshot.failures[0].file_path == "src/orders.py"
    assert result.source_snapshot.is_complete is False


async def test_build_with_source_enforces_aggregate_budget() -> None:
    large_source = "VALUE = 'large'\n"
    small_source = "x=1\n"

    budget = len(small_source.encode("utf-8"))

    client = FakeGitHubClient(
        make_repository_tree(("src/a.py", "src/b.py")),
        {
            "src/a.py": large_source,
            "src/b.py": small_source,
        },
    )

    builder = RepositoryContextBuilder(
        client,
        max_source_snapshot_bytes=budget,
    )

    result = await builder.build_with_source(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert [item.path for item in result.source_snapshot.files] == ["src/b.py"]

    assert result.source_snapshot.omitted_source_paths == ("src/a.py",)
    assert result.source_snapshot.total_source_bytes <= budget
    assert result.source_snapshot.is_complete is False


async def test_build_with_source_preserves_tree_and_file_limit_metadata() -> None:
    client = FakeGitHubClient(
        make_repository_tree(
            ("src/a.py", "src/b.py"),
            truncated=True,
        ),
        {
            "src/a.py": "x=1\n",
        },
    )

    builder = RepositoryContextBuilder(
        client,
        max_files=1,
    )

    result = await builder.build_with_source(
        owner="octocat",
        repo="CodeGuard",
        commit_sha=COMMIT_SHA,
    )

    assert result.source_snapshot.tree_truncated is True
    assert result.source_snapshot.file_limit_reached is True
    assert result.source_snapshot.is_complete is False
