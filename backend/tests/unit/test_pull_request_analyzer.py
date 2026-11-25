import pytest

from codeguard.ai.context_builder import AIReviewContextBuilder
from codeguard.ai.contracts import AIReviewInput, AIReviewResult
from codeguard.ai.engine import AIReviewEngine, AIReviewExecutionResult
from codeguard.ai.mock_provider import MockAIReviewProvider
from codeguard.analyzers.python_ast import (
    PythonAnalysisResult,
    PythonASTAnalyzer,
)
from codeguard.context.file_discovery import DiscoveredPythonFile
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
from codeguard.github.exceptions import GitHubNotFoundError
from codeguard.github.schemas import (
    GitHubPullRequest,
    GitHubPullRequestFile,
)
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer

PULL_REQUEST_RESPONSE = {
    "url": "https://api.github.com/repos/octocat/CodeGuard/pulls/42",
    "html_url": "https://github.com/octocat/CodeGuard/pull/42",
    "number": 42,
    "state": "open",
    "title": "Improve authentication",
    "body": "Update authentication logic.",
    "draft": False,
    "merged": False,
    "user": {
        "login": "octocat",
        "id": 100,
    },
    "base": {
        "ref": "main",
        "sha": "a" * 40,
    },
    "head": {
        "ref": "feature/authentication",
        "sha": "b" * 40,
    },
}


def make_pull_request() -> GitHubPullRequest:
    return GitHubPullRequest.model_validate(PULL_REQUEST_RESPONSE)


def make_file(
    filename: str,
    status: str = "modified",
    additions: int = 2,
    deletions: int = 1,
    previous_filename: str | None = None,
    patch: str | None = None,
) -> GitHubPullRequestFile:
    return GitHubPullRequestFile.model_validate(
        {
            "filename": filename,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
            "previous_filename": previous_filename,
            "patch": patch,
        }
    )


class FakeGitHubClient:
    """Deterministic fake for GitHub integration tests."""

    def __init__(
        self,
        files: list[GitHubPullRequestFile] | None = None,
        contents: dict[str, str] | None = None,
        content_errors: dict[str, Exception] | None = None,
        metadata_error: Exception | None = None,
    ) -> None:
        self.pull_request = make_pull_request()
        self.files = files or []
        self.contents = contents or {}
        self.content_errors = content_errors or {}
        self.metadata_error = metadata_error

        self.metadata_calls: list[tuple[str, str, int]] = []
        self.file_list_calls: list[tuple[str, str, int]] = []
        self.content_calls: list[tuple[str, str, str, str]] = []

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
        self.metadata_calls.append((owner, repo, pull_number))

        if self.metadata_error is not None:
            raise self.metadata_error

        return self.pull_request

    async def get_all_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[GitHubPullRequestFile]:
        self.file_list_calls.append((owner, repo, pull_number))
        return self.files

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        self.content_calls.append((owner, repo, path, ref))

        if path in self.content_errors:
            raise self.content_errors[path]

        if path not in self.contents:
            raise GitHubNotFoundError(
                "File not found.",
                status_code=404,
                request_url=f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            )

        return self.contents[path]


class FakeRepositoryContextBuilder:
    """Fake repository context builder for analyzer tests."""

    def __init__(self, result: RepositorySymbolIndex) -> None:
        self.result = result
        self.calls: list[tuple[str, str, str]] = []

    async def build(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> RepositorySymbolIndex:
        self.calls.append((owner, repo, commit_sha))
        return self.result


class FakeSnapshotAwareRepositoryContextBuilder:
    """Fake that exposes both the legacy and source-snapshot build methods."""

    def __init__(self, result: RepositoryContextBuildResult) -> None:
        self.result = result
        self.build_calls: list[tuple[str, str, str]] = []
        self.build_with_source_calls: list[tuple[str, str, str]] = []

    async def build(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> RepositorySymbolIndex:
        self.build_calls.append((owner, repo, commit_sha))
        return self.result.symbol_index

    async def build_with_source(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> RepositoryContextBuildResult:
        self.build_with_source_calls.append((owner, repo, commit_sha))
        return self.result


class RecordingAIReviewEngine(AIReviewEngine):
    """Capture the exact input and snapshot passed through the real engine."""

    def __init__(self, provider: MockAIReviewProvider) -> None:
        super().__init__(provider)
        self.review_calls: list[tuple[AIReviewInput, RepositorySourceSnapshot]] = []
        self.execution_results: list[AIReviewExecutionResult] = []

    async def review(
        self,
        review_input: AIReviewInput,
        source_snapshot: RepositorySourceSnapshot,
    ) -> AIReviewExecutionResult:
        self.review_calls.append((review_input, source_snapshot))
        result = await super().review(review_input, source_snapshot)
        self.execution_results.append(result)
        return result


def make_repository_context_build_result(
    sources: dict[str, str],
) -> RepositoryContextBuildResult:
    ast_analyzer = PythonASTAnalyzer()
    symbol_files: list[FileSymbolIndex] = []
    source_files: list[RepositorySourceFile] = []

    for path, source in sources.items():
        discovered_file = DiscoveredPythonFile(
            path=path,
            sha="c" * 40,
            size_bytes=len(source.encode("utf-8")),
        )
        symbol_files.append(
            FileSymbolIndex(
                discovered_file=discovered_file,
                analysis=ast_analyzer.analyze(source, file_path=path),
            )
        )
        source_files.append(
            RepositorySourceFile(
                discovered_file=discovered_file,
                source=source,
            )
        )

    commit_sha = "b" * 40
    symbol_index = RepositorySymbolIndex(
        commit_sha=commit_sha,
        files=tuple(symbol_files),
    )
    source_snapshot = RepositorySourceSnapshot(
        commit_sha=commit_sha,
        files=tuple(source_files),
    )

    return RepositoryContextBuildResult(
        symbol_index=symbol_index,
        source_snapshot=source_snapshot,
    )


def make_ai_integration_client() -> FakeGitHubClient:
    auth_source = "def authenticate(token):\n    return eval(token)\n"
    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return eval(token)\n"
    )

    return FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/auth.py": auth_source},
    )


def make_ai_integration_context() -> RepositoryContextBuildResult:
    return make_repository_context_build_result(
        {
            "src/auth.py": "def authenticate(token):\n    return eval(token)\n",
            "src/views.py": (
                "from src.auth import authenticate\n"
                "\n"
                "def login(token):\n"
                "    return authenticate(token)\n"
            ),
            "src/unrelated.py": (
                "UNRELATED_REPOSITORY_SENTINEL = 'must not enter AI context'\n"
            ),
        }
    )


@pytest.mark.asyncio
async def test_analyzer_builds_repository_context_once_at_pr_head_sha() -> None:
    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(),
    )

    context_builder = FakeRepositoryContextBuilder(context)

    client = FakeGitHubClient(
        files=[
            make_file("src/auth.py"),
            make_file("src/users.py"),
        ],
        contents={
            "src/auth.py": "def authenticate():\n    return True\n",
            "src/users.py": "def get_user():\n    return None\n",
        },
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=context_builder,
    )

    result = await analyzer.analyze(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    assert context_builder.calls == [
        ("octocat", "CodeGuard", "b" * 40),
    ]

    assert result.repository_context is context


@pytest.mark.asyncio
async def test_analyzes_changed_python_file() -> None:
    source = "def process_input(value):\n    return eval(value)\n"

    client = FakeGitHubClient(
        files=[
            make_file("backend/src/auth.py"),
        ],
        contents={
            "backend/src/auth.py": source,
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    assert result.owner == "octocat"
    assert result.repo == "CodeGuard"
    assert result.pull_number == 42
    assert result.title == "Improve authentication"
    assert result.head_sha == "b" * 40

    assert len(result.files) == 1

    file_result = result.files[0]

    assert file_result.file_path == "backend/src/auth.py"
    assert file_result.status == "modified"
    assert file_result.ast_result is not None
    assert isinstance(file_result.ast_result, PythonAnalysisResult)

    assert file_result.ast_result.functions[0].name == "process_input"

    assert len(file_result.risk_findings) == 1
    assert file_result.risk_findings[0].rule_id == "PY001"


@pytest.mark.asyncio
async def test_retrieves_source_using_pull_request_head_sha() -> None:
    client = FakeGitHubClient(
        files=[make_file("src/example.py")],
        contents={"src/example.py": "def hello():\n    return 'hello'\n"},
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    await analyzer.analyze("octocat", "CodeGuard", 42)

    assert client.content_calls == [
        (
            "octocat",
            "CodeGuard",
            "src/example.py",
            "b" * 40,
        )
    ]


@pytest.mark.asyncio
async def test_skips_non_python_files() -> None:
    client = FakeGitHubClient(
        files=[
            make_file("README.md"),
            make_file("docs/architecture.md"),
        ],
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert len(result.files) == 2

    assert all(file_result.ast_result is None for file_result in result.files)
    assert all(
        file_result.skip_reason == "unsupported_file_type"
        for file_result in result.files
    )

    assert client.content_calls == []


@pytest.mark.asyncio
async def test_skips_deleted_python_files() -> None:
    client = FakeGitHubClient(
        files=[
            make_file("src/deleted.py", status="removed"),
        ],
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert len(result.files) == 1

    file_result = result.files[0]

    assert file_result.status == "removed"
    assert file_result.skip_reason == "deleted_file"
    assert file_result.ast_result is None

    assert client.content_calls == []


@pytest.mark.asyncio
async def test_analyzes_renamed_python_files_using_new_path() -> None:
    client = FakeGitHubClient(
        files=[
            make_file(
                "src/new_auth.py",
                status="renamed",
                previous_filename="src/old_auth.py",
            )
        ],
        contents={
            "src/new_auth.py": "def authenticate():\n    return True\n",
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert len(result.files) == 1
    assert result.files[0].file_path == "src/new_auth.py"
    assert result.files[0].ast_result is not None

    assert client.content_calls[0][2] == "src/new_auth.py"


@pytest.mark.asyncio
async def test_invalid_python_syntax_does_not_stop_other_files() -> None:
    client = FakeGitHubClient(
        files=[
            make_file("src/broken.py"),
            make_file("src/working.py"),
        ],
        contents={
            "src/broken.py": "def broken(:\n",
            "src/working.py": "def working():\n    return True\n",
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert len(result.files) == 2

    broken, working = result.files

    assert broken.error is not None
    assert broken.error.startswith("SyntaxError:")

    assert working.error is None
    assert working.ast_result is not None
    assert working.ast_result.functions[0].name == "working"


@pytest.mark.asyncio
async def test_github_file_error_does_not_stop_other_files() -> None:
    client = FakeGitHubClient(
        files=[
            make_file("src/missing.py"),
            make_file("src/working.py"),
        ],
        contents={
            "src/working.py": "def working():\n    return True\n",
        },
        content_errors={
            "src/missing.py": GitHubNotFoundError(
                "File not found.",
                status_code=404,
            ),
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    missing, working = result.files

    assert missing.error is not None
    assert missing.error.startswith("GitHubNotFoundError:")

    assert working.error is None
    assert working.ast_result is not None


@pytest.mark.asyncio
async def test_empty_pull_request_returns_empty_file_results() -> None:
    client = FakeGitHubClient(files=[])

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert result.pull_number == 42
    assert result.title == "Improve authentication"
    assert result.files == ()


@pytest.mark.asyncio
async def test_multiple_python_files_are_analyzed() -> None:
    client = FakeGitHubClient(
        files=[
            make_file("src/auth.py"),
            make_file("src/users.py"),
            make_file("src/orders.py"),
        ],
        contents={
            "src/auth.py": "def authenticate():\n    return True\n",
            "src/users.py": "class User:\n    pass\n",
            "src/orders.py": "def create_order():\n    return None\n",
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert len(result.files) == 3

    assert all(file_result.ast_result is not None for file_result in result.files)

    assert all(file_result.skip_reason is None for file_result in result.files)

    assert all(file_result.error is None for file_result in result.files)

    assert len(client.content_calls) == 3


@pytest.mark.asyncio
async def test_pull_request_metadata_error_is_propagated() -> None:
    client = FakeGitHubClient(
        metadata_error=GitHubNotFoundError(
            "Pull request not found.",
            status_code=404,
        )
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    with pytest.raises(GitHubNotFoundError):
        await analyzer.analyze("octocat", "CodeGuard", 999)

    assert client.file_list_calls == []
    assert client.content_calls == []


@pytest.mark.asyncio
async def test_unexpected_programming_error_is_not_swallowed() -> None:
    client = FakeGitHubClient(
        files=[make_file("src/example.py")],
        content_errors={
            "src/example.py": RuntimeError("unexpected programming error"),
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    with pytest.raises(
        RuntimeError,
        match="unexpected programming error",
    ):
        await analyzer.analyze("octocat", "CodeGuard", 42)


@pytest.mark.asyncio
async def test_includes_findings_on_added_lines() -> None:
    source = "def process_input(value):\n    return eval(value)\n"

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def process_input(value):\n"
        "-    return value\n"
        "+    return eval(value)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            ),
        ],
        contents={
            "src/auth.py": source,
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.added_line_numbers == frozenset({2})
    assert len(file_result.risk_findings) == 1
    assert len(file_result.changed_line_findings) == 1
    assert file_result.changed_line_findings[0].lineno == 2
    assert file_result.line_mapping_error is None


@pytest.mark.asyncio
async def test_excludes_findings_on_unchanged_lines() -> None:
    source = "def run(command):\n    eval(command)\n    exec(command)\n"

    patch = (
        "@@ -1,2 +1,3 @@\n def run(command):\n     eval(command)\n+    exec(command)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=0,
                patch=patch,
            ),
        ],
        contents={
            "src/auth.py": source,
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert {finding.lineno for finding in file_result.risk_findings} == {
        2,
        3,
    }

    assert file_result.added_line_numbers == frozenset({3})

    assert tuple(finding.lineno for finding in file_result.changed_line_findings) == (
        3,
    )


@pytest.mark.asyncio
async def test_missing_patch_marks_line_mapping_unavailable() -> None:
    client = FakeGitHubClient(
        files=[
            make_file("src/auth.py"),
        ],
        contents={
            "src/auth.py": ("def process_input(value):\n    return eval(value)\n"),
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.ast_result is not None
    assert len(file_result.risk_findings) == 1
    assert file_result.added_line_numbers is None
    assert file_result.changed_line_findings is None
    assert file_result.line_mapping_error == "patch_unavailable"


@pytest.mark.asyncio
async def test_empty_added_python_file_has_no_changed_lines() -> None:
    client = FakeGitHubClient(
        files=[
            make_file(
                "src/empty.py",
                status="added",
                additions=0,
                deletions=0,
                patch=None,
            ),
        ],
        contents={
            "src/empty.py": "",
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.ast_result is not None
    assert file_result.added_line_numbers == frozenset()
    assert file_result.changed_line_findings == ()
    assert file_result.line_mapping_error is None


@pytest.mark.asyncio
async def test_zero_additions_with_missing_patch_has_no_changed_lines() -> None:
    source = 'def safe():\n    eval("2 + 2")\n'

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/safe.py",
                status="modified",
                additions=0,
                deletions=1,
                patch=None,
            ),
        ],
        contents={
            "src/safe.py": source,
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.ast_result is not None
    assert len(file_result.risk_findings) == 1
    assert file_result.added_line_numbers == frozenset()
    assert file_result.changed_line_findings == ()
    assert file_result.line_mapping_error is None


@pytest.mark.asyncio
async def test_deletion_only_patch_has_no_added_lines() -> None:
    source = 'def safe():\n    eval("2 + 2")\n'

    patch = '@@ -1,3 +1,2 @@\n def safe():\n     eval("2 + 2")\n-    unused_call()\n'

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/safe.py",
                additions=0,
                deletions=1,
                patch=patch,
            ),
        ],
        contents={
            "src/safe.py": source,
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.added_line_numbers == frozenset()
    assert len(file_result.risk_findings) == 1
    assert file_result.changed_line_findings == ()
    assert file_result.line_mapping_error is None


@pytest.mark.asyncio
async def test_malformed_patch_does_not_stop_file_analysis() -> None:
    source = "def process_input(value):\n    return eval(value)\n"

    malformed_patch = "@@ -invalid,2 +1,3 @@\n+    return eval(value)\n"

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                patch=malformed_patch,
            ),
        ],
        contents={
            "src/auth.py": source,
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)

    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.ast_result is not None
    assert len(file_result.risk_findings) == 1
    assert file_result.added_line_numbers is None
    assert file_result.changed_line_findings is None
    assert file_result.line_mapping_error is not None
    assert file_result.line_mapping_error.startswith("ValueError:")


@pytest.mark.asyncio
async def test_patch_without_hunks_marks_mapping_unavailable() -> None:
    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=0,
                deletions=0,
                patch="Binary files differ",
            ),
        ],
        contents={
            "src/auth.py": ("def authenticate(value):\n    return eval(value)\n"),
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)
    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.ast_result is not None
    assert len(file_result.risk_findings) == 1
    assert file_result.added_line_numbers is None
    assert file_result.changed_line_findings is None
    assert file_result.line_mapping_error == "patch_has_no_hunks"


@pytest.mark.asyncio
async def test_incomplete_patch_marks_mapping_unavailable() -> None:
    patch = "@@ -1,1 +1,2 @@\n def authenticate(value):\n+    return eval(value)\n"

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=2,
                deletions=0,
                patch=patch,
            ),
        ],
        contents={
            "src/auth.py": ("def authenticate(value):\n    return eval(value)\n"),
        },
    )

    analyzer = PullRequestAnalyzer(github_client=client)
    result = await analyzer.analyze("octocat", "CodeGuard", 42)
    file_result = result.files[0]

    assert file_result.ast_result is not None
    assert len(file_result.risk_findings) == 1
    assert file_result.added_line_numbers is None
    assert file_result.changed_line_findings is None
    assert file_result.line_mapping_error == "patch_addition_count_mismatch"


@pytest.mark.asyncio
async def test_analyzer_identifies_changed_functions_and_classes() -> None:
    source = (
        "def authenticate(token):\n"
        "    return str(token)\n"
        "\n"
        "class AuthService:\n"
        "    def verify(self, token):\n"
        "        return authenticate(token)\n"
    )

    patch = (
        "@@ -1,6 +1,6 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
        " \n"
        "-class LegacyAuthService:\n"
        "+class AuthService:\n"
        "     def verify(self, token):\n"
        "         return authenticate(token)\n"
    )

    discovered_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="c" * 40,
        size_bytes=len(source.encode("utf-8")),
    )

    file_index = FileSymbolIndex(
        discovered_file=discovered_file,
        analysis=PythonASTAnalyzer().analyze(
            source,
            file_path="src/auth.py",
        ),
    )

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(file_index,),
    )

    context_builder = FakeRepositoryContextBuilder(context)

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=2,
                deletions=2,
                patch=patch,
            )
        ],
        contents={"src/auth.py": source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=context_builder,
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    file_result = result.files[0]

    assert tuple(
        function.qualified_name for function in file_result.changed_functions
    ) == ("authenticate",)

    assert tuple(
        class_info.qualified_name for class_info in file_result.changed_classes
    ) == ("AuthService",)


@pytest.mark.asyncio
async def test_analyzer_discovers_callers_in_other_repository_files() -> None:
    auth_source = "def authenticate(token):\n    return str(token)\n"

    views_source = (
        "from src.auth import authenticate\n"
        "\n"
        "def login(token):\n"
        "    return authenticate(token)\n"
    )

    auth_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="c" * 40,
        size_bytes=len(auth_source.encode("utf-8")),
    )

    views_file = DiscoveredPythonFile(
        path="src/views.py",
        sha="d" * 40,
        size_bytes=len(views_source.encode("utf-8")),
    )

    ast_analyzer = PythonASTAnalyzer()

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(
            FileSymbolIndex(
                discovered_file=auth_file,
                analysis=ast_analyzer.analyze(
                    auth_source,
                    file_path="src/auth.py",
                ),
            ),
            FileSymbolIndex(
                discovered_file=views_file,
                analysis=ast_analyzer.analyze(
                    views_source,
                    file_path="src/views.py",
                ),
            ),
        ),
    )

    context_builder = FakeRepositoryContextBuilder(context)

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/auth.py": auth_source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=context_builder,
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    callers = result.files[0].related_callers

    assert callers is not None

    assert [
        (
            caller.file_path,
            caller.caller_qualified_name,
            caller.target_function_qualified_name,
            caller.lineno,
        )
        for caller in callers
    ] == [
        ("src/views.py", "login", "authenticate", 4),
    ]


@pytest.mark.asyncio
async def test_analyzer_discovers_aliased_import_callers() -> None:
    auth_source = "def authenticate(token):\n    return str(token)\n"

    views_source = (
        "from src.auth import authenticate as verify_token\n"
        "\n"
        "def login(token):\n"
        "    return verify_token(token)\n"
    )

    ast_analyzer = PythonASTAnalyzer()

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(
            FileSymbolIndex(
                discovered_file=DiscoveredPythonFile(
                    path="src/auth.py",
                    sha="c" * 40,
                    size_bytes=len(auth_source.encode("utf-8")),
                ),
                analysis=ast_analyzer.analyze(
                    auth_source,
                    file_path="src/auth.py",
                ),
            ),
            FileSymbolIndex(
                discovered_file=DiscoveredPythonFile(
                    path="src/views.py",
                    sha="d" * 40,
                    size_bytes=len(views_source.encode("utf-8")),
                ),
                analysis=ast_analyzer.analyze(
                    views_source,
                    file_path="src/views.py",
                ),
            ),
        ),
    )

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/auth.py": auth_source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=FakeRepositoryContextBuilder(context),
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert result.files[0].related_callers is not None

    assert [
        (
            caller.file_path,
            caller.caller_qualified_name,
            caller.target_function_qualified_name,
            caller.lineno,
        )
        for caller in result.files[0].related_callers
    ] == [
        ("src/views.py", "login", "authenticate", 4),
    ]


@pytest.mark.parametrize(
    "related_source",
    [
        (
            "def authenticate(token):\n"
            "    return token\n"
            "\n"
            "def login(token):\n"
            "    return authenticate(token)\n"
        ),
        ("from src.auth import authenticate\n\nauthenticate('startup-token')\n"),
    ],
    ids=[
        "unrelated-local-function",
        "module-level-call",
    ],
)
@pytest.mark.asyncio
async def test_analyzer_ignores_unrelated_and_module_level_calls(
    related_source: str,
) -> None:
    auth_source = "def authenticate(token):\n    return str(token)\n"

    ast_analyzer = PythonASTAnalyzer()

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(
            FileSymbolIndex(
                discovered_file=DiscoveredPythonFile(
                    path="src/auth.py",
                    sha="c" * 40,
                    size_bytes=len(auth_source.encode("utf-8")),
                ),
                analysis=ast_analyzer.analyze(
                    auth_source,
                    file_path="src/auth.py",
                ),
            ),
            FileSymbolIndex(
                discovered_file=DiscoveredPythonFile(
                    path="src/related.py",
                    sha="d" * 40,
                    size_bytes=len(related_source.encode("utf-8")),
                ),
                analysis=ast_analyzer.analyze(
                    related_source,
                    file_path="src/related.py",
                ),
            ),
        ),
    )

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/auth.py": auth_source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=FakeRepositoryContextBuilder(context),
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert result.files[0].related_callers == ()


@pytest.mark.asyncio
async def test_analyzer_preserves_partial_repository_context_metadata() -> None:
    source = "def authenticate(token):\n    return str(token)\n"

    discovered_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="c" * 40,
        size_bytes=len(source.encode("utf-8")),
    )

    file_index = FileSymbolIndex(
        discovered_file=discovered_file,
        analysis=PythonASTAnalyzer().analyze(
            source,
            file_path="src/auth.py",
        ),
    )

    failure = RepositoryFileFailure(
        file_path="src/missing.py",
        stage="retrieval",
        error_type="GitHubNotFoundError",
        message="Repository file could not be retrieved.",
    )

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(file_index,),
        tree_truncated=True,
        file_limit_reached=True,
        failures=(failure,),
    )

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/auth.py": source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=FakeRepositoryContextBuilder(context),
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert result.repository_context is context
    assert result.repository_context.is_complete is False
    assert result.repository_context.tree_truncated is True
    assert result.repository_context.file_limit_reached is True
    assert result.repository_context.failures == (failure,)


@pytest.mark.asyncio
async def test_analyzer_marks_symbols_unavailable_when_file_index_failed() -> None:
    source = "def authenticate(token):\n    return str(token)\n"

    failure = RepositoryFileFailure(
        file_path="src/auth.py",
        stage="retrieval",
        error_type="GitHubNotFoundError",
        message="Repository context retrieval failed.",
    )

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(),
        failures=(failure,),
    )

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/auth.py": source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=FakeRepositoryContextBuilder(context),
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    file_result = result.files[0]

    assert result.repository_context is not None
    assert result.repository_context.is_complete is False
    assert result.repository_context.failures == (failure,)

    assert file_result.changed_functions is None
    assert file_result.changed_classes is None
    assert file_result.related_callers is None


@pytest.mark.asyncio
async def test_analyzer_resolves_imports_from_src_layout_packages() -> None:
    auth_source = "def authenticate(token):\n    return str(token)\n"

    views_source = (
        "from codeguard.auth import authenticate\n"
        "\n"
        "def login(token):\n"
        "    return authenticate(token)\n"
    )

    ast_analyzer = PythonASTAnalyzer()

    context = RepositorySymbolIndex(
        commit_sha="b" * 40,
        files=(
            FileSymbolIndex(
                discovered_file=DiscoveredPythonFile(
                    path="src/codeguard/auth.py",
                    sha="c" * 40,
                    size_bytes=len(auth_source.encode("utf-8")),
                ),
                analysis=ast_analyzer.analyze(
                    auth_source,
                    file_path="src/codeguard/auth.py",
                ),
            ),
            FileSymbolIndex(
                discovered_file=DiscoveredPythonFile(
                    path="src/codeguard/views.py",
                    sha="d" * 40,
                    size_bytes=len(views_source.encode("utf-8")),
                ),
                analysis=ast_analyzer.analyze(
                    views_source,
                    file_path="src/codeguard/views.py",
                ),
            ),
        ),
    )

    patch = (
        "@@ -1,2 +1,2 @@\n"
        " def authenticate(token):\n"
        "-    return token\n"
        "+    return str(token)\n"
    )

    client = FakeGitHubClient(
        files=[
            make_file(
                "src/codeguard/auth.py",
                additions=1,
                deletions=1,
                patch=patch,
            )
        ],
        contents={"src/codeguard/auth.py": auth_source},
    )

    analyzer = PullRequestAnalyzer(
        github_client=client,
        repository_context_builder=FakeRepositoryContextBuilder(context),
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert result.files[0].related_callers is not None

    assert [
        (
            caller.file_path,
            caller.caller_qualified_name,
            caller.target_function_qualified_name,
            caller.lineno,
        )
        for caller in result.files[0].related_callers
    ] == [
        ("src/codeguard/views.py", "login", "authenticate", 4),
    ]


@pytest.mark.asyncio
async def test_analyzer_keeps_static_behavior_when_ai_is_disabled() -> None:
    repository_context = make_ai_integration_context()
    context_builder = FakeSnapshotAwareRepositoryContextBuilder(repository_context)
    analyzer = PullRequestAnalyzer(
        github_client=make_ai_integration_client(),
        repository_context_builder=context_builder,
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert context_builder.build_calls == [
        ("octocat", "CodeGuard", "b" * 40),
    ]
    assert context_builder.build_with_source_calls == []
    assert result.repository_context is repository_context.symbol_index
    assert len(result.files) == 1
    assert result.files[0].risk_findings


@pytest.mark.asyncio
async def test_analyzer_runs_ai_and_preserves_static_results() -> None:
    repository_context = make_ai_integration_context()

    baseline_context_builder = FakeSnapshotAwareRepositoryContextBuilder(
        repository_context
    )
    baseline = await PullRequestAnalyzer(
        github_client=make_ai_integration_client(),
        repository_context_builder=baseline_context_builder,
    ).analyze("octocat", "CodeGuard", 42)

    context_builder = FakeSnapshotAwareRepositoryContextBuilder(repository_context)
    provider_result = AIReviewResult(
        summary="Mock integration review completed.",
    )
    provider = MockAIReviewProvider(result=provider_result)
    engine = RecordingAIReviewEngine(provider)
    result = await PullRequestAnalyzer(
        github_client=make_ai_integration_client(),
        repository_context_builder=context_builder,
        ai_review_engine=engine,
        ai_review_context_builder=AIReviewContextBuilder(),
    ).analyze("octocat", "CodeGuard", 42)

    assert context_builder.build_calls == []
    assert context_builder.build_with_source_calls == [
        ("octocat", "CodeGuard", "b" * 40),
    ]
    assert result.repository_context is repository_context.symbol_index
    assert result.files == baseline.files

    assert len(engine.review_calls) == 1
    review_input, received_snapshot = engine.review_calls[0]
    assert isinstance(review_input, AIReviewInput)
    assert review_input.owner == "octocat"
    assert review_input.repo == "CodeGuard"
    assert review_input.pull_number == 42
    assert review_input.head_sha == "b" * 40
    assert len(review_input.context) <= 50_000
    assert "def authenticate(token):\n    return eval(token)\n" in review_input.context
    assert "def login(token):\n    return authenticate(token)" in review_input.context
    assert "UNRELATED_REPOSITORY_SENTINEL" not in review_input.context
    assert received_snapshot is repository_context.source_snapshot
    assert provider.reviewed_inputs == (review_input,)

    assert len(engine.execution_results) == 1
    assert result.ai_review_result is engine.execution_results[0]
    assert isinstance(result.ai_review_result, AIReviewExecutionResult)
    assert result.ai_review_result.provider_result == provider_result


@pytest.mark.asyncio
async def test_analyzer_skips_ai_when_no_relevant_source_is_available() -> None:
    repository_context = make_repository_context_build_result({})
    context_builder = FakeSnapshotAwareRepositoryContextBuilder(repository_context)
    provider = MockAIReviewProvider()
    engine = RecordingAIReviewEngine(provider)
    analyzer = PullRequestAnalyzer(
        github_client=make_ai_integration_client(),
        repository_context_builder=context_builder,
        ai_review_engine=engine,
        ai_review_context_builder=AIReviewContextBuilder(),
    )

    result = await analyzer.analyze("octocat", "CodeGuard", 42)

    assert context_builder.build_with_source_calls == [
        ("octocat", "CodeGuard", "b" * 40),
    ]
    assert provider.reviewed_inputs == ()
    assert engine.review_calls == []
    assert result.ai_review_result is None


@pytest.mark.asyncio
async def test_analyzer_propagates_ai_provider_exceptions() -> None:
    repository_context = make_ai_integration_context()
    provider_error = RuntimeError("Mock AI provider failed.")
    provider = MockAIReviewProvider(error=provider_error)
    analyzer = PullRequestAnalyzer(
        github_client=make_ai_integration_client(),
        repository_context_builder=FakeSnapshotAwareRepositoryContextBuilder(
            repository_context
        ),
        ai_review_engine=RecordingAIReviewEngine(provider),
        ai_review_context_builder=AIReviewContextBuilder(),
    )

    with pytest.raises(RuntimeError, match="Mock AI provider failed"):
        await analyzer.analyze("octocat", "CodeGuard", 42)

    assert len(provider.reviewed_inputs) == 1
