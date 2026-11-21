import pytest

from codeguard.analyzers.python_ast import PythonAnalysisResult
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
