from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from codeguard.analyzers.python_ast import (
    PythonAnalysisResult,
    PythonASTAnalyzer,
)
from codeguard.analyzers.python_risk import (
    PythonRiskAnalyzer,
    RiskFinding,
)
from codeguard.github.exceptions import GitHubError
from codeguard.github.schemas import (
    GitHubPullRequest,
    GitHubPullRequestFile,
)
from codeguard.review.changed_lines import parse_added_line_numbers
from codeguard.review.finding_mapper import (
    filter_findings_to_added_lines,
)


class GitHubClientProtocol(Protocol):
    """Interface required by the pull request analysis service."""

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest: ...

    async def get_all_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[GitHubPullRequestFile]: ...

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class FileAnalysisResult:
    """Analysis result for one changed pull request file."""

    file_path: str
    status: str
    additions: int
    deletions: int
    changes: int
    previous_filename: str | None
    ast_result: PythonAnalysisResult | None
    risk_findings: tuple[RiskFinding, ...]

    # Changed-line analysis
    added_line_numbers: frozenset[int] | None = None
    changed_line_findings: tuple[RiskFinding, ...] | None = None
    line_mapping_error: str | None = None

    skip_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PullRequestAnalysisResult:
    """Complete analysis result for a GitHub pull request."""

    owner: str
    repo: str
    pull_number: int
    title: str
    state: str
    head_sha: str
    base_sha: str
    html_url: str
    files: tuple[FileAnalysisResult, ...]


class PullRequestAnalyzer:
    """Orchestrate GitHub retrieval and Python static analysis."""

    def __init__(
        self,
        github_client: GitHubClientProtocol,
        ast_analyzer: PythonASTAnalyzer | None = None,
        risk_analyzer: PythonRiskAnalyzer | None = None,
    ) -> None:
        self._github_client = github_client
        self._ast_analyzer = ast_analyzer or PythonASTAnalyzer()
        self._risk_analyzer = risk_analyzer or PythonRiskAnalyzer()

    async def analyze(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> PullRequestAnalysisResult:
        """Retrieve a PR and analyze its eligible changed Python files."""

        if not owner or not owner.strip():
            raise ValueError("Repository owner must not be empty.")

        if not repo or not repo.strip():
            raise ValueError("Repository name must not be empty.")

        if (
            isinstance(pull_number, bool)
            or not isinstance(pull_number, int)
            or pull_number < 1
        ):
            raise ValueError("Pull request number must be a positive integer.")

        owner = owner.strip()
        repo = repo.strip()

        # Retrieve authoritative PR metadata first.
        pull_request = await self._github_client.get_pull_request(
            owner,
            repo,
            pull_number,
        )

        # Retrieve every changed file using the existing pagination client.
        changed_files = await self._github_client.get_all_pull_request_files(
            owner,
            repo,
            pull_number,
        )

        file_results: list[FileAnalysisResult] = []

        for changed_file in changed_files:
            file_result = await self._analyze_file(
                owner=owner,
                repo=repo,
                head_sha=pull_request.head.sha,
                changed_file=changed_file,
            )

            file_results.append(file_result)

        return PullRequestAnalysisResult(
            owner=owner,
            repo=repo,
            pull_number=pull_request.number,
            title=pull_request.title,
            state=pull_request.state,
            head_sha=pull_request.head.sha,
            base_sha=pull_request.base.sha,
            html_url=str(pull_request.html_url),
            files=tuple(file_results),
        )

    async def _analyze_file(
        self,
        owner: str,
        repo: str,
        head_sha: str,
        changed_file: GitHubPullRequestFile,
    ) -> FileAnalysisResult:
        """Analyze a changed file and map findings to added lines."""

        filename = changed_file.filename

        if changed_file.status == "removed":
            return self._skipped_file(
                changed_file,
                skip_reason="deleted_file",
            )

        if PurePosixPath(filename).suffix.lower() != ".py":
            return self._skipped_file(
                changed_file,
                skip_reason="unsupported_file_type",
            )

        if changed_file.status == "unchanged":
            return self._skipped_file(
                changed_file,
                skip_reason="unchanged_file",
            )

        # Retrieve source from the exact PR head commit.
        try:
            source = await self._github_client.get_file_content(
                owner,
                repo,
                filename,
                head_sha,
            )
        except (GitHubError, ValueError) as exc:
            return self._failed_file(
                changed_file,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Run complete-file analysis.
        try:
            ast_result = self._ast_analyzer.analyze(
                source,
                file_path=filename,
            )

            risk_findings = self._risk_analyzer.analyze(
                source,
                file_path=filename,
            )

        except SyntaxError as exc:
            return self._failed_file(
                changed_file,
                error=f"SyntaxError: {exc.msg}",
            )

        # Preserve full-file findings, even when line mapping
        # is unavailable.
        added_line_numbers: frozenset[int] | None
        changed_line_findings: tuple[RiskFinding, ...] | None
        line_mapping_error: str | None = None

        patch = changed_file.patch

        if patch is None or not patch.strip():
            added_line_numbers = None
            changed_line_findings = None
            line_mapping_error = "patch_unavailable"

        elif not any(line.startswith("@@ ") for line in patch.splitlines()):
            added_line_numbers = None
            changed_line_findings = None
            line_mapping_error = "patch_has_no_hunks"

        else:
            try:
                parsed_line_numbers = parse_added_line_numbers(patch)

                if len(parsed_line_numbers) != changed_file.additions:
                    added_line_numbers = None
                    changed_line_findings = None
                    line_mapping_error = "patch_addition_count_mismatch"

                else:
                    added_line_numbers = parsed_line_numbers

                    changed_line_findings = filter_findings_to_added_lines(
                        risk_findings,
                        added_line_numbers,
                    )

            except ValueError as exc:
                added_line_numbers = None
                changed_line_findings = None
                line_mapping_error = f"{type(exc).__name__}: {exc}"

        return FileAnalysisResult(
            file_path=filename,
            status=changed_file.status,
            additions=changed_file.additions,
            deletions=changed_file.deletions,
            changes=changed_file.changes,
            previous_filename=changed_file.previous_filename,
            ast_result=ast_result,
            risk_findings=risk_findings,
            added_line_numbers=added_line_numbers,
            changed_line_findings=changed_line_findings,
            line_mapping_error=line_mapping_error,
        )

    @staticmethod
    def _skipped_file(
        changed_file: GitHubPullRequestFile,
        skip_reason: str,
    ) -> FileAnalysisResult:
        """Create a result for a file intentionally skipped."""

        return FileAnalysisResult(
            file_path=changed_file.filename,
            status=changed_file.status,
            additions=changed_file.additions,
            deletions=changed_file.deletions,
            changes=changed_file.changes,
            previous_filename=changed_file.previous_filename,
            ast_result=None,
            risk_findings=(),
            skip_reason=skip_reason,
        )

    @staticmethod
    def _failed_file(
        changed_file: GitHubPullRequestFile,
        error: str,
    ) -> FileAnalysisResult:
        """Create a result for an individual file-analysis failure."""

        return FileAnalysisResult(
            file_path=changed_file.filename,
            status=changed_file.status,
            additions=changed_file.additions,
            deletions=changed_file.deletions,
            changes=changed_file.changes,
            previous_filename=changed_file.previous_filename,
            ast_result=None,
            risk_findings=(),
            error=error,
        )
