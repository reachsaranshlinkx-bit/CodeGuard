from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from codeguard.ai.context_builder import AIReviewContextBuilder
from codeguard.ai.contracts import AIReviewInput
from codeguard.ai.engine import AIReviewEngine, AIReviewExecutionResult
from codeguard.ai.execution_coordinator import AIExecutionCoordinator
from codeguard.analyzers.python_ast import (
    ClassInfo,
    FunctionInfo,
    PythonAnalysisResult,
    PythonASTAnalyzer,
)
from codeguard.analyzers.python_risk import (
    PythonRiskAnalyzer,
    RiskFinding,
)
from codeguard.context.source_snapshot import RepositoryContextBuildResult
from codeguard.context.symbols import RepositorySymbolIndex
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


class RepositoryContextBuilderProtocol(Protocol):
    """Interface required to build repository context."""

    async def build(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> RepositorySymbolIndex: ...


@dataclass(frozen=True, slots=True)
class RelatedCaller:
    """A function that calls a changed repository function."""

    file_path: str
    caller_qualified_name: str
    target_function_qualified_name: str
    lineno: int


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
    changed_functions: tuple[FunctionInfo, ...] | None = None
    changed_classes: tuple[ClassInfo, ...] | None = None
    related_callers: tuple[RelatedCaller, ...] | None = None


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
    repository_context: RepositorySymbolIndex | None = None
    ai_review_result: AIReviewExecutionResult | None = None


class PullRequestAnalyzer:
    """Orchestrate GitHub retrieval and Python static analysis."""

    def __init__(
        self,
        github_client: GitHubClientProtocol,
        ast_analyzer: PythonASTAnalyzer | None = None,
        risk_analyzer: PythonRiskAnalyzer | None = None,
        repository_context_builder: RepositoryContextBuilderProtocol | None = None,
        ai_review_engine: AIReviewEngine | None = None,
        ai_review_context_builder: AIReviewContextBuilder | None = None,
        ai_execution_coordinator: AIExecutionCoordinator | None = None,
    ) -> None:
        if ai_review_engine is not None and ai_execution_coordinator is not None:
            raise ValueError(
                "ai_review_engine and ai_execution_coordinator cannot both be set."
            )

        self._github_client = github_client
        self._ast_analyzer = ast_analyzer or PythonASTAnalyzer()
        self._risk_analyzer = risk_analyzer or PythonRiskAnalyzer()
        self._repository_context_builder = repository_context_builder
        self._ai_review_engine = ai_review_engine
        self._ai_execution_coordinator = ai_execution_coordinator
        self._ai_review_context_builder = (
            ai_review_context_builder
            if ai_review_context_builder is not None
            else AIReviewContextBuilder()
        )

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
        repository_context = None
        source_snapshot = None
        ai_execution_configured = (
            self._ai_execution_coordinator is not None
            or self._ai_review_engine is not None
        )

        if self._repository_context_builder is not None:
            if not ai_execution_configured:
                repository_context = await self._repository_context_builder.build(
                    owner=owner,
                    repo=repo,
                    commit_sha=pull_request.head.sha,
                )
            else:
                build_with_source = getattr(
                    self._repository_context_builder,
                    "build_with_source",
                    None,
                )

                if not callable(build_with_source):
                    raise TypeError(
                        "AI review requires a repository context builder with "
                        "a build_with_source() method."
                    )

                context_result = await build_with_source(
                    owner=owner,
                    repo=repo,
                    commit_sha=pull_request.head.sha,
                )

                if not isinstance(context_result, RepositoryContextBuildResult):
                    raise TypeError(
                        "Repository context builder build_with_source() must "
                        "return a RepositoryContextBuildResult."
                    )

                repository_context = context_result.symbol_index
                source_snapshot = context_result.source_snapshot
        elif ai_execution_configured:
            raise TypeError(
                "AI review requires a repository context builder with "
                "a build_with_source() method."
            )

        file_results: list[FileAnalysisResult] = []

        for changed_file in changed_files:
            file_result = await self._analyze_file(
                owner=owner,
                repo=repo,
                head_sha=pull_request.head.sha,
                changed_file=changed_file,
                repository_context=repository_context,
            )

            file_results.append(file_result)

        ai_review_result = None

        if ai_execution_configured:
            if source_snapshot is None:
                raise RuntimeError(
                    "AI review cannot proceed without a repository source snapshot."
                )

            changed_file_paths = tuple(
                file_result.file_path
                for file_result in file_results
                if file_result.ast_result is not None
            )
            related_file_paths = tuple(
                caller.file_path
                for file_result in file_results
                if file_result.related_callers is not None
                for caller in file_result.related_callers
            )
            context = self._ai_review_context_builder.build(
                changed_file_paths=changed_file_paths,
                related_file_paths=related_file_paths,
                source_snapshot=source_snapshot,
            )

            if context is not None:
                review_input = AIReviewInput(
                    owner=owner,
                    repo=repo,
                    pull_number=pull_request.number,
                    head_sha=pull_request.head.sha,
                    context=context,
                )
                if self._ai_execution_coordinator is not None:
                    outcome = await self._ai_execution_coordinator.execute(
                        review_input,
                        source_snapshot,
                        base_sha=pull_request.base.sha,
                    )
                    ai_review_result = outcome.execution_result
                elif self._ai_review_engine is not None:
                    ai_review_result = await self._ai_review_engine.review(
                        review_input,
                        source_snapshot,
                    )

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
            repository_context=repository_context,
            ai_review_result=ai_review_result,
        )

    async def _analyze_file(
        self,
        owner: str,
        repo: str,
        head_sha: str,
        changed_file: GitHubPullRequestFile,
        repository_context: RepositorySymbolIndex | None,
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

        if changed_file.additions == 0 and (patch is None or not patch.strip()):
            added_line_numbers = frozenset()
            changed_line_findings = ()
            line_mapping_error = None

        elif patch is None or not patch.strip():
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

        changed_functions, changed_classes = self._find_changed_symbols(
            file_path=filename,
            added_line_numbers=added_line_numbers,
            repository_context=repository_context,
        )
        related_callers = self._find_related_callers(
            changed_file_path=filename,
            changed_functions=changed_functions,
            repository_context=repository_context,
        )

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
            changed_functions=changed_functions,
            changed_classes=changed_classes,
            related_callers=related_callers,
        )

    @staticmethod
    def _find_related_callers(
        changed_file_path: str,
        changed_functions: tuple[FunctionInfo, ...] | None,
        repository_context: RepositorySymbolIndex | None,
    ) -> tuple[RelatedCaller, ...] | None:
        """Find direct callers of changed top-level imported functions."""

        if changed_functions is None or repository_context is None:
            return None

        if not changed_functions:
            return ()

        # Map Python module names to repository file paths.
        module_to_paths: dict[str, set[str]] = {}

        for file_index in repository_context.files:
            path = file_index.discovered_file.path

            if not path.endswith(".py"):
                continue

            module_path = path[:-3]

            if module_path.endswith("/__init__"):
                module_path = module_path[:-9]

            module_name = module_path.replace("/", ".")

            # Support both repository-relative imports and src-layout imports.
            module_names = {module_name}

            if module_name.startswith("src."):
                module_names.add(module_name.removeprefix("src."))

            for candidate_name in module_names:
                module_to_paths.setdefault(candidate_name, set()).add(path)

        changed_functions_by_name = {
            function.qualified_name: function for function in changed_functions
        }

        related_callers: list[RelatedCaller] = []
        seen: set[tuple[str, str, str, int]] = set()

        # Inspect every other indexed repository file.
        for caller_file in repository_context.files:
            caller_path = caller_file.discovered_file.path

            if caller_path == changed_file_path:
                continue

            for imported in caller_file.analysis.imports:
                # Only resolve explicit absolute imports at this stage.
                if imported.level != 0 or imported.module is None:
                    continue

                target_paths = module_to_paths.get(imported.module, set())

                # Resolve only when the import uniquely identifies this file.
                if target_paths != {changed_file_path}:
                    continue

                target_function = changed_functions_by_name.get(imported.name)

                if target_function is None:
                    continue

                local_name = imported.alias or imported.name

                for call in caller_file.analysis.calls:
                    if call.callee != local_name:
                        continue

                    if call.caller_qualified_name is None:
                        continue

                    key = (
                        caller_path,
                        call.caller_qualified_name,
                        target_function.qualified_name,
                        call.lineno,
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    related_callers.append(
                        RelatedCaller(
                            file_path=caller_path,
                            caller_qualified_name=call.caller_qualified_name,
                            target_function_qualified_name=(
                                target_function.qualified_name
                            ),
                            lineno=call.lineno,
                        )
                    )

        return tuple(related_callers)

    @staticmethod
    def _find_changed_symbols(
        file_path: str,
        added_line_numbers: frozenset[int] | None,
        repository_context: RepositorySymbolIndex | None,
    ) -> tuple[
        tuple[FunctionInfo, ...] | None,
        tuple[ClassInfo, ...] | None,
    ]:
        """Find repository symbols whose source ranges overlap added lines."""

        if added_line_numbers is None or repository_context is None:
            return None, None

        file_index = next(
            (
                candidate
                for candidate in repository_context.files
                if candidate.discovered_file.path == file_path
            ),
            None,
        )

        if file_index is None:
            return None, None

        changed_functions = tuple(
            function
            for function in file_index.analysis.functions
            if any(
                function.lineno <= line <= function.end_lineno
                for line in added_line_numbers
            )
        )

        changed_classes = tuple(
            class_info
            for class_info in file_index.analysis.classes
            if any(
                class_info.lineno <= line <= class_info.end_lineno
                for line in added_line_numbers
            )
        )

        return changed_functions, changed_classes

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
