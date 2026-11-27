from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from codeguard.ai.context_builder import AIReviewContextBuilder
from codeguard.ai.contracts import AIFindingCandidate, AIReviewInput, AIReviewResult
from codeguard.ai.engine import AIReviewExecutionResult
from codeguard.ai.execution_coordinator import (
    AIExecutionCoordinator,
    AIExecutionInvalidStoredResultError,
    AIExecutionOutcomeUnknownError,
)
from codeguard.ai.execution_fingerprint import build_ai_execution_input_fingerprint
from codeguard.ai.factory import (
    AIReviewProviderSelection,
    create_ai_review_provider_selection,
)
from codeguard.ai.mock_provider import MockAIReviewProvider
from codeguard.ai.operation_identity import (
    AIOperationIdentity,
    LegacyAIOperationIdentity,
    build_ai_operation_key,
    build_legacy_ai_operation_key,
)
from codeguard.ai.operation_repository import (
    AIOperationRepository,
    AIOperationStatus,
)
from codeguard.ai.operation_result_codec import encode
from codeguard.analyzers.python_ast import PythonASTAnalyzer
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositoryContextBuildResult,
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.context.symbols import FileSymbolIndex, RepositorySymbolIndex
from codeguard.core.config import Settings
from codeguard.github.schemas import GitHubPullRequest, GitHubPullRequestFile
from codeguard.review.comment_mapper import build_inline_review_comments
from codeguard.review.report_builder import build_review_report
from codeguard.review.review_publisher import ReviewPublisher
from codeguard.services.pull_request_analyzer import (
    PullRequestAnalyzer,
)
from codeguard.webhooks.schema_migrations import migrate_webhook_database

OWNER = "octocat"
REPO = "CodeGuard"
PULL_NUMBER = 42
HEAD_SHA = "b" * 40
BASE_SHA = "a" * 40
PROFILE_IDENTITY = "pull-request-analyzer-integration-v1"
SOURCE = "def authenticate(value):\n    return eval(value)\n"
PATCH_TEMPLATE = (
    "@@ -1,2 +1,2 @@\n def authenticate(value):\n-    return value\n+{added_line}\n"
)


def make_pull_request(*, base_sha: str = BASE_SHA) -> GitHubPullRequest:
    return GitHubPullRequest.model_validate(
        {
            "url": f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}",
            "html_url": f"https://github.com/{OWNER}/{REPO}/pull/{PULL_NUMBER}",
            "number": PULL_NUMBER,
            "state": "open",
            "title": "Improve authentication",
            "base": {"ref": "main", "sha": base_sha},
            "head": {"ref": "feature/authentication", "sha": HEAD_SHA},
        }
    )


def make_changed_file(source: str = SOURCE) -> GitHubPullRequestFile:
    added_line = source.splitlines()[1]
    patch = PATCH_TEMPLATE.format(added_line=added_line)
    return GitHubPullRequestFile.model_validate(
        {
            "filename": "src/auth.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "changes": 2,
            "patch": patch,
        }
    )


class FakeGitHubClient:
    """Provide deterministic PR data and changed-file source without network calls."""

    def __init__(
        self,
        *,
        source: str = SOURCE,
        pull_request: GitHubPullRequest | None = None,
    ) -> None:
        self.pull_request = pull_request or make_pull_request()
        self.changed_files = [make_changed_file(source)]
        self.source = source
        self.file_content_calls: list[tuple[str, str, str, str]] = []
        self.review_lookup_calls = 0
        self.review_creation_calls = 0

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
        return self.pull_request

    async def get_all_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[GitHubPullRequestFile]:
        return self.changed_files

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        self.file_content_calls.append((owner, repo, path, ref))
        return self.source

    async def get_all_pull_request_reviews(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[dict[str, object]]:
        self.review_lookup_calls += 1
        return []

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, object]],
    ) -> dict[str, object]:
        self.review_creation_calls += 1
        return {"id": self.review_creation_calls, "state": "COMMENTED"}


class FakeRepositoryContextBuilder:
    """Return one deterministic symbol index and source snapshot."""

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


@dataclass(slots=True)
class AnalyzerHarness:
    analyzer: PullRequestAnalyzer
    github_client: FakeGitHubClient
    context_builder: FakeRepositoryContextBuilder
    provider: MockAIReviewProvider
    selection: AIReviewProviderSelection
    repository: AIOperationRepository
    coordinator: AIExecutionCoordinator
    context_result: RepositoryContextBuildResult


def make_context_result(
    source: str = SOURCE,
    *,
    tree_truncated: bool = False,
) -> RepositoryContextBuildResult:
    discovered_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="c" * 40,
        size_bytes=len(source.encode("utf-8")),
    )
    file_index = FileSymbolIndex(
        discovered_file=discovered_file,
        analysis=PythonASTAnalyzer().analyze(
            source,
            file_path=discovered_file.path,
        ),
    )
    symbol_index = RepositorySymbolIndex(
        commit_sha=HEAD_SHA,
        files=(file_index,),
        tree_truncated=tree_truncated,
    )
    source_snapshot = RepositorySourceSnapshot(
        commit_sha=HEAD_SHA,
        files=(RepositorySourceFile(discovered_file, source),),
        tree_truncated=tree_truncated,
    )
    return RepositoryContextBuildResult(
        symbol_index=symbol_index,
        source_snapshot=source_snapshot,
    )


def make_provider_selection(
    result: AIReviewResult,
    *,
    profile_identity: str = PROFILE_IDENTITY,
) -> tuple[MockAIReviewProvider, AIReviewProviderSelection]:
    selection = create_ai_review_provider_selection(
        Settings(
            _env_file=None,
            environment="testing",
            ai_provider="mock",
        ),
        mock_result=result,
        mock_profile_identity=profile_identity,
    )
    assert selection is not None
    assert isinstance(selection.provider, MockAIReviewProvider)
    return selection.provider, selection


def make_harness(
    database_path: Path,
    *,
    result: AIReviewResult,
    profile_identity: str = PROFILE_IDENTITY,
    source: str = SOURCE,
    pull_request: GitHubPullRequest | None = None,
    tree_truncated: bool = False,
) -> AnalyzerHarness:
    provider, selection = make_provider_selection(
        result,
        profile_identity=profile_identity,
    )
    repository = AIOperationRepository(database_path)
    coordinator = AIExecutionCoordinator(
        provider_selection=selection,
        repository=repository,
        lease_duration_seconds=60,
    )
    context_result = make_context_result(
        source,
        tree_truncated=tree_truncated,
    )
    github_client = FakeGitHubClient(
        source=source,
        pull_request=pull_request,
    )
    context_builder = FakeRepositoryContextBuilder(context_result)
    analyzer = PullRequestAnalyzer(
        github_client=github_client,
        repository_context_builder=context_builder,
        ai_execution_coordinator=coordinator,
    )
    return AnalyzerHarness(
        analyzer=analyzer,
        github_client=github_client,
        context_builder=context_builder,
        provider=provider,
        selection=selection,
        repository=repository,
        coordinator=coordinator,
        context_result=context_result,
    )


def make_expected_input(
    pull_request: GitHubPullRequest,
    source_snapshot: RepositorySourceSnapshot,
) -> AIReviewInput:
    context = AIReviewContextBuilder().build(
        changed_file_paths=("src/auth.py",),
        related_file_paths=(),
        source_snapshot=source_snapshot,
    )
    assert context is not None
    return AIReviewInput(
        owner=OWNER,
        repo=REPO,
        pull_number=pull_request.number,
        head_sha=pull_request.head.sha,
        context=context,
    )


def make_expected_identity(
    pull_request: GitHubPullRequest,
    source_snapshot: RepositorySourceSnapshot,
    selection: AIReviewProviderSelection,
) -> AIOperationIdentity:
    review_input = make_expected_input(pull_request, source_snapshot)
    fingerprint = build_ai_execution_input_fingerprint(
        pull_request.base.sha,
        review_input,
        source_snapshot,
    )
    return AIOperationIdentity(
        owner=review_input.owner,
        repo=review_input.repo,
        pull_number=review_input.pull_number,
        head_sha=review_input.head_sha,
        input_fingerprint=fingerprint,
        **selection.metadata.to_identity_components(),
    )


def make_candidate() -> AIFindingCandidate:
    return AIFindingCandidate(
        title="Potential unsafe dynamic execution",
        severity="high",
        confidence="high",
        file_path="src/auth.py",
        line_number=2,
        problem="The function evaluates its input as code.",
        evidence="    return eval(value)",
        why_it_matters="Untrusted input could execute arbitrary code.",
        suggested_fix="Replace dynamic evaluation with a constrained parser.",
        recommended_validation="Test that executable input is rejected.",
    )


def make_provider_result() -> AIReviewResult:
    return AIReviewResult(
        summary="One potential issue was assessed.",
        findings=(make_candidate(),),
    )


def read_operation_keys(database_path: Path) -> tuple[str, ...]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT operation_key FROM ai_operations ORDER BY operation_key",
        ).fetchall()
    return tuple(row[0] for row in rows)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    path = tmp_path / "pull-request-analysis.sqlite3"
    migrate_webhook_database(path)
    return path


@pytest.mark.asyncio
async def test_first_analysis_persists_complete_v2_result_and_repeat_reuses_it(
    database_path: Path,
) -> None:
    expected_result = make_provider_result()
    first_harness = make_harness(database_path, result=expected_result)

    first_analysis = await first_harness.analyzer.analyze(
        OWNER,
        REPO,
        PULL_NUMBER,
    )

    assert len(first_harness.provider.reviewed_inputs) == 1
    assert first_harness.context_builder.build_calls == []
    assert first_harness.context_builder.build_with_source_calls == [
        (OWNER, REPO, HEAD_SHA),
    ]
    assert first_harness.github_client.file_content_calls == [
        (OWNER, REPO, "src/auth.py", HEAD_SHA),
    ]
    assert isinstance(first_analysis.ai_review_result, AIReviewExecutionResult)

    expected_review_input = make_expected_input(
        make_pull_request(),
        first_harness.context_result.source_snapshot,
    )
    assert first_harness.provider.reviewed_inputs == (expected_review_input,)
    expected_identity = make_expected_identity(
        make_pull_request(),
        first_harness.context_result.source_snapshot,
        first_harness.selection,
    )
    expected_key = build_ai_operation_key(expected_identity)
    assert read_operation_keys(database_path) == (expected_key,)

    persisted = first_harness.repository.get(expected_key)
    assert persisted is not None
    assert persisted.identity == expected_identity
    assert persisted.operation_key == expected_key
    assert persisted.operation_key.startswith("ai-operation-v2:")
    assert persisted.status is AIOperationStatus.SUCCEEDED
    assert persisted.result_json is not None
    assert persisted.result_schema_version is not None
    assert persisted.serialization_version is not None
    encoded = encode(first_analysis.ai_review_result)
    assert persisted.result_json == encoded.result_json
    assert persisted.result_schema_version == encoded.result_schema_version
    assert persisted.serialization_version == encoded.serialization_version

    second_harness = make_harness(database_path, result=expected_result)
    second_analysis = await second_harness.analyzer.analyze(
        OWNER,
        REPO,
        PULL_NUMBER,
    )

    assert second_harness.provider.reviewed_inputs == ()
    assert read_operation_keys(database_path) == (expected_key,)
    assert second_analysis.ai_review_result == first_analysis.ai_review_result
    assert isinstance(second_analysis.ai_review_result, AIReviewExecutionResult)

    first_report = build_review_report(first_analysis)
    second_report = build_review_report(second_analysis)
    assert first_report == second_report
    assert first_report.ai_review_summary == second_report.ai_review_summary
    assert first_report.ai_assessments == second_report.ai_assessments
    assert first_report.ai_review_summary == expected_result.summary
    assert len(first_report.ai_assessments) == 1
    assert first_report.ai_assessments[0].evidence_match.status.value == "matched"
    assert first_report.ai_assessments[0].evidence_match.reason.value == "exact_match"
    assert first_report.findings == second_report.findings
    assert first_report.findings[0].severity == "high"
    assert first_report.findings[0].file_path == "src/auth.py"
    assert first_report.findings[0].line_number == 2

    first_comments = build_inline_review_comments(first_report, first_analysis)
    second_comments = build_inline_review_comments(second_report, second_analysis)
    assert first_comments == second_comments
    assert len(first_comments) == 2
    assert {(comment.path, comment.line) for comment in first_comments} == {
        ("src/auth.py", 2),
    }

    first_prepared = await ReviewPublisher(
        github_client=first_harness.github_client,
        dry_run=True,
    ).publish(first_report, first_analysis)
    second_prepared = await ReviewPublisher(
        github_client=second_harness.github_client,
        dry_run=True,
    ).publish(second_report, second_analysis)
    assert first_prepared.status == second_prepared.status == "dry_run"
    assert first_prepared.comments == second_prepared.comments
    assert first_prepared.review_body == second_prepared.review_body
    assert first_prepared.comment_count == second_prepared.comment_count == 2
    assert first_harness.github_client.review_lookup_calls == 0
    assert second_harness.github_client.review_lookup_calls == 0
    assert first_harness.github_client.review_creation_calls == 0
    assert second_harness.github_client.review_creation_calls == 0


@pytest.mark.parametrize(
    "change",
    ["base_sha", "source_content", "snapshot_coverage", "provider_profile"],
)
@pytest.mark.asyncio
async def test_changed_execution_inputs_create_distinct_durable_operations(
    database_path: Path,
    change: str,
) -> None:
    safe_result = AIReviewResult(summary="Identity test result.", findings=())
    first_harness = make_harness(database_path, result=safe_result)
    first_analysis = await first_harness.analyzer.analyze(
        OWNER,
        REPO,
        PULL_NUMBER,
    )
    assert isinstance(first_analysis.ai_review_result, AIReviewExecutionResult)

    base_sha = BASE_SHA
    source = SOURCE
    profile_identity = PROFILE_IDENTITY
    tree_truncated = False

    if change == "base_sha":
        base_sha = "d" * 40
    elif change == "source_content":
        source = "def authenticate(value):\n    return value\n"
    elif change == "snapshot_coverage":
        tree_truncated = True
    elif change == "provider_profile":
        profile_identity = f"{PROFILE_IDENTITY}-alternate"
    else:
        raise AssertionError(f"Unexpected test case: {change}")

    second_pull_request = make_pull_request(base_sha=base_sha)
    second_harness = make_harness(
        database_path,
        result=safe_result,
        profile_identity=profile_identity,
        source=source,
        pull_request=second_pull_request,
        tree_truncated=tree_truncated,
    )
    second_analysis = await second_harness.analyzer.analyze(
        OWNER,
        REPO,
        PULL_NUMBER,
    )

    assert isinstance(second_analysis.ai_review_result, AIReviewExecutionResult)
    assert len(first_harness.provider.reviewed_inputs) == 1
    assert len(second_harness.provider.reviewed_inputs) == 1

    first_identity = make_expected_identity(
        make_pull_request(),
        first_harness.context_result.source_snapshot,
        first_harness.selection,
    )
    second_identity = make_expected_identity(
        second_pull_request,
        second_harness.context_result.source_snapshot,
        second_harness.selection,
    )
    first_key = build_ai_operation_key(first_identity)
    second_key = build_ai_operation_key(second_identity)

    assert first_key != second_key
    assert set(read_operation_keys(database_path)) == {first_key, second_key}
    assert len(read_operation_keys(database_path)) == 2
    assert first_harness.repository.get(first_key).status is AIOperationStatus.SUCCEEDED  # type: ignore[union-attr]
    assert (
        second_harness.repository.get(second_key).status is AIOperationStatus.SUCCEEDED
    )  # type: ignore[union-attr]
    assert first_identity.input_fingerprint != second_identity.input_fingerprint or (
        first_identity.model_identity != second_identity.model_identity
    )


@pytest.mark.asyncio
async def test_request_started_operation_propagates_unknown_without_provider_retry(
    database_path: Path,
) -> None:
    provider, selection = make_provider_selection(
        AIReviewResult(summary="Must not execute again.", findings=())
    )
    context_result = make_context_result()
    expected_identity = make_expected_identity(
        make_pull_request(),
        context_result.source_snapshot,
        selection,
    )
    repository = AIOperationRepository(database_path)
    operation = repository.create_or_get(expected_identity)
    claim = repository.claim(operation.operation_key, 60)
    assert claim.lease_token is not None
    repository.mark_request_started(operation.operation_key, claim.lease_token)

    harness = make_harness(
        database_path,
        result=AIReviewResult(
            summary="Must not execute again.",
            findings=(),
        ),
    )
    assert harness.selection.metadata == selection.metadata
    with pytest.raises(AIExecutionOutcomeUnknownError):
        await harness.analyzer.analyze(OWNER, REPO, PULL_NUMBER)

    assert provider.reviewed_inputs == ()
    assert harness.provider.reviewed_inputs == ()
    persisted = repository.get(operation.operation_key)
    assert persisted is not None
    assert persisted.status is AIOperationStatus.REQUEST_STARTED
    assert read_operation_keys(database_path) == (operation.operation_key,)


@pytest.mark.asyncio
async def test_corrupt_cached_result_fails_closed_through_analyzer(
    database_path: Path,
) -> None:
    safe_result = AIReviewResult(summary="Persisted review.", findings=())
    first_harness = make_harness(database_path, result=safe_result)
    first_analysis = await first_harness.analyzer.analyze(
        OWNER,
        REPO,
        PULL_NUMBER,
    )
    assert isinstance(first_analysis.ai_review_result, AIReviewExecutionResult)
    operation_key = read_operation_keys(database_path)[0]

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET result_json = ? WHERE operation_key = ?",
            ("not-json", operation_key),
        )

    second_harness = make_harness(database_path, result=safe_result)
    with pytest.raises(AIExecutionInvalidStoredResultError):
        await second_harness.analyzer.analyze(OWNER, REPO, PULL_NUMBER)

    assert len(first_harness.provider.reviewed_inputs) == 1
    assert second_harness.provider.reviewed_inputs == ()
    assert (
        second_harness.repository.get(operation_key).status
        is AIOperationStatus.SUCCEEDED
    )  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_analyzer_v2_execution_does_not_reuse_legacy_v1_success(
    database_path: Path,
) -> None:
    safe_result = AIReviewResult(summary="Fresh v2 execution.", findings=())
    harness = make_harness(database_path, result=safe_result)
    expected_identity = make_expected_identity(
        make_pull_request(),
        harness.context_result.source_snapshot,
        harness.selection,
    )
    legacy_identity = LegacyAIOperationIdentity(
        owner=expected_identity.owner,
        repo=expected_identity.repo,
        pull_number=expected_identity.pull_number,
        head_sha=expected_identity.head_sha,
        provider_identity=expected_identity.provider_identity,
        model_identity=expected_identity.model_identity,
        prompt_contract_version=expected_identity.prompt_contract_version,
        context_contract_version=expected_identity.context_contract_version,
        evidence_contract_version=expected_identity.evidence_contract_version,
        review_contract_version=expected_identity.review_contract_version,
    )
    legacy_key = build_legacy_ai_operation_key(legacy_identity)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ai_operations (
                operation_key, owner, repo, pull_number, head_sha,
                provider_identity, model_identity, prompt_contract_version,
                context_contract_version, evidence_contract_version,
                review_contract_version, input_fingerprint, status, result_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                legacy_key,
                legacy_identity.owner,
                legacy_identity.repo,
                legacy_identity.pull_number,
                legacy_identity.head_sha,
                legacy_identity.provider_identity,
                legacy_identity.model_identity,
                legacy_identity.prompt_contract_version,
                legacy_identity.context_contract_version,
                legacy_identity.evidence_contract_version,
                legacy_identity.review_contract_version,
                AIOperationStatus.SUCCEEDED.value,
                '{"legacy":"opaque"}',
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    analysis = await harness.analyzer.analyze(OWNER, REPO, PULL_NUMBER)

    assert isinstance(analysis.ai_review_result, AIReviewExecutionResult)
    assert harness.provider.reviewed_inputs
    assert len(harness.provider.reviewed_inputs) == 1
    keys = read_operation_keys(database_path)
    assert legacy_key in keys
    assert len(keys) == 2
    v2_key = build_ai_operation_key(expected_identity)
    assert v2_key in keys
    assert v2_key != legacy_key
    assert harness.repository.get(legacy_key).status is AIOperationStatus.SUCCEEDED  # type: ignore[union-attr]
    assert harness.repository.get(v2_key).status is AIOperationStatus.SUCCEEDED  # type: ignore[union-attr]
