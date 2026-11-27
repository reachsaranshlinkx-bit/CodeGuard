from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from codeguard.ai.contracts import AIFindingCandidate, AIReviewInput
from codeguard.ai.engine import AIReviewEngine, AIReviewExecutionResult
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchStatus,
)
from codeguard.ai.evidence_validation import EvidenceLocationReason
from codeguard.ai.execution_coordinator import AIExecutionCoordinator
from codeguard.ai.factory import AIReviewProviderSelection
from codeguard.ai.openai_provider import OpenAIReviewProvider
from codeguard.ai.operation_repository import (
    AIOperationRepository,
    AIOperationStatus,
)
from codeguard.context.repository_context_builder import RepositoryContextBuilder
from codeguard.github.schemas import (
    GitHubGitCommit,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubRepositoryTree,
)
from codeguard.review.report_builder import build_review_report
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer
from codeguard.services.webhook_job_worker import WebhookJobWorker
from codeguard.services.webhook_review_processor import (
    ReviewPublishResult,
    WebhookReviewProcessor,
)
from codeguard.webhooks.job_queue import WebhookJobQueue
from codeguard.webhooks.schema_migrations import migrate_webhook_database

OWNER = "octocat"
REPO = "CodeGuard"
PULL_NUMBER = 42
HEAD_SHA = "b" * 40
CHANGED_HEAD_SHA = "c" * 40
BASE_SHA = "a" * 40
TREE_SHA = "d" * 40
SOURCE = "def authenticate(value):\n    return eval(value)\n"
PATCH = (
    "@@ -1,2 +1,2 @@\n"
    " def authenticate(value):\n"
    "-    return value\n"
    "+    return eval(value)\n"
)
TEST_API_KEY = "test-openai-key-not-a-real-credential"
TEST_MODEL = "gpt-test-model"
ProviderResponseHandler = Callable[[httpx.Request], httpx.Response]


def make_pull_request(head_sha: str = HEAD_SHA) -> GitHubPullRequest:
    return GitHubPullRequest.model_validate(
        {
            "url": f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}",
            "html_url": f"https://github.com/{OWNER}/{REPO}/pull/{PULL_NUMBER}",
            "number": PULL_NUMBER,
            "state": "open",
            "title": "Improve authentication",
            "base": {"ref": "main", "sha": BASE_SHA},
            "head": {"ref": "feature/authentication", "sha": head_sha},
        }
    )


def make_changed_file() -> GitHubPullRequestFile:
    return GitHubPullRequestFile.model_validate(
        {
            "filename": "src/auth.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "changes": 2,
            "patch": PATCH,
        }
    )


def make_candidate(
    *,
    file_path: str = "src/auth.py",
    line_number: int = 2,
    evidence: str = "    return eval(value)",
) -> AIFindingCandidate:
    return AIFindingCandidate(
        title="Potential unsafe dynamic execution",
        severity="high",
        confidence="high",
        file_path=file_path,
        line_number=line_number,
        problem="The function evaluates its input as code.",
        evidence=evidence,
        why_it_matters="Untrusted input could execute arbitrary code.",
        suggested_fix="Replace dynamic evaluation with a constrained parser.",
        recommended_validation="Test that executable input is rejected.",
    )


def make_responses_api_body(candidate: AIFindingCandidate) -> dict[str, object]:
    structured_result = {
        "summary": "One potential issue was assessed.",
        "findings": [candidate.model_dump(mode="json")],
    }
    return {
        "id": "resp_integration_test",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(structured_result),
                        "annotations": [],
                    }
                ],
            }
        ],
    }


class MockGitHubClient:
    """Provide deterministic GitHub API responses without network access."""

    def __init__(
        self,
        *,
        commit_sha_override: str | None = None,
        missing_source_paths: tuple[str, ...] = (),
    ) -> None:
        self.pull_request = make_pull_request()
        self.changed_files = [make_changed_file()]
        self.commit_sha_override = commit_sha_override
        self.missing_source_paths = missing_source_paths
        self.review_lookup_calls = 0
        self.review_creation_calls = 0
        self.file_content_refs: list[tuple[str, str]] = []

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

    async def get_git_commit(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> GitHubGitCommit:
        resolved_sha = self.commit_sha_override or commit_sha
        return GitHubGitCommit.model_validate(
            {
                "sha": resolved_sha,
                "tree": {
                    "sha": TREE_SHA,
                    "url": (
                        f"https://api.github.com/repos/{owner}/{repo}"
                        f"/git/trees/{TREE_SHA}"
                    ),
                },
            }
        )

    async def get_repository_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        *,
        recursive: bool = True,
    ) -> GitHubRepositoryTree:
        paths = ("src/auth.py", *self.missing_source_paths)
        return GitHubRepositoryTree.model_validate(
            {
                "sha": tree_sha,
                "truncated": False,
                "tree": [
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": chr(ord("e") + index) * 40,
                        "size": len(SOURCE.encode("utf-8")),
                    }
                    for index, path in enumerate(paths)
                ],
            }
        )

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str,
    ) -> str:
        self.file_content_refs.append((path, ref))
        if path in self.missing_source_paths:
            raise RuntimeError("Mock source retrieval failed.")
        return SOURCE

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


@dataclass(slots=True)
class OpenAIReviewHarness:
    provider: OpenAIReviewProvider
    selection: AIReviewProviderSelection
    github_client: MockGitHubClient
    repository: AIOperationRepository
    analyzer: PullRequestAnalyzer
    processor: WebhookReviewProcessor
    requests: list[httpx.Request]


def make_harness(
    database_path: Path,
    *,
    candidate: AIFindingCandidate,
    commit_sha_override: str | None = None,
    missing_source_paths: tuple[str, ...] = (),
    provider_response_handler: ProviderResponseHandler | None = None,
) -> OpenAIReviewHarness:
    migrate_webhook_database(database_path)
    requests: list[httpx.Request] = []
    response_body = make_responses_api_body(candidate)

    def handle_provider_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if provider_response_handler is not None:
            return provider_response_handler(request)
        return httpx.Response(200, json=response_body, request=request)

    provider = OpenAIReviewProvider(
        model_identity=TEST_MODEL,
        api_key=SecretStr(TEST_API_KEY),
        transport=httpx.MockTransport(handle_provider_request),
    )
    selection = AIReviewProviderSelection(
        provider=provider,
        metadata=provider.execution_metadata,
    )
    github_client = MockGitHubClient(
        commit_sha_override=commit_sha_override,
        missing_source_paths=missing_source_paths,
    )
    repository = AIOperationRepository(database_path)
    coordinator = AIExecutionCoordinator(
        provider_selection=selection,
        repository=repository,
        lease_duration_seconds=60,
    )
    analyzer = PullRequestAnalyzer(
        github_client=github_client,
        repository_context_builder=RepositoryContextBuilder(github_client),
        ai_execution_coordinator=coordinator,
    )
    processor = WebhookReviewProcessor(
        analyzer=analyzer,
        github_client=github_client,
        publishing_enabled=False,
    )
    return OpenAIReviewHarness(
        provider=provider,
        selection=selection,
        github_client=github_client,
        repository=repository,
        analyzer=analyzer,
        processor=processor,
        requests=requests,
    )


def read_operations(database_path: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT operation_key, status FROM ai_operations ORDER BY operation_key"
        ).fetchall()
    return [(str(operation_key), str(status)) for operation_key, status in rows]


async def process_review(harness: OpenAIReviewHarness) -> ReviewPublishResult:
    return await harness.processor.process(
        owner=OWNER,
        repo=REPO,
        pull_number=PULL_NUMBER,
    )


@pytest.mark.asyncio
async def test_openai_pipeline_reuses_durable_result_and_distinguishes_pr_heads(
    tmp_path: Path,
) -> None:
    harness = make_harness(
        tmp_path / "openai-review.sqlite3",
        candidate=make_candidate(),
    )
    try:
        first_result = await process_review(harness)
        repeated_result = await process_review(harness)

        assert first_result.status == repeated_result.status == "dry_run"
        assert first_result.comments == repeated_result.comments
        assert len(harness.requests) == 1
        assert len(first_result.comments) == 2
        assert (
            sum(
                "AI review candidate" in comment.body
                for comment in first_result.comments
            )
            == 1
        )
        assert any("PY001" in comment.body for comment in first_result.comments)
        first_operations = read_operations(tmp_path / "openai-review.sqlite3")
        assert len(first_operations) == 1
        first_operation_key, first_status = first_operations[0]
        assert first_operation_key.startswith("ai-operation-v2:")
        assert first_status == AIOperationStatus.SUCCEEDED.value
        assert harness.github_client.review_lookup_calls == 0
        assert harness.github_client.review_creation_calls == 0

        harness.github_client.pull_request = make_pull_request(CHANGED_HEAD_SHA)
        changed_head_result = await process_review(harness)
        repeated_changed_head_result = await process_review(harness)

        assert changed_head_result.status == "dry_run"
        assert repeated_changed_head_result.comments == changed_head_result.comments
        assert len(harness.requests) == 2
        assert len(read_operations(tmp_path / "openai-review.sqlite3")) == 2
        assert (
            len(
                {
                    operation_key
                    for operation_key, _ in read_operations(
                        tmp_path / "openai-review.sqlite3"
                    )
                }
            )
            == 2
        )
        assert all(
            status == AIOperationStatus.SUCCEEDED.value
            for _, status in read_operations(tmp_path / "openai-review.sqlite3")
        )

        request_heads = [
            json.loads(request.content)["input"][0]["content"][0]["text"]
            for request in harness.requests
        ]
        assert json.loads(request_heads[0])["head_sha"] == HEAD_SHA
        assert json.loads(request_heads[1])["head_sha"] == CHANGED_HEAD_SHA
        assert all(
            request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
            for request in harness.requests
        )
    finally:
        await harness.selection.aclose()


@pytest.mark.parametrize(
    (
        "candidate_file_path",
        "candidate_line_number",
        "candidate_evidence",
        "missing_source_paths",
        "expected_status",
        "expected_reason",
    ),
    [
        pytest.param(
            "src/auth.py",
            2,
            "    return value",
            (),
            EvidenceMatchStatus.MISMATCHED,
            EvidenceMatchReason.CONTENT_MISMATCH,
            id="incorrect-evidence",
        ),
        pytest.param(
            "src/auth.py",
            3,
            "    return eval(value)",
            (),
            EvidenceMatchStatus.MISMATCHED,
            EvidenceMatchReason.LOCATION_INVALID,
            id="incorrect-line-number",
        ),
        pytest.param(
            "src/missing.py",
            1,
            "def missing():",
            ("src/missing.py",),
            EvidenceMatchStatus.UNVERIFIABLE,
            EvidenceMatchReason.LOCATION_UNVERIFIABLE,
            id="missing-source-file",
        ),
        pytest.param(
            "src/auth.py",
            1,
            "def authenticate(value):",
            (),
            EvidenceMatchStatus.MATCHED,
            EvidenceMatchReason.EXACT_MATCH,
            id="verified-but-not-on-an-added-line",
        ),
    ],
)
@pytest.mark.asyncio
async def test_uneligible_ai_candidates_never_become_review_comments(
    tmp_path: Path,
    candidate_file_path: str,
    candidate_line_number: int,
    candidate_evidence: str,
    missing_source_paths: tuple[str, ...],
    expected_status: EvidenceMatchStatus,
    expected_reason: EvidenceMatchReason,
) -> None:
    database_path = tmp_path / "openai-evidence.sqlite3"
    harness = make_harness(
        database_path,
        candidate=make_candidate(
            file_path=candidate_file_path,
            line_number=candidate_line_number,
            evidence=candidate_evidence,
        ),
        missing_source_paths=missing_source_paths,
    )
    try:
        analysis = await harness.analyzer.analyze(OWNER, REPO, PULL_NUMBER)

        assert isinstance(analysis.ai_review_result, AIReviewExecutionResult)
        assessment = analysis.ai_review_result.assessments[0]
        assert assessment.evidence_match.status is expected_status
        assert assessment.evidence_match.reason is expected_reason

        report = build_review_report(analysis)
        assert len(report.findings) == 1
        assert report.findings[0].rule_id == "PY001"
        assert len(report.ai_assessments) == 1

        publish_result = await process_review(harness)
        assert publish_result.status == "dry_run"
        assert publish_result.comment_count == 1
        assert all(
            "AI review candidate" not in comment.body
            for comment in publish_result.comments
        )
        assert "PY001" in publish_result.comments[0].body
        assert len(harness.requests) == 1
        assert len(read_operations(database_path)) == 1
        assert harness.github_client.review_lookup_calls == 0
        assert harness.github_client.review_creation_calls == 0
    finally:
        await harness.selection.aclose()


@pytest.mark.asyncio
async def test_worker_retries_confirmed_not_sent_failure_and_completes(
    tmp_path: Path,
) -> None:
    candidate = make_candidate()
    provider_calls = 0

    def provider_response_handler(request: httpx.Request) -> httpx.Response:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise httpx.ConnectError("private connection detail", request=request)
        return httpx.Response(
            200,
            json=make_responses_api_body(candidate),
            request=request,
        )

    database_path = tmp_path / "openai-worker-safe-retry.sqlite3"
    harness = make_harness(
        database_path,
        candidate=candidate,
        provider_response_handler=provider_response_handler,
    )
    queue = WebhookJobQueue(database_path)
    queue.enqueue_job(
        delivery_id="openai-safe-retry",
        event="pull_request",
        action="opened",
        owner=OWNER,
        repo=REPO,
        pull_number=PULL_NUMBER,
    )
    worker = WebhookJobWorker(
        queue=queue,
        processor=harness.processor,
        max_attempts=2,
        retry_delay_seconds=0,
        lease_seconds=10,
    )

    try:
        assert await worker.run_once() is True
        queued = queue.get_job("openai-safe-retry")
        assert queued is not None
        assert queued.status == "queued"
        assert queued.attempts == 1
        assert "AIExecutionRetryableError" in (queued.last_error or "")

        pending_operations = read_operations(database_path)
        assert len(pending_operations) == 1
        operation_key, operation_status = pending_operations[0]
        assert operation_status == AIOperationStatus.PENDING.value
        pending = harness.repository.get(operation_key)
        assert pending is not None
        assert pending.lease_expires_at is None
        assert pending.provider_request_started_at is None

        assert await worker.run_once() is True
        completed = queue.get_job("openai-safe-retry")
        assert completed is not None
        assert completed.status == "completed"
        assert completed.attempts == 2
        assert harness.repository.get(operation_key).status is (  # type: ignore[union-attr]
            AIOperationStatus.SUCCEEDED
        )
        assert len(harness.requests) == 2
    finally:
        await harness.selection.aclose()


@pytest.mark.asyncio
async def test_safe_not_sent_retries_stop_at_webhook_attempt_limit(
    tmp_path: Path,
) -> None:
    def provider_response_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private connection detail", request=request)

    database_path = tmp_path / "openai-worker-retry-limit.sqlite3"
    harness = make_harness(
        database_path,
        candidate=make_candidate(),
        provider_response_handler=provider_response_handler,
    )
    queue = WebhookJobQueue(database_path)
    queue.enqueue_job(
        delivery_id="openai-safe-retry-limit",
        event="pull_request",
        action="opened",
        owner=OWNER,
        repo=REPO,
        pull_number=PULL_NUMBER,
    )
    worker = WebhookJobWorker(
        queue=queue,
        processor=harness.processor,
        max_attempts=2,
        retry_delay_seconds=0,
        lease_seconds=10,
    )

    try:
        assert await worker.run_once() is True
        assert await worker.run_once() is True

        failed = queue.get_job("openai-safe-retry-limit")
        assert failed is not None
        assert failed.status == "failed"
        assert failed.attempts == 2
        assert await worker.run_once() is False
        assert len(harness.requests) == 2
        operation_key, operation_status = read_operations(database_path)[0]
        assert operation_status == AIOperationStatus.PENDING.value
        assert harness.repository.get(operation_key).status is (  # type: ignore[union-attr]
            AIOperationStatus.PENDING
        )
    finally:
        await harness.selection.aclose()


@pytest.mark.parametrize(
    ("failure_case", "expected_operation_status", "expected_diagnostic_parts"),
    [
        (
            "read_timeout",
            AIOperationStatus.OUTCOME_UNKNOWN,
            ("AIExecutionOutcomeUnknownError",),
        ),
        (
            "write_timeout",
            AIOperationStatus.OUTCOME_UNKNOWN,
            ("AIExecutionOutcomeUnknownError",),
        ),
        (
            "authentication",
            AIOperationStatus.FAILED_PERMANENT,
            ("AIExecutionPermanentFailureError", "status=401"),
        ),
        (
            "permission",
            AIOperationStatus.FAILED_PERMANENT,
            ("AIExecutionPermanentFailureError", "status=403"),
        ),
        (
            "malformed_response",
            AIOperationStatus.FAILED_PERMANENT,
            ("AIExecutionPermanentFailureError", "status=200"),
        ),
        (
            "rate_limit",
            AIOperationStatus.OUTCOME_UNKNOWN,
            (
                "AIExecutionOutcomeUnknownError",
                "status=429",
                "request_id=rate_limit_request-123",
                "retry_after_seconds=2.5",
            ),
        ),
        (
            "server_error",
            AIOperationStatus.OUTCOME_UNKNOWN,
            ("AIExecutionOutcomeUnknownError", "status=503"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_worker_persists_provider_failure_disposition_without_unsafe_retry(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure_case: str,
    expected_operation_status: AIOperationStatus,
    expected_diagnostic_parts: tuple[str, ...],
) -> None:
    private_response = "private provider response body and secret credential"

    def provider_response_handler(request: httpx.Request) -> httpx.Response:
        if failure_case == "read_timeout":
            raise httpx.ReadTimeout(private_response, request=request)
        if failure_case == "write_timeout":
            raise httpx.WriteTimeout(private_response, request=request)
        if failure_case == "malformed_response":
            return httpx.Response(
                200,
                content=private_response.encode(),
                headers={"x-request-id": "malformed-request-123"},
                request=request,
            )

        status_code = {
            "authentication": 401,
            "permission": 403,
            "rate_limit": 429,
            "server_error": 503,
        }[failure_case]
        headers = {"x-request-id": f"{failure_case}/request-123"}
        if failure_case == "rate_limit":
            headers["Retry-After"] = "2.5"
        return httpx.Response(
            status_code,
            content=private_response.encode(),
            headers=headers,
            request=request,
        )

    database_path = tmp_path / f"openai-worker-{failure_case}.sqlite3"
    harness = make_harness(
        database_path,
        candidate=make_candidate(),
        provider_response_handler=provider_response_handler,
    )
    queue = WebhookJobQueue(database_path)
    delivery_id = f"openai-failure-{failure_case}"
    queue.enqueue_job(
        delivery_id=delivery_id,
        event="pull_request",
        action="opened",
        owner=OWNER,
        repo=REPO,
        pull_number=PULL_NUMBER,
    )
    worker = WebhookJobWorker(
        queue=queue,
        processor=harness.processor,
        max_attempts=5,
        retry_delay_seconds=0,
        lease_seconds=10,
    )

    try:
        with caplog.at_level(logging.ERROR):
            assert await worker.run_once() is True

        failed_job = queue.get_job(delivery_id)
        assert failed_job is not None
        assert failed_job.status == "failed"
        assert failed_job.attempts == 1
        assert failed_job.last_error is not None
        for expected_part in expected_diagnostic_parts:
            assert expected_part in failed_job.last_error

        operations = read_operations(database_path)
        assert len(operations) == 1
        operation_key, operation_status = operations[0]
        assert operation_status == expected_operation_status.value
        assert await worker.run_once() is False
        assert len(harness.requests) == 1

        assert private_response not in failed_job.last_error
        assert private_response not in caplog.text
        assert TEST_API_KEY not in failed_job.last_error
        assert TEST_API_KEY not in caplog.text
        assert queue.claim_next_job(max_attempts=5) is None

        if failure_case == "rate_limit":
            assert "retry_after_seconds=2.5" in failed_job.last_error
    finally:
        await harness.selection.aclose()


@pytest.mark.asyncio
async def test_expired_webhook_lease_reuses_persisted_ai_result(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "openai-expired-webhook-lease.sqlite3"
    harness = make_harness(
        database_path,
        candidate=make_candidate(),
    )
    queue = WebhookJobQueue(database_path)
    queue.enqueue_job(
        delivery_id="openai-expired-lease",
        event="pull_request",
        action="opened",
        owner=OWNER,
        repo=REPO,
        pull_number=PULL_NUMBER,
    )
    first_claim = queue.claim_next_job(lease_seconds=10, max_attempts=3)
    assert first_claim is not None
    assert first_claim.attempts == 1

    try:
        first_result = await harness.processor.process(
            owner=OWNER,
            repo=REPO,
            pull_number=PULL_NUMBER,
        )
        assert first_result.status == "dry_run"
        assert len(harness.requests) == 1

        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """
                UPDATE webhook_jobs
                SET lease_expires_at = ?
                WHERE delivery_id = ?
                """,
                ("2000-01-01T00:00:00+00:00", "openai-expired-lease"),
            )

        worker = WebhookJobWorker(
            queue=queue,
            processor=harness.processor,
            max_attempts=3,
            retry_delay_seconds=0,
            lease_seconds=10,
        )
        assert await worker.run_once() is True

        recovered_job = queue.get_job("openai-expired-lease")
        assert recovered_job is not None
        assert recovered_job.status == "completed"
        assert recovered_job.attempts == 2
        assert len(harness.requests) == 1
        operation_key, operation_status = read_operations(database_path)[0]
        assert operation_status == AIOperationStatus.SUCCEEDED.value
        assert harness.repository.get(operation_key).status is (  # type: ignore[union-attr]
            AIOperationStatus.SUCCEEDED
        )
    finally:
        await harness.selection.aclose()


@pytest.mark.asyncio
async def test_openai_engine_marks_mismatched_commit_evidence_unverifiable() -> None:
    candidate = make_candidate()
    requests: list[httpx.Request] = []
    response_body = make_responses_api_body(candidate)

    def handle_provider_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body, request=request)

    provider = OpenAIReviewProvider(
        model_identity=TEST_MODEL,
        api_key=SecretStr(TEST_API_KEY),
        transport=httpx.MockTransport(handle_provider_request),
    )
    github_client = MockGitHubClient(commit_sha_override="f" * 40)
    try:
        context_result = await RepositoryContextBuilder(
            github_client
        ).build_with_source(
            owner=OWNER,
            repo=REPO,
            commit_sha=HEAD_SHA,
        )
        review_input = AIReviewInput(
            owner=OWNER,
            repo=REPO,
            pull_number=PULL_NUMBER,
            head_sha=HEAD_SHA,
            context="Test review input for commit verification.",
        )

        result = await AIReviewEngine(provider).review(
            review_input,
            context_result.source_snapshot,
        )

        assert result.assessments[0].evidence_match.status is (
            EvidenceMatchStatus.UNVERIFIABLE
        )
        assert result.assessments[0].evidence_match.reason is (
            EvidenceMatchReason.LOCATION_UNVERIFIABLE
        )
        assert result.assessments[0].evidence_match.location_reason is (
            EvidenceLocationReason.COMMIT_MISMATCH
        )
        assert len(requests) == 1
    finally:
        await provider.aclose()
