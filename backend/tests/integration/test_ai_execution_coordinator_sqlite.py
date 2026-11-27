from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from codeguard.ai.context_builder import CONTEXT_CONTRACT_VERSION
from codeguard.ai.contracts import (
    REVIEW_CONTRACT_VERSION,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.engine import AIReviewEngine, AIReviewExecutionResult
from codeguard.ai.evidence_matching import EVIDENCE_CONTRACT_VERSION
from codeguard.ai.exceptions import (
    AIProviderTimeoutError,
    AIRequestTransmissionState,
)
from codeguard.ai.execution_coordinator import (
    AIExecutionCoordinator,
    AIExecutionInProgressError,
    AIExecutionOutcomeUnknownError,
    AIExecutionPermanentFailureError,
    AIExecutionRetryableError,
)
from codeguard.ai.execution_fingerprint import build_ai_execution_input_fingerprint
from codeguard.ai.execution_metadata import AIExecutionMetadata
from codeguard.ai.factory import AIReviewProviderSelection
from codeguard.ai.mock_provider import (
    MOCK_PROMPT_CONTRACT_VERSION,
    MOCK_PROVIDER_IDENTITY,
    MockAIReviewProvider,
)
from codeguard.ai.operation_identity import (
    AIOperationIdentity,
    build_ai_operation_key,
)
from codeguard.ai.operation_repository import (
    AIOperationClaimStatus,
    AIOperationRepository,
    AIOperationStatus,
)
from codeguard.ai.operation_result_codec import encode
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.webhooks.schema_migrations import migrate_webhook_database

HEAD_SHA = "a" * 40
BASE_SHA = "c" * 40
PROFILE_IDENTITY = "durable-integration-profile-v1"


def make_review_input() -> AIReviewInput:
    return AIReviewInput(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        head_sha=HEAD_SHA,
        context="Review the changed authentication logic.",
    )


def make_snapshot() -> RepositorySourceSnapshot:
    source = "def authenticate(token):\n    return token\n"
    discovered_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="b" * 40,
        size_bytes=len(source.encode("utf-8")),
    )
    return RepositorySourceSnapshot(
        commit_sha=HEAD_SHA,
        files=(RepositorySourceFile(discovered_file, source),),
    )


def make_selection(
    *,
    provider: MockAIReviewProvider | None = None,
    profile_identity: str = PROFILE_IDENTITY,
    result: AIReviewResult | None = None,
) -> tuple[MockAIReviewProvider, AIReviewProviderSelection]:
    selected_provider = provider or MockAIReviewProvider(
        result=result,
        profile_identity=profile_identity,
    )
    metadata = AIExecutionMetadata(
        provider_identity=MOCK_PROVIDER_IDENTITY,
        model_identity=profile_identity,
        prompt_contract_version=MOCK_PROMPT_CONTRACT_VERSION,
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )
    return selected_provider, AIReviewProviderSelection(selected_provider, metadata)


def make_repository(
    database_path: Path, *, initialize: bool = False
) -> AIOperationRepository:
    if initialize:
        migrate_webhook_database(database_path)
    return AIOperationRepository(database_path)


def make_coordinator(
    repository: AIOperationRepository,
    selection: AIReviewProviderSelection,
) -> AIExecutionCoordinator:
    return AIExecutionCoordinator(
        provider_selection=selection,
        repository=repository,
        lease_duration_seconds=60,
        engine_factory=AIReviewEngine,
    )


def make_identity(
    selection: AIReviewProviderSelection,
) -> AIOperationIdentity:
    review_input = make_review_input()
    fingerprint = build_ai_execution_input_fingerprint(
        BASE_SHA,
        review_input,
        make_snapshot(),
    )
    return AIOperationIdentity(
        owner=review_input.owner,
        repo=review_input.repo,
        pull_number=review_input.pull_number,
        head_sha=review_input.head_sha,
        input_fingerprint=fingerprint,
        **selection.metadata.to_identity_components(),
    )


def operation_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        result = connection.execute(
            "SELECT COUNT(*) FROM ai_operations",
        ).fetchone()
    assert result is not None
    return result[0]


def stored_lease(
    database_path: Path, operation_key: str
) -> tuple[str | None, str | None]:
    with sqlite3.connect(database_path) as connection:
        result = connection.execute(
            """
            SELECT lease_owner, lease_expires_at
            FROM ai_operations
            WHERE operation_key = ?
            """,
            (operation_key,),
        ).fetchone()
    assert result is not None
    return result[0], result[1]


class GatedMockAIReviewProvider(MockAIReviewProvider):
    """Mock provider held at review() until the test releases its asyncio event."""

    def __init__(
        self,
        *,
        profile_identity: str = PROFILE_IDENTITY,
        result: AIReviewResult | None = None,
    ) -> None:
        super().__init__(result=result, profile_identity=profile_identity)
        self.review_started = asyncio.Event()
        self.release_review = asyncio.Event()
        self.call_count = 0

    async def review(self, review_input: AIReviewInput) -> AIReviewResult:
        self.call_count += 1
        self.review_started.set()
        await self.release_review.wait()
        return await super().review(review_input)


@pytest.mark.asyncio
async def test_successful_result_is_reused_by_a_new_coordinator_instance(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-reuse.sqlite3"
    expected_result = AIReviewResult(
        summary="Persisted mock review result.",
        findings=(),
    )
    provider_a, selection_a = make_selection(result=expected_result)
    repository_a = make_repository(database_path, initialize=True)
    coordinator_a = make_coordinator(repository_a, selection_a)

    first = await coordinator_a.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    persisted = repository_a.get(first.operation_key)
    assert persisted is not None
    assert persisted.status is AIOperationStatus.SUCCEEDED
    assert persisted.identity == make_identity(selection_a)
    assert operation_count(database_path) == 1

    provider_b, selection_b = make_selection(result=expected_result)
    repository_b = make_repository(database_path)
    coordinator_b = make_coordinator(repository_b, selection_b)
    second = await coordinator_b.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    assert second.operation_key == first.operation_key
    assert second.reused is True
    assert second.execution_result == first.execution_result
    assert second.execution_result.provider_result == expected_result
    assert provider_a.reviewed_inputs == (make_review_input(),)
    assert provider_b.reviewed_inputs == ()
    assert operation_count(database_path) == 1


@pytest.mark.asyncio
async def test_two_coordinators_cannot_execute_the_same_operation_concurrently(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-concurrent.sqlite3"
    repository_a = make_repository(database_path, initialize=True)
    repository_b = make_repository(database_path)
    provider_a = GatedMockAIReviewProvider()
    provider_b = MockAIReviewProvider(profile_identity=PROFILE_IDENTITY)
    _, selection_a = make_selection(provider=provider_a)
    _, selection_b = make_selection(provider=provider_b)
    coordinator_a = make_coordinator(repository_a, selection_a)
    coordinator_b = make_coordinator(repository_b, selection_b)

    first_task = asyncio.create_task(
        coordinator_a.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )
    )
    try:
        await asyncio.wait_for(provider_a.review_started.wait(), timeout=5)
        active_record = repository_b.get(
            build_ai_operation_key(make_identity(selection_b))
        )
        assert active_record is not None
        assert active_record.status is AIOperationStatus.REQUEST_STARTED

        with pytest.raises(AIExecutionOutcomeUnknownError):
            await coordinator_b.execute(
                make_review_input(),
                make_snapshot(),
                base_sha=BASE_SHA,
            )

        assert provider_a.call_count == 1
        assert provider_b.reviewed_inputs == ()
        assert operation_count(database_path) == 1
    finally:
        provider_a.release_review.set()
        await asyncio.wait_for(
            asyncio.gather(first_task, return_exceptions=True),
            timeout=5,
        )

    first = first_task.result()

    succeeded = repository_a.get(first.operation_key)
    assert first.reused is False
    assert succeeded is not None
    assert succeeded.status is AIOperationStatus.SUCCEEDED
    assert succeeded.result_json is not None
    assert succeeded.result_schema_version is not None
    assert succeeded.serialization_version is not None
    assert provider_a.call_count == 1
    assert len(provider_a.reviewed_inputs) == 1
    assert provider_b.reviewed_inputs == ()
    assert operation_count(database_path) == 1


@pytest.mark.asyncio
async def test_active_claimed_operation_is_reported_in_progress_without_reclaim(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-active-claim.sqlite3"
    repository_a = make_repository(database_path, initialize=True)
    repository_b = make_repository(database_path)
    provider_a, selection_a = make_selection()
    provider_b, selection_b = make_selection()
    operation = repository_a.create_or_get(make_identity(selection_a))
    claim = repository_a.claim(operation.operation_key, 60)
    assert claim.status is AIOperationClaimStatus.CLAIMED
    assert claim.lease_token is not None
    assert stored_lease(database_path, operation.operation_key)[0] == claim.lease_token

    with pytest.raises(AIExecutionInProgressError):
        await make_coordinator(repository_b, selection_b).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    active_record = repository_a.get(operation.operation_key)
    assert active_record is not None
    assert active_record.status is AIOperationStatus.CLAIMED
    assert stored_lease(database_path, operation.operation_key)[0] == claim.lease_token
    assert provider_a.reviewed_inputs == ()
    assert provider_b.reviewed_inputs == ()
    assert operation_count(database_path) == 1


@pytest.mark.asyncio
async def test_both_coordinators_reuse_result_after_initial_success(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-multiple-reuse.sqlite3"
    expected_result = AIReviewResult(
        summary="One durable result for all deliveries.",
        findings=(),
    )
    provider_a, selection_a = make_selection(result=expected_result)
    provider_b, selection_b = make_selection(result=expected_result)
    repository_a = make_repository(database_path, initialize=True)
    repository_b = make_repository(database_path)
    coordinator_a = make_coordinator(repository_a, selection_a)
    coordinator_b = make_coordinator(repository_b, selection_b)

    original = await coordinator_a.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    reused_a = await coordinator_a.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    reused_b = await coordinator_b.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    assert reused_a.reused is True
    assert reused_b.reused is True
    assert reused_a.execution_result == original.execution_result
    assert reused_b.execution_result == original.execution_result
    assert provider_a.reviewed_inputs == (make_review_input(),)
    assert provider_b.reviewed_inputs == ()
    assert operation_count(database_path) == 1


@pytest.mark.asyncio
async def test_success_is_durable_after_reopening_repository(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-reopen.sqlite3"
    expected_result = AIReviewResult(
        summary="Durable after reopening SQLite.",
        findings=(),
    )
    provider_a, selection_a = make_selection(result=expected_result)
    repository_a = make_repository(database_path, initialize=True)
    first = await make_coordinator(repository_a, selection_a).execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    initial_record = repository_a.get(first.operation_key)
    assert initial_record is not None

    del repository_a
    provider_b, selection_b = make_selection(result=expected_result)
    repository_b = make_repository(database_path)
    second = await make_coordinator(repository_b, selection_b).execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    reopened_record = repository_b.get(first.operation_key)

    assert reopened_record is not None
    assert second.reused is True
    assert second.execution_result == first.execution_result
    assert reopened_record.operation_key == build_ai_operation_key(
        make_identity(selection_b)
    )
    assert reopened_record.identity.input_fingerprint == (
        build_ai_execution_input_fingerprint(
            BASE_SHA,
            make_review_input(),
            make_snapshot(),
        )
    )
    assert reopened_record.result_json == initial_record.result_json
    assert reopened_record.result_schema_version == initial_record.result_schema_version
    assert reopened_record.serialization_version == initial_record.serialization_version
    assert provider_a.reviewed_inputs == (make_review_input(),)
    assert provider_b.reviewed_inputs == ()
    assert operation_count(database_path) == 1


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        (AIOperationStatus.REQUEST_STARTED, AIExecutionOutcomeUnknownError),
        (AIOperationStatus.OUTCOME_UNKNOWN, AIExecutionOutcomeUnknownError),
        (AIOperationStatus.FAILED_PERMANENT, AIExecutionPermanentFailureError),
    ],
)
@pytest.mark.asyncio
async def test_terminal_operation_states_are_not_retried(
    tmp_path: Path,
    status: AIOperationStatus,
    expected_error: type[Exception],
) -> None:
    database_path = tmp_path / f"coordinator-terminal-{status.value}.sqlite3"
    repository_a = make_repository(database_path, initialize=True)
    provider_a, selection_a = make_selection()
    operation = repository_a.create_or_get(make_identity(selection_a))
    claim = repository_a.claim(operation.operation_key, 60)
    assert claim.status is AIOperationClaimStatus.CLAIMED
    assert claim.lease_token is not None

    if status is AIOperationStatus.REQUEST_STARTED:
        repository_a.mark_request_started(operation.operation_key, claim.lease_token)
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "UPDATE ai_operations SET lease_expires_at = ? WHERE operation_key = ?",
                ("2000-01-01T00:00:00+00:00", operation.operation_key),
            )
    elif status is AIOperationStatus.OUTCOME_UNKNOWN:
        repository_a.mark_outcome_unknown(
            operation.operation_key,
            claim.lease_token,
            disposition="OUTCOME_UNKNOWN",
            safe_error="test",
        )
    else:
        repository_a.mark_failed_permanent(
            operation.operation_key,
            claim.lease_token,
            disposition="PERMANENT",
            safe_error="test",
        )

    provider_b, selection_b = make_selection()
    with pytest.raises(expected_error):
        await make_coordinator(
            make_repository(database_path),
            selection_b,
        ).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    final_record = repository_a.get(operation.operation_key)
    assert final_record is not None
    assert final_record.status is status
    assert provider_a.reviewed_inputs == ()
    assert provider_b.reviewed_inputs == ()
    assert operation_count(database_path) == 1
    if status is AIOperationStatus.REQUEST_STARTED:
        assert (
            stored_lease(database_path, operation.operation_key)[0] == claim.lease_token
        )


@pytest.mark.asyncio
async def test_confirmed_not_sent_can_be_retried_by_a_later_explicit_attempt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-safe-retry.sqlite3"
    repository_a = make_repository(database_path, initialize=True)
    provider_a = MockAIReviewProvider(
        error=AIProviderTimeoutError(
            "Connection was not established.",
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        ),
        profile_identity=PROFILE_IDENTITY,
    )
    _, selection_a = make_selection(provider=provider_a)

    with pytest.raises(AIExecutionRetryableError):
        await make_coordinator(repository_a, selection_a).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    identity = make_identity(selection_a)
    pending = repository_a.get(build_ai_operation_key(identity))
    assert pending is not None
    assert pending.status is AIOperationStatus.PENDING
    assert pending.lease_expires_at is None
    assert provider_a.reviewed_inputs == (make_review_input(),)

    provider_b, selection_b = make_selection()
    second_attempt = await make_coordinator(
        make_repository(database_path),
        selection_b,
    ).execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    succeeded = repository_a.get(second_attempt.operation_key)
    assert second_attempt.reused is False
    assert succeeded is not None
    assert succeeded.status is AIOperationStatus.SUCCEEDED
    assert provider_b.reviewed_inputs == (make_review_input(),)
    assert operation_count(database_path) == 1


@pytest.mark.asyncio
async def test_expired_claim_is_reclaimed_and_old_owner_cannot_persist_success(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "coordinator-lease-recovery.sqlite3"
    repository_a = make_repository(database_path, initialize=True)

    provider_a, selection_a = make_selection()
    identity = make_identity(selection_a)
    operation = repository_a.create_or_get(identity)

    old_claim = repository_a.claim(operation.operation_key, 60)
    assert old_claim.status is AIOperationClaimStatus.CLAIMED
    assert old_claim.lease_token is not None

    # Simulate the original lease expiring.
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET lease_expires_at = ? WHERE operation_key = ?",
            ("2000-01-01T00:00:00+00:00", operation.operation_key),
        )

    provider_b = GatedMockAIReviewProvider()
    _, selection_b = make_selection(provider=provider_b)

    coordinator_b = make_coordinator(
        make_repository(database_path),
        selection_b,
    )

    execution_task = asyncio.create_task(
        coordinator_b.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )
    )

    try:
        # Wait until the new owner has entered provider execution.
        await asyncio.wait_for(
            provider_b.review_started.wait(),
            timeout=5,
        )

        # The expired CLAIMED lease must have been replaced.
        active_record = repository_a.get(operation.operation_key)
        assert active_record is not None
        assert active_record.status is AIOperationStatus.REQUEST_STARTED

        new_lease_owner, new_lease_expiry = stored_lease(
            database_path,
            operation.operation_key,
        )

        assert new_lease_owner is not None
        assert new_lease_owner != old_claim.lease_token
        assert new_lease_expiry is not None

        assert provider_b.call_count == 1
        assert operation_count(database_path) == 1

        # The old lease owner must not be allowed to persist success.
        encoded = encode(
            AIReviewExecutionResult(
                provider_result=AIReviewResult(
                    summary="Stale lease must not persist this.",
                    findings=(),
                ),
                assessments=(),
            )
        )

        with pytest.raises(RuntimeError, match="transition rejected"):
            repository_a.mark_succeeded(
                operation.operation_key,
                old_claim.lease_token,
                result_json=encoded.result_json,
                result_schema_version=encoded.result_schema_version,
                serialization_version=encoded.serialization_version,
            )

        unchanged = repository_a.get(operation.operation_key)
        assert unchanged is not None
        assert unchanged.status is AIOperationStatus.REQUEST_STARTED
        assert unchanged.result_json is None

        assert (
            stored_lease(database_path, operation.operation_key)[0] == new_lease_owner
        )

    finally:
        # Always release the provider, even when an assertion fails.
        provider_b.release_review.set()

        await asyncio.wait_for(
            asyncio.gather(execution_task, return_exceptions=True),
            timeout=5,
        )

    # Only after release should the operation become SUCCEEDED.
    outcome = execution_task.result()
    succeeded = repository_a.get(operation.operation_key)

    assert outcome.reused is False
    assert succeeded is not None
    assert succeeded.status is AIOperationStatus.SUCCEEDED
    assert succeeded.result_json is not None
    assert succeeded.result_schema_version is not None
    assert succeeded.serialization_version is not None

    assert provider_b.call_count == 1
    assert len(provider_b.reviewed_inputs) == 1
    assert provider_a.reviewed_inputs == ()
    assert operation_count(database_path) == 1
