from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import Event

import pytest

import codeguard.ai.execution_coordinator as coordinator_module
from codeguard.ai.context_builder import CONTEXT_CONTRACT_VERSION
from codeguard.ai.contracts import (
    REVIEW_CONTRACT_VERSION,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.engine import AIReviewEngine
from codeguard.ai.evidence_matching import EVIDENCE_CONTRACT_VERSION
from codeguard.ai.exceptions import (
    AIProviderConfigurationError,
    AIProviderRequestError,
    AIProviderTimeoutError,
    AIRequestTransmissionState,
)
from codeguard.ai.execution_coordinator import (
    AIExecutionCoordinator,
    AIExecutionInProgressError,
    AIExecutionInvalidStoredResultError,
    AIExecutionLeaseLostError,
    AIExecutionOutcome,
    AIExecutionOutcomeUnknownError,
    AIExecutionPermanentFailureError,
    AIExecutionPersistenceError,
    AIExecutionRetryableError,
    ThreadedRepositoryCallAdapter,
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
    LegacyAIOperationIdentity,
    build_ai_operation_key,
    build_legacy_ai_operation_key,
)
from codeguard.ai.operation_repository import (
    AIOperationClaimStatus,
    AIOperationRepository,
    AIOperationStatus,
)
from codeguard.ai.operation_result_codec import (
    AIReviewResultCodecError,
    encode,
)
from codeguard.ai.provider import AIReviewProvider
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)
from codeguard.webhooks.schema_migrations import migrate_webhook_database

HEAD_SHA = "a" * 40
BASE_SHA = "c" * 40
PROFILE_IDENTITY = "coordinator-test-profile-v1"


def make_review_input(**updates: object) -> AIReviewInput:
    values: dict[str, object] = {
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
        "head_sha": HEAD_SHA,
        "context": "Review the changed authentication logic.",
    }
    values.update(updates)
    return AIReviewInput(**values)  # type: ignore[arg-type]


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
) -> tuple[MockAIReviewProvider, AIReviewProviderSelection]:
    selected_provider = provider or MockAIReviewProvider(
        profile_identity=PROFILE_IDENTITY,
    )
    metadata = AIExecutionMetadata(
        provider_identity=MOCK_PROVIDER_IDENTITY,
        model_identity=PROFILE_IDENTITY,
        prompt_contract_version=MOCK_PROMPT_CONTRACT_VERSION,
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )
    return selected_provider, AIReviewProviderSelection(selected_provider, metadata)


def make_identity(
    selection: AIReviewProviderSelection,
    review_input: AIReviewInput | None = None,
    snapshot: RepositorySourceSnapshot | None = None,
) -> AIOperationIdentity:
    actual_input = review_input or make_review_input()
    actual_snapshot = snapshot or make_snapshot()
    fingerprint = build_ai_execution_input_fingerprint(
        BASE_SHA,
        actual_input,
        actual_snapshot,
    )
    return AIOperationIdentity(
        owner=actual_input.owner,
        repo=actual_input.repo,
        pull_number=actual_input.pull_number,
        head_sha=actual_input.head_sha,
        input_fingerprint=fingerprint,
        **selection.metadata.to_identity_components(),
    )


@pytest.fixture
def repository(tmp_path: Path) -> AIOperationRepository:
    database_path = tmp_path / "ai_operations.sqlite3"
    migrate_webhook_database(database_path)
    return AIOperationRepository(database_path)


def make_coordinator(
    repository: AIOperationRepository,
    selection: AIReviewProviderSelection,
    *,
    engine_factory: Callable[[AIReviewProvider], AIReviewEngine] | None = None,
) -> AIExecutionCoordinator:
    kwargs: dict[str, object] = {
        "provider_selection": selection,
        "repository": repository,
        "lease_duration_seconds": 60,
    }
    if engine_factory is not None:
        kwargs["engine_factory"] = engine_factory
    return AIExecutionCoordinator(**kwargs)  # type: ignore[arg-type]


def seed_legacy_success(
    repository: AIOperationRepository,
    selection: AIReviewProviderSelection,
    *,
    review_input: AIReviewInput,
) -> str:
    legacy_identity = LegacyAIOperationIdentity(
        owner=review_input.owner,
        repo=review_input.repo,
        pull_number=review_input.pull_number,
        head_sha=review_input.head_sha,
        **selection.metadata.to_identity_components(),
    )
    operation_key = build_legacy_ai_operation_key(legacy_identity)
    with sqlite3.connect(repository.database_path) as connection:
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
                operation_key,
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
    return operation_key


@pytest.mark.asyncio
async def test_identity_fingerprint_uses_exact_input_and_selected_metadata(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    received_providers: list[object] = []

    def engine_factory(selected_provider: AIReviewProvider) -> AIReviewEngine:
        received_providers.append(selected_provider)
        return AIReviewEngine(selected_provider)

    coordinator = make_coordinator(
        repository,
        selection,
        engine_factory=engine_factory,
    )

    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    record = repository.get(outcome.operation_key)
    assert record is not None
    assert isinstance(record.identity, AIOperationIdentity)
    assert record.identity.input_fingerprint == build_ai_execution_input_fingerprint(
        BASE_SHA,
        make_review_input(),
        make_snapshot(),
    )
    assert record.operation_key == build_ai_operation_key(record.identity)
    assert record.identity.provider_identity == selection.metadata.provider_identity
    assert record.identity.model_identity == selection.metadata.model_identity
    assert received_providers == [provider]


@pytest.mark.asyncio
async def test_first_execution_persists_complete_result_and_returns_typed_outcome(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    configured_result = AIReviewResult(
        summary="Mock review completed.",
        findings=(),
    )
    provider = MockAIReviewProvider(
        result=configured_result,
        profile_identity=PROFILE_IDENTITY,
    )
    _, selection = make_selection(provider=provider)
    coordinator = make_coordinator(repository, selection)

    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    record = repository.get(outcome.operation_key)
    encoded = encode(outcome.execution_result)
    assert isinstance(outcome, AIExecutionOutcome)
    assert outcome.reused is False
    assert outcome.execution_result.provider_result == configured_result
    assert outcome.execution_result.assessments == ()
    assert provider.reviewed_inputs == (make_review_input(),)
    assert record is not None
    assert record.status is AIOperationStatus.SUCCEEDED
    assert record.result_json == encoded.result_json
    assert record.result_schema_version == encoded.result_schema_version
    assert record.serialization_version == encoded.serialization_version


@pytest.mark.asyncio
async def test_successful_result_is_reused_without_second_provider_call(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    coordinator = make_coordinator(repository, selection)

    first = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    second = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    assert first.reused is False
    assert second.reused is True
    assert second.execution_result == first.execution_result
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_missing_cached_result_json_is_rejected(
    repository: AIOperationRepository,
) -> None:
    _, selection = make_selection()
    coordinator = make_coordinator(repository, selection)
    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET result_json = NULL WHERE operation_key = ?",
            (outcome.operation_key,),
        )

    with pytest.raises(AIExecutionInvalidStoredResultError, match="missing"):
        await coordinator.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )


@pytest.mark.parametrize(
    "column",
    ["result_schema_version", "serialization_version"],
)
@pytest.mark.asyncio
async def test_missing_cached_codec_version_is_rejected(
    repository: AIOperationRepository,
    column: str,
) -> None:
    _, selection = make_selection()
    coordinator = make_coordinator(repository, selection)
    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            f"UPDATE ai_operations SET {column} = NULL WHERE operation_key = ?",
            (outcome.operation_key,),
        )

    with pytest.raises(AIExecutionInvalidStoredResultError, match="versions"):
        await coordinator.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )


@pytest.mark.asyncio
async def test_corrupted_cached_result_fails_closed_without_provider_execution(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    coordinator = make_coordinator(repository, selection)
    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET result_json = ? WHERE operation_key = ?",
            ("not-json", outcome.operation_key),
        )

    with pytest.raises(AIExecutionInvalidStoredResultError) as error:
        await coordinator.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    assert isinstance(error.value.__cause__, AIReviewResultCodecError)
    assert len(provider.reviewed_inputs) == 1
    assert repository.get(outcome.operation_key).status is AIOperationStatus.SUCCEEDED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_unsupported_cached_codec_version_fails_closed(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    coordinator = make_coordinator(repository, selection)
    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )
    record = repository.get(outcome.operation_key)
    assert record is not None and record.result_json is not None
    envelope = json.loads(record.result_json)
    envelope["serialization_version"] = 900
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            UPDATE ai_operations
            SET result_json = ?, serialization_version = 900
            WHERE operation_key = ?
            """,
            (json.dumps(envelope), outcome.operation_key),
        )

    with pytest.raises(AIExecutionInvalidStoredResultError) as error:
        await coordinator.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    assert isinstance(error.value.__cause__, AIReviewResultCodecError)
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_active_claim_is_not_executed(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    identity = make_identity(selection)
    record = repository.create_or_get(identity)
    claim = repository.claim(record.operation_key, 60)
    assert claim.status is AIOperationClaimStatus.CLAIMED

    with pytest.raises(AIExecutionInProgressError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    assert provider.reviewed_inputs == ()


@pytest.mark.asyncio
async def test_request_started_operation_is_not_reclaimed(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    identity = make_identity(selection)
    record = repository.create_or_get(identity)
    claim = repository.claim(record.operation_key, 60)
    assert claim.lease_token is not None
    repository.mark_request_started(record.operation_key, claim.lease_token)

    with pytest.raises(AIExecutionOutcomeUnknownError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    assert provider.reviewed_inputs == ()


@pytest.mark.asyncio
async def test_permanent_provider_failure_is_persisted(
    repository: AIOperationRepository,
) -> None:
    provider = MockAIReviewProvider(
        error=AIProviderConfigurationError("Invalid mock configuration."),
        profile_identity=PROFILE_IDENTITY,
    )
    _, selection = make_selection(provider=provider)

    with pytest.raises(AIExecutionPermanentFailureError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    identity = make_identity(selection)
    record = repository.get(build_ai_operation_key(identity))
    assert record is not None
    assert record.status is AIOperationStatus.FAILED_PERMANENT
    assert record.disposition == "PERMANENT"
    assert record.safe_error == "AIProviderConfigurationError"
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_unknown_provider_failure_is_persisted_and_not_retried(
    repository: AIOperationRepository,
) -> None:
    provider = MockAIReviewProvider(
        error=AIProviderRequestError("Transmission was not confirmed."),
        profile_identity=PROFILE_IDENTITY,
    )
    _, selection = make_selection(provider=provider)
    coordinator = make_coordinator(repository, selection)

    with pytest.raises(AIExecutionOutcomeUnknownError):
        await coordinator.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )
    with pytest.raises(AIExecutionOutcomeUnknownError):
        await coordinator.execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.OUTCOME_UNKNOWN
    assert record.disposition == "OUTCOME_UNKNOWN"
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_confirmed_not_sent_failure_returns_operation_to_pending(
    repository: AIOperationRepository,
) -> None:
    provider = MockAIReviewProvider(
        error=AIProviderTimeoutError(
            "Connection was not established.",
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        ),
        profile_identity=PROFILE_IDENTITY,
    )
    _, selection = make_selection(provider=provider)

    with pytest.raises(AIExecutionRetryableError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.PENDING
    assert record.provider_request_started_at is None
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_concurrent_claim_loss_is_reported_as_in_progress(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    identity = make_identity(selection)
    record = repository.create_or_get(identity)
    active_claim = repository.claim(record.operation_key, 60)
    assert active_claim.status is AIOperationClaimStatus.CLAIMED

    with pytest.raises(AIExecutionInProgressError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    assert provider.reviewed_inputs == ()


@pytest.mark.asyncio
async def test_stale_lease_cannot_persist_success(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()
    actual_mark_succeeded = repository.mark_succeeded
    actual_mark_safe_to_retry = repository.mark_safe_to_retry
    actual_claim = repository.claim

    def replace_owner_before_persist(
        operation_key: str,
        lease_token: str,
        **kwargs: object,
    ):
        actual_mark_safe_to_retry(
            operation_key,
            lease_token,
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )
        new_claim = actual_claim(operation_key, 60)
        assert new_claim.status is AIOperationClaimStatus.CLAIMED
        return actual_mark_succeeded(operation_key, lease_token, **kwargs)

    monkeypatch.setattr(repository, "mark_succeeded", replace_owner_before_persist)

    with pytest.raises(AIExecutionLeaseLostError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.CLAIMED
    assert record.result_json is None
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_lost_lease_before_request_start_does_not_invoke_provider(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()
    actual_claim = repository.claim

    def claim_then_replace_owner(operation_key: str, lease_duration_seconds: float):
        stale_claim = actual_claim(operation_key, lease_duration_seconds)
        assert stale_claim.lease_token is not None
        with sqlite3.connect(repository.database_path) as connection:
            connection.execute(
                "UPDATE ai_operations SET lease_expires_at = ? WHERE operation_key = ?",
                ("2000-01-01T00:00:00+00:00", operation_key),
            )
        replacement_claim = actual_claim(operation_key, lease_duration_seconds)
        assert replacement_claim.status is AIOperationClaimStatus.CLAIMED
        return stale_claim

    monkeypatch.setattr(repository, "claim", claim_then_replace_owner)

    with pytest.raises(AIExecutionLeaseLostError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    assert provider.reviewed_inputs == ()
    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.CLAIMED


@pytest.mark.asyncio
async def test_cancel_during_claim_waits_for_sqlite_call_and_never_invokes_provider(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()
    actual_claim = repository.claim
    claim_completed = Event()
    allow_claim_return = Event()

    def delayed_claim(operation_key: str, lease_duration_seconds: float):
        result = actual_claim(operation_key, lease_duration_seconds)
        claim_completed.set()
        if not allow_claim_return.wait(timeout=5):
            raise TimeoutError("Test did not release the claim call.")
        return result

    monkeypatch.setattr(repository, "claim", delayed_claim)
    task = asyncio.create_task(
        make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )
    )
    assert await asyncio.to_thread(claim_completed.wait, 5)
    task.cancel()
    allow_claim_return.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.reviewed_inputs == ()
    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.CLAIMED


@pytest.mark.asyncio
async def test_cancel_during_request_start_marks_unknown_without_provider_call(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()
    actual_mark_started = repository.mark_request_started
    start_completed = Event()
    allow_start_return = Event()

    def delayed_start(operation_key: str, lease_token: str):
        record = actual_mark_started(operation_key, lease_token)
        start_completed.set()
        if not allow_start_return.wait(timeout=5):
            raise TimeoutError("Test did not release request-start persistence.")
        return record

    monkeypatch.setattr(repository, "mark_request_started", delayed_start)
    task = asyncio.create_task(
        make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )
    )
    assert await asyncio.to_thread(start_completed.wait, 5)
    task.cancel()
    allow_start_return.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.reviewed_inputs == ()
    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.OUTCOME_UNKNOWN
    assert record.disposition == "cancelled_after_request_started"


class BlockingMockProvider(MockAIReviewProvider):
    def __init__(self) -> None:
        super().__init__(profile_identity=PROFILE_IDENTITY)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def review(self, review_input: AIReviewInput) -> AIReviewResult:
        self.started.set()
        await self.release.wait()
        return await super().review(review_input)


@pytest.mark.asyncio
async def test_cancel_after_provider_execution_starts_marks_unknown(
    repository: AIOperationRepository,
) -> None:
    provider = BlockingMockProvider()
    _, selection = make_selection(provider=provider)
    task = asyncio.create_task(
        make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.OUTCOME_UNKNOWN
    assert provider.reviewed_inputs == ()


@pytest.mark.asyncio
async def test_threaded_repository_adapter_waits_before_propagating_cancel() -> None:
    adapter = ThreadedRepositoryCallAdapter()
    started = Event()
    release = Event()
    finished = Event()

    def blocking_operation() -> str:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Test did not release repository operation.")
        finished.set()
        return "committed"

    task = asyncio.create_task(adapter.run(blocking_operation))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()


@pytest.mark.asyncio
async def test_encoding_failure_after_execution_marks_unknown(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()

    def fail_encoding(result: object):
        raise AIReviewResultCodecError("Mock encoding failure.")

    monkeypatch.setattr(
        coordinator_module.operation_result_codec,
        "encode",
        fail_encoding,
    )

    with pytest.raises(AIExecutionPersistenceError, match="encoded"):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.OUTCOME_UNKNOWN
    assert record.disposition == "result_encoding_failed"
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_persistence_failure_after_provider_success_is_not_reported_as_success(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()

    def fail_success_persistence(*args: object, **kwargs: object):
        raise RuntimeError("Simulated SQLite failure.")

    monkeypatch.setattr(repository, "mark_succeeded", fail_success_persistence)

    with pytest.raises(AIExecutionPersistenceError):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha=BASE_SHA,
        )

    record = repository.get(build_ai_operation_key(make_identity(selection)))
    assert record is not None
    assert record.status is AIOperationStatus.OUTCOME_UNKNOWN
    assert record.result_json is None
    assert len(provider.reviewed_inputs) == 1


@pytest.mark.asyncio
async def test_engine_factory_receives_provider_from_metadata_selection(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()
    received_providers: list[object] = []

    def engine_factory(selected_provider: AIReviewProvider) -> AIReviewEngine:
        received_providers.append(selected_provider)
        return AIReviewEngine(selected_provider)

    coordinator = make_coordinator(
        repository,
        selection,
        engine_factory=engine_factory,
    )
    outcome = await coordinator.execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    assert received_providers == [provider]
    assert provider.reviewed_inputs == (make_review_input(),)
    assert outcome.reused is False


@pytest.mark.asyncio
async def test_legacy_success_is_not_decoded_or_reused(
    repository: AIOperationRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, selection = make_selection()
    legacy_key = seed_legacy_success(
        repository,
        selection,
        review_input=make_review_input(),
    )

    def reject_legacy_decode(*args: object, **kwargs: object):
        raise AssertionError("Legacy results must not be decoded.")

    monkeypatch.setattr(
        coordinator_module.operation_result_codec,
        "decode",
        reject_legacy_decode,
    )
    outcome = await make_coordinator(repository, selection).execute(
        make_review_input(),
        make_snapshot(),
        base_sha=BASE_SHA,
    )

    assert outcome.operation_key != legacy_key
    assert outcome.reused is False
    assert len(provider.reviewed_inputs) == 1
    assert repository.get(legacy_key).result_json == '{"legacy":"opaque"}'  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_invalid_fingerprint_inputs_fail_before_operation_creation(
    repository: AIOperationRepository,
) -> None:
    provider, selection = make_selection()

    with pytest.raises(ValueError, match="base_sha"):
        await make_coordinator(repository, selection).execute(
            make_review_input(),
            make_snapshot(),
            base_sha="not-a-git-sha",
        )

    assert provider.reviewed_inputs == ()
    with sqlite3.connect(repository.database_path) as connection:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM ai_operations",
        ).fetchone()[0]
    assert row_count == 0
