from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Event
from time import sleep

import pytest

import codeguard.ai.operation_repository as operation_repository_module
from codeguard.ai.operation_identity import AIOperationIdentity
from codeguard.ai.operation_repository import (
    AIOperationClaimStatus,
    AIOperationRepository,
    AIOperationStatus,
)
from codeguard.webhooks.schema_migrations import migrate_webhook_database


def make_identity(**updates: object) -> AIOperationIdentity:
    values: dict[str, object] = {
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
        "head_sha": "ABCDEF123456",
        "provider_identity": "provider-openai-compatible",
        "model_identity": "model-review-v1",
        "prompt_contract_version": "prompt-v1",
        "context_contract_version": "context-v1",
        "evidence_contract_version": "evidence-v1",
        "review_contract_version": "review-v1",
    }
    values.update(updates)
    return AIOperationIdentity(**values)  # type: ignore[arg-type]


@pytest.fixture
def repository(tmp_path: Path) -> tuple[AIOperationRepository, Path]:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)
    return AIOperationRepository(database_path), database_path


def get_claimed_operation(
    repository: AIOperationRepository,
) -> tuple[str, str]:
    created = repository.create_or_get(make_identity())
    claim = repository.claim(created.operation_key, 60)
    assert claim.status is AIOperationClaimStatus.CLAIMED
    assert claim.lease_token is not None
    return created.operation_key, claim.lease_token


def test_create_new_operation(repository: tuple[AIOperationRepository, Path]) -> None:
    repo, _ = repository

    record = repo.create_or_get(make_identity())

    assert record.status is AIOperationStatus.PENDING
    assert record.identity == make_identity()
    assert record.result_json is None


def test_retrieve_existing_operation(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    created = repo.create_or_get(make_identity())

    assert repo.get(created.operation_key) == created


def test_create_or_get_is_idempotent(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    first = repo.create_or_get(make_identity())
    second = repo.create_or_get(make_identity())

    assert second == first
    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM ai_operations").fetchone()[0]
    assert count == 1


def test_complete_identity_including_review_contract_is_persisted(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    identity = make_identity(review_contract_version="review-v7")
    record = repo.create_or_get(identity)

    assert repo.get(record.operation_key).identity == identity  # type: ignore[union-attr]
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT owner, repo, pull_number, head_sha, provider_identity,
                   model_identity, prompt_contract_version,
                   context_contract_version, evidence_contract_version,
                   review_contract_version
            FROM ai_operations WHERE operation_key = ?
            """,
            (record.operation_key,),
        ).fetchone()
    assert row == (
        identity.owner,
        identity.repo,
        identity.pull_number,
        identity.head_sha,
        identity.provider_identity,
        identity.model_identity,
        identity.prompt_contract_version,
        identity.context_contract_version,
        identity.evidence_contract_version,
        identity.review_contract_version,
    )


def test_mismatched_stored_identity_is_rejected(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    record = repo.create_or_get(make_identity())
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE ai_operations
            SET review_contract_version = ?
            WHERE operation_key = ?
            """,
            ("tampered-review-version", record.operation_key),
        )

    with pytest.raises(RuntimeError, match="identity"):
        repo.create_or_get(make_identity())


def test_get_returns_none_for_missing_operation(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository

    assert repo.get("ai-operation-v1:" + "0" * 64) is None


@pytest.mark.parametrize("operation_key", ["", "not-a-key", "ai-operation-v1:xyz"])
def test_invalid_operation_keys_are_rejected(
    repository: tuple[AIOperationRepository, Path],
    operation_key: str,
) -> None:
    repo, _ = repository

    with pytest.raises(ValueError, match="operation_key"):
        repo.get(operation_key)


def test_claim_from_pending_is_atomic(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    created = repo.create_or_get(make_identity())

    result = repo.claim(created.operation_key, 30)

    assert result.status is AIOperationClaimStatus.CLAIMED
    assert result.record.status is AIOperationStatus.CLAIMED
    assert result.lease_token
    assert result.record.lease_expires_at is not None
    with sqlite3.connect(database_path) as connection:
        persisted_token = connection.execute(
            "SELECT lease_owner FROM ai_operations WHERE operation_key = ?",
            (created.operation_key,),
        ).fetchone()[0]
    assert persisted_token == result.lease_token


def test_active_lease_prevents_another_claim(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    created = repo.create_or_get(make_identity())
    first = repo.claim(created.operation_key, 60)

    second = repo.claim(created.operation_key, 60)

    assert second.status is AIOperationClaimStatus.ALREADY_CLAIMED
    assert second.lease_token is None
    assert first.lease_token is not None
    assert not hasattr(second.record, "lease_owner")


def test_expired_claimed_operation_can_be_reclaimed(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    created = repo.create_or_get(make_identity())
    first = repo.claim(created.operation_key, 60)
    assert first.lease_token is not None
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET lease_expires_at = ? WHERE operation_key = ?",
            ("2000-01-01T00:00:00.000000+00:00", created.operation_key),
        )
    with pytest.raises(RuntimeError):
        repo.mark_request_started(created.operation_key, first.lease_token)

    reclaimed = repo.claim(created.operation_key, 60)

    assert reclaimed.status is AIOperationClaimStatus.CLAIMED
    assert reclaimed.lease_token is not None
    assert reclaimed.lease_token != first.lease_token


def test_reclaim_rotates_lease_token_and_invalidates_old_owner(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key, old_token = get_claimed_operation(repo)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET lease_expires_at = ? WHERE operation_key = ?",
            ("2000-01-01T00:00:00.000000+00:00", operation_key),
        )
    new_claim = repo.claim(operation_key, 60)

    assert new_claim.lease_token is not None
    assert new_claim.lease_token != old_token
    with pytest.raises(RuntimeError):
        repo.mark_request_started(operation_key, old_token)


def test_request_started_operation_cannot_be_reclaimed(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key, token = get_claimed_operation(repo)
    repo.mark_request_started(operation_key, token)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET lease_expires_at = ? WHERE operation_key = ?",
            ("2000-01-01T00:00:00.000000+00:00", operation_key),
        )

    result = repo.claim(operation_key, 60)

    assert result.status is AIOperationClaimStatus.REQUEST_STARTED
    assert result.record.status is AIOperationStatus.REQUEST_STARTED
    assert result.lease_token is None


def test_successful_result_persistence_is_atomic(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, token = get_claimed_operation(repo)
    repo.mark_request_started(operation_key, token)

    succeeded = repo.mark_succeeded(
        operation_key,
        token,
        result_json='{"opaque":true}',
        result_schema_version=3,
        serialization_version=2,
        provider_request_id="request-123",
    )

    assert succeeded.status is AIOperationStatus.SUCCEEDED
    assert succeeded.result_json == '{"opaque":true}'
    assert succeeded.result_schema_version == 3
    assert succeeded.serialization_version == 2
    assert succeeded.provider_request_id == "request-123"


def test_succeeded_operation_cannot_be_overwritten(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, token = get_claimed_operation(repo)
    repo.mark_request_started(operation_key, token)
    repo.mark_succeeded(
        operation_key,
        token,
        result_json='{"first":true}',
        result_schema_version=1,
        serialization_version=1,
    )

    with pytest.raises(RuntimeError):
        repo.mark_succeeded(
            operation_key,
            token,
            result_json='{"second":true}',
            result_schema_version=2,
            serialization_version=2,
        )
    assert repo.get(operation_key).result_json == '{"first":true}'  # type: ignore[union-attr]


def test_succeeded_operation_cannot_be_reclaimed(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, token = get_claimed_operation(repo)
    repo.mark_request_started(operation_key, token)
    repo.mark_succeeded(
        operation_key,
        token,
        result_json='{"complete":true}',
        result_schema_version=1,
        serialization_version=1,
    )

    result = repo.claim(operation_key, 60)

    assert result.status is AIOperationClaimStatus.SUCCEEDED
    assert result.record.status is AIOperationStatus.SUCCEEDED


def test_permanent_failure_transition(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, token = get_claimed_operation(repo)

    failed = repo.mark_failed_permanent(
        operation_key,
        token,
        disposition="permanent",
        safe_error="Provider rejected the request.",
    )

    assert failed.status is AIOperationStatus.FAILED_PERMANENT
    assert failed.disposition == "permanent"
    assert failed.safe_error == "Provider rejected the request."
    assert failed.result_json is None


def test_outcome_unknown_transition(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, token = get_claimed_operation(repo)

    unknown = repo.mark_outcome_unknown(
        operation_key,
        token,
        disposition="outcome_unknown",
        safe_error="The provider outcome could not be confirmed.",
    )

    assert unknown.status is AIOperationStatus.OUTCOME_UNKNOWN
    assert unknown.disposition == "outcome_unknown"
    assert unknown.safe_error is not None


@pytest.mark.parametrize(
    ("method_name", "terminal_status"),
    [
        ("mark_failed_permanent", AIOperationStatus.FAILED_PERMANENT),
        ("mark_outcome_unknown", AIOperationStatus.OUTCOME_UNKNOWN),
    ],
)
def test_terminal_states_cannot_be_reclaimed(
    repository: tuple[AIOperationRepository, Path],
    method_name: str,
    terminal_status: AIOperationStatus,
) -> None:
    repo, _ = repository
    operation_key, token = get_claimed_operation(repo)
    transition = getattr(repo, method_name)
    transition(
        operation_key,
        token,
        disposition="test",
        safe_error="safe",
    )

    result = repo.claim(operation_key, 60)

    assert result.record.status is terminal_status
    assert result.status in {
        AIOperationClaimStatus.FAILED_PERMANENT,
        AIOperationClaimStatus.OUTCOME_UNKNOWN,
    }


@pytest.mark.parametrize(
    "lease_duration",
    [0, -1, math.inf, -math.inf, math.nan, True, False, 10**1000],
)
def test_invalid_lease_durations_are_rejected(
    repository: tuple[AIOperationRepository, Path],
    lease_duration: float,
) -> None:
    repo, _ = repository
    operation_key = repo.create_or_get(make_identity()).operation_key

    with pytest.raises(ValueError, match="lease_duration_seconds"):
        repo.claim(operation_key, lease_duration)


def test_two_concurrent_claim_attempts_cannot_both_succeed(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key = repo.create_or_get(make_identity()).operation_key
    claim_barrier = Barrier(2)

    def claim() -> AIOperationClaimStatus:
        claim_barrier.wait(timeout=5)
        return repo.claim(operation_key, 60).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))

    assert results.count(AIOperationClaimStatus.CLAIMED) == 1
    assert results.count(AIOperationClaimStatus.ALREADY_CLAIMED) == 1


def test_claim_lease_starts_after_write_lock_is_acquired(
    repository: tuple[AIOperationRepository, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, database_path = repository
    operation_key = repo.create_or_get(make_identity()).operation_key
    external_connection = sqlite3.connect(database_path, timeout=5.0)
    claim_connection_opened = Event()
    original_connection = repo._connection
    lease_duration_seconds = 1.0
    executor = ThreadPoolExecutor(max_workers=1)

    @contextmanager
    def signal_claim_connection() -> Iterator[sqlite3.Connection]:
        with original_connection() as connection:
            claim_connection_opened.set()
            yield connection

    monkeypatch.setattr(repo, "_connection", signal_claim_connection)

    try:
        try:
            external_connection.execute("BEGIN IMMEDIATE")
            future = executor.submit(
                repo.claim,
                operation_key,
                lease_duration_seconds,
            )
            assert claim_connection_opened.wait(timeout=5)
            sleep(lease_duration_seconds + 0.25)
        finally:
            if external_connection.in_transaction:
                external_connection.rollback()
            external_connection.close()

        result = future.result(timeout=5)
        returned_at = datetime.now(UTC)
    finally:
        executor.shutdown(wait=True)

    assert result.status is AIOperationClaimStatus.CLAIMED
    assert result.record.lease_expires_at is not None
    assert datetime.fromisoformat(result.record.lease_expires_at) > returned_at


def test_failed_transition_rolls_back_database_update(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key, token = get_claimed_operation(repo)
    repo.mark_request_started(operation_key, token)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_success
            BEFORE UPDATE ON ai_operations
            WHEN NEW.status = 'SUCCEEDED'
            BEGIN
                SELECT RAISE(ABORT, 'trigger failure');
            END
            """
        )

    with pytest.raises(RuntimeError):
        repo.mark_succeeded(
            operation_key,
            token,
            result_json='{"should":"rollback"}',
            result_schema_version=1,
            serialization_version=1,
        )

    record = repo.get(operation_key)
    assert record.status is AIOperationStatus.REQUEST_STARTED  # type: ignore[union-attr]
    assert record.result_json is None  # type: ignore[union-attr]


def test_database_connections_are_closed(
    repository: tuple[AIOperationRepository, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = repository
    original_connect = sqlite3.connect
    connections: list[TrackingConnection] = []

    def tracked_connect(*args: object, **kwargs: object) -> TrackingConnection:
        kwargs["factory"] = TrackingConnection
        connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]
        connections.append(connection)
        return connection

    class TrackingConnection(sqlite3.Connection):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    monkeypatch.setattr(operation_repository_module.sqlite3, "connect", tracked_connect)
    repo.get(repo.create_or_get(make_identity()).operation_key)

    assert connections
    assert all(connection.closed for connection in connections)


def test_unknown_stored_status_is_rejected(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    record = repo.create_or_get(make_identity())
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE ai_operations SET status = ? WHERE operation_key = ?",
            ("INVALID", record.operation_key),
        )

    with pytest.raises(RuntimeError, match="status"):
        repo.get(record.operation_key)


def test_database_errors_do_not_leak_details(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE ai_operations")

    with pytest.raises(RuntimeError) as error:
        repo.get("ai-operation-v1:" + "0" * 64)

    assert "ai_operations" not in str(error.value)
