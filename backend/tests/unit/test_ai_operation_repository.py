from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Event, current_thread
from time import sleep

import pytest

import codeguard.ai.operation_repository as operation_repository_module
from codeguard.ai.exceptions import AIRequestTransmissionState
from codeguard.ai.operation_identity import (
    AIOperationIdentity,
    LegacyAIOperationIdentity,
    build_legacy_ai_operation_key,
)
from codeguard.ai.operation_repository import (
    AIOperationClaimStatus,
    AIOperationRepository,
    AIOperationRepositoryError,
    AIOperationStatus,
)
from codeguard.webhooks.schema_migrations import migrate_webhook_database

TEST_FINGERPRINT = "ai-execution-input-v1:" + "f" * 64


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
        "input_fingerprint": TEST_FINGERPRINT,
    }
    values.update(updates)
    return AIOperationIdentity(**values)  # type: ignore[arg-type]


def make_legacy_identity(**updates: object) -> LegacyAIOperationIdentity:
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
    return LegacyAIOperationIdentity(**values)  # type: ignore[arg-type]


@pytest.fixture
def repository(tmp_path: Path) -> tuple[AIOperationRepository, Path]:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)
    return AIOperationRepository(database_path), database_path


def insert_legacy_operation(
    database_path: Path,
    *,
    identity: LegacyAIOperationIdentity | None = None,
    status: str = "SUCCEEDED",
    result_json: str | None = '{"legacy":"opaque"}',
    input_fingerprint: str | None = None,
) -> str:
    legacy_identity = identity or make_legacy_identity()
    operation_key = build_legacy_ai_operation_key(legacy_identity)
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                input_fingerprint,
                status,
                result_json,
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
            ),
        )
    return operation_key


def get_claimed_operation(
    repository: AIOperationRepository,
) -> tuple[str, str]:
    created = repository.create_or_get(make_identity())
    claim = repository.claim(created.operation_key, 60)
    assert claim.status is AIOperationClaimStatus.CLAIMED
    assert claim.lease_token is not None
    return created.operation_key, claim.lease_token


def get_request_started_operation(
    repository: AIOperationRepository,
) -> tuple[str, str]:
    operation_key, lease_token = get_claimed_operation(repository)
    repository.mark_request_started(operation_key, lease_token)
    return operation_key, lease_token


def test_create_new_operation(repository: tuple[AIOperationRepository, Path]) -> None:
    repo, _ = repository

    record = repo.create_or_get(make_identity())

    assert record.status is AIOperationStatus.PENDING
    assert record.identity == make_identity()
    assert record.identity.input_fingerprint == TEST_FINGERPRINT
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
                   review_contract_version, input_fingerprint
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
        identity.input_fingerprint,
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


def test_create_or_get_rejects_legacy_identity(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository

    with pytest.raises(TypeError, match="v2"):
        repo.create_or_get(make_legacy_identity())  # type: ignore[arg-type]


def test_valid_legacy_successful_row_remains_readable_and_opaque(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    legacy_identity = make_legacy_identity()
    opaque_result = '{"legacy":"opaque"}'
    operation_key = insert_legacy_operation(
        database_path,
        identity=legacy_identity,
        result_json=opaque_result,
    )

    record = repo.get(operation_key)

    assert record is not None
    assert type(record.identity) is LegacyAIOperationIdentity
    assert record.identity == legacy_identity
    assert record.operation_key == operation_key
    assert record.status is AIOperationStatus.SUCCEEDED
    assert record.result_json == opaque_result
    assert record.created_at == "2026-01-01T00:00:00+00:00"
    assert record.updated_at == "2026-01-02T00:00:00+00:00"


def test_matching_v2_operation_is_separate_and_does_not_reuse_legacy_result(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    legacy_key = insert_legacy_operation(database_path)

    v2_identity = make_identity()
    new_record = repo.create_or_get(v2_identity)

    assert new_record.operation_key != legacy_key
    assert new_record.identity == v2_identity
    assert new_record.status is AIOperationStatus.PENDING
    assert new_record.result_json is None
    assert repo.get(legacy_key).result_json == '{"legacy":"opaque"}'  # type: ignore[union-attr]

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT operation_key, input_fingerprint, result_json
            FROM ai_operations ORDER BY operation_key
            """
        ).fetchall()

    assert len(rows) == 2
    v2_row = next(row for row in rows if row[0] == new_record.operation_key)
    assert v2_row == (
        new_record.operation_key,
        v2_identity.input_fingerprint,
        None,
    )


def test_legacy_pending_operation_cannot_be_claimed_or_transitioned(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key = insert_legacy_operation(
        database_path,
        status=AIOperationStatus.PENDING.value,
        result_json=None,
    )

    with pytest.raises(AIOperationRepositoryError, match="Legacy"):
        repo.claim(operation_key, 60)

    with pytest.raises(AIOperationRepositoryError, match="Legacy"):
        repo.mark_request_started(operation_key, "historical-lease")


def test_legacy_key_with_fingerprint_is_rejected(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key = insert_legacy_operation(
        database_path,
        input_fingerprint=TEST_FINGERPRINT,
    )

    with pytest.raises(AIOperationRepositoryError, match="unexpected fingerprint"):
        repo.get(operation_key)


def test_v2_key_without_fingerprint_is_rejected(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    record = repo.create_or_get(make_identity())
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET input_fingerprint = NULL WHERE operation_key = ?",
            (record.operation_key,),
        )

    with pytest.raises(
        AIOperationRepositoryError, match="missing its input fingerprint"
    ):
        repo.get(record.operation_key)


@pytest.mark.parametrize(
    "fingerprint",
    ["", "invalid", "ai-execution-input-v1:" + "F" * 64],
)
def test_v2_key_with_malformed_fingerprint_is_rejected(
    repository: tuple[AIOperationRepository, Path],
    fingerprint: str,
) -> None:
    repo, database_path = repository
    record = repo.create_or_get(make_identity())
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET input_fingerprint = ? WHERE operation_key = ?",
            (fingerprint, record.operation_key),
        )

    with pytest.raises(AIOperationRepositoryError, match="identity is invalid"):
        repo.get(record.operation_key)


def test_identity_values_that_do_not_reproduce_key_are_rejected(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    record = repo.create_or_get(make_identity())
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE ai_operations SET provider_identity = ? WHERE operation_key = ?",
            ("different-provider", record.operation_key),
        )

    with pytest.raises(AIOperationRepositoryError, match="does not match"):
        repo.get(record.operation_key)


def test_get_returns_none_for_missing_operation(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository

    assert repo.get("ai-operation-v2:" + "0" * 64) is None


@pytest.mark.parametrize(
    "operation_key",
    ["", "not-a-key", "ai-operation-v1:xyz", "ai-operation-v2:xyz"],
)
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


def test_transition_rechecks_claimed_lease_after_write_lock_wait(
    repository: tuple[AIOperationRepository, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, database_path = repository
    created = repo.create_or_get(make_identity())
    claim = repo.claim(created.operation_key, 1)
    assert claim.lease_token is not None

    external_connection = sqlite3.connect(database_path, timeout=5.0)
    transition_connection_opened = Event()
    original_connection = repo._connection
    executor = ThreadPoolExecutor(max_workers=1)

    @contextmanager
    def signal_transition_connection() -> Iterator[sqlite3.Connection]:
        with original_connection() as connection:
            transition_connection_opened.set()
            yield connection

    monkeypatch.setattr(repo, "_connection", signal_transition_connection)

    try:
        try:
            external_connection.execute("BEGIN IMMEDIATE")
            future = executor.submit(
                repo.mark_request_started,
                created.operation_key,
                claim.lease_token,
            )
            assert transition_connection_opened.wait(timeout=5)
            sleep(1.25)
        finally:
            try:
                if external_connection.in_transaction:
                    external_connection.rollback()
            finally:
                external_connection.close()

        with pytest.raises(RuntimeError, match="transition rejected"):
            future.result(timeout=5)
    finally:
        executor.shutdown(wait=True)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT status, provider_request_started_at
            FROM ai_operations WHERE operation_key = ?
            """,
            (created.operation_key,),
        ).fetchone()
    assert row == (AIOperationStatus.CLAIMED.value, None)


def test_safe_retry_resets_request_started_operation_and_lease_fields(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key, lease_token = get_request_started_operation(repo)

    record = repo.mark_safe_to_retry(
        operation_key,
        lease_token,
        transmission_state=AIRequestTransmissionState.NOT_SENT,
    )

    assert record.status is AIOperationStatus.PENDING
    assert record.lease_expires_at is None
    assert record.provider_request_started_at is None
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT lease_owner, lease_expires_at, provider_request_started_at
            FROM ai_operations WHERE operation_key = ?
            """,
            (operation_key,),
        ).fetchone()
    assert row == (None, None, None)


def test_safe_retry_is_allowed_after_request_started_lease_expires(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, database_path = repository
    operation_key = repo.create_or_get(make_identity()).operation_key
    lease_duration_seconds = 1.0
    initial_claim = repo.claim(operation_key, lease_duration_seconds)
    assert initial_claim.status is AIOperationClaimStatus.CLAIMED
    assert initial_claim.lease_token is not None
    original_lease_token = initial_claim.lease_token
    started = repo.mark_request_started(operation_key, original_lease_token)
    assert started.status is AIOperationStatus.REQUEST_STARTED

    sleep(lease_duration_seconds + 0.1)

    after_expiry_claim = repo.claim(operation_key, 60)
    assert after_expiry_claim.status is AIOperationClaimStatus.REQUEST_STARTED
    assert after_expiry_claim.record.status is AIOperationStatus.REQUEST_STARTED
    assert after_expiry_claim.lease_token is None

    reset = repo.mark_safe_to_retry(
        operation_key,
        original_lease_token,
        transmission_state=AIRequestTransmissionState.NOT_SENT,
    )

    assert reset.status is AIOperationStatus.PENDING
    assert reset.lease_expires_at is None
    assert reset.provider_request_started_at is None
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT lease_owner, lease_expires_at, provider_request_started_at
            FROM ai_operations WHERE operation_key = ?
            """,
            (operation_key,),
        ).fetchone()
    assert row == (None, None, None)


def test_safe_retry_requires_a_valid_lease_owner_token(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, _ = get_request_started_operation(repo)

    with pytest.raises(RuntimeError, match="transition rejected"):
        repo.mark_safe_to_retry(
            operation_key,
            "wrong-token",
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )

    with pytest.raises(ValueError, match="lease_token"):
        repo.mark_safe_to_retry(
            operation_key,
            None,  # type: ignore[arg-type]
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )

    with pytest.raises(TypeError):
        repo.mark_safe_to_retry(  # type: ignore[call-arg]
            operation_key,
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )


def test_safe_retry_rejects_stale_lease_owner_token(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, stale_token = get_request_started_operation(repo)
    repo.mark_safe_to_retry(
        operation_key,
        stale_token,
        transmission_state=AIRequestTransmissionState.NOT_SENT,
    )
    new_claim = repo.claim(operation_key, 60)
    assert new_claim.lease_token is not None
    repo.mark_request_started(operation_key, new_claim.lease_token)

    with pytest.raises(RuntimeError, match="transition rejected"):
        repo.mark_safe_to_retry(
            operation_key,
            stale_token,
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )


def test_safe_retry_rejects_unconfirmed_transmission_state(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, lease_token = get_request_started_operation(repo)

    with pytest.raises(ValueError, match="NOT_SENT"):
        repo.mark_safe_to_retry(
            operation_key,
            lease_token,
            transmission_state=AIRequestTransmissionState.UNKNOWN,
        )

    assert repo.get(operation_key).status is AIOperationStatus.REQUEST_STARTED  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "source_status",
    [AIOperationStatus.PENDING, AIOperationStatus.CLAIMED],
)
def test_safe_retry_rejects_non_started_source_states(
    repository: tuple[AIOperationRepository, Path],
    source_status: AIOperationStatus,
) -> None:
    repo, _ = repository
    record = repo.create_or_get(make_identity())
    lease_token = "not-current"
    if source_status is AIOperationStatus.CLAIMED:
        claim = repo.claim(record.operation_key, 60)
        assert claim.lease_token is not None
        lease_token = claim.lease_token

    with pytest.raises(RuntimeError, match="transition rejected"):
        repo.mark_safe_to_retry(
            record.operation_key,
            lease_token,
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )


@pytest.mark.parametrize(
    "terminal_status",
    [
        AIOperationStatus.SUCCEEDED,
        AIOperationStatus.FAILED_PERMANENT,
        AIOperationStatus.OUTCOME_UNKNOWN,
    ],
)
def test_safe_retry_cannot_reset_terminal_operations(
    repository: tuple[AIOperationRepository, Path],
    terminal_status: AIOperationStatus,
) -> None:
    repo, _ = repository
    operation_key, lease_token = get_claimed_operation(repo)

    if terminal_status is AIOperationStatus.SUCCEEDED:
        repo.mark_request_started(operation_key, lease_token)
        repo.mark_succeeded(
            operation_key,
            lease_token,
            result_json='{"complete":true}',
            result_schema_version=1,
            serialization_version=1,
        )
    elif terminal_status is AIOperationStatus.FAILED_PERMANENT:
        repo.mark_failed_permanent(
            operation_key,
            lease_token,
            disposition="permanent",
            safe_error="safe",
        )
    else:
        repo.mark_outcome_unknown(
            operation_key,
            lease_token,
            disposition="unknown",
            safe_error="safe",
        )

    with pytest.raises(RuntimeError, match="transition rejected"):
        repo.mark_safe_to_retry(
            operation_key,
            lease_token,
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )

    assert repo.get(operation_key).status is terminal_status  # type: ignore[union-attr]


def test_safe_retry_operation_can_be_claimed_again(
    repository: tuple[AIOperationRepository, Path],
) -> None:
    repo, _ = repository
    operation_key, old_token = get_request_started_operation(repo)
    repo.mark_safe_to_retry(
        operation_key,
        old_token,
        transmission_state=AIRequestTransmissionState.NOT_SENT,
    )

    claim = repo.claim(operation_key, 60)

    assert claim.status is AIOperationClaimStatus.CLAIMED
    assert claim.lease_token is not None
    assert claim.lease_token != old_token


def test_claim_waits_for_safe_retry_transition_commit(
    repository: tuple[AIOperationRepository, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = repository
    operation_key, lease_token = get_request_started_operation(repo)
    transition_commit_started = Event()
    allow_transition_commit = Event()
    claim_begin_started = Event()
    original_connect = sqlite3.connect

    class CoordinatedConnection(sqlite3.Connection):
        def execute(
            self,
            sql: str,
            parameters: tuple[object, ...] = (),
        ) -> sqlite3.Cursor:
            if (
                sql == "BEGIN IMMEDIATE"
                and current_thread().name == "safe-retry-claim_1"
            ):
                claim_begin_started.set()
            return super().execute(sql, parameters)

        def commit(self) -> None:
            if current_thread().name == "safe-retry-claim_0":
                transition_commit_started.set()
                if not allow_transition_commit.wait(timeout=5):
                    raise TimeoutError("Timed out waiting to commit safe retry.")
            super().commit()

    def coordinated_connect(
        *args: object,
        **kwargs: object,
    ) -> CoordinatedConnection:
        kwargs["factory"] = CoordinatedConnection
        connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]
        return connection

    monkeypatch.setattr(
        operation_repository_module.sqlite3,
        "connect",
        coordinated_connect,
    )

    executor = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="safe-retry-claim",
    )
    try:
        transition_future = executor.submit(
            repo.mark_safe_to_retry,
            operation_key,
            lease_token,
            transmission_state=AIRequestTransmissionState.NOT_SENT,
        )
        assert transition_commit_started.wait(timeout=5)

        claim_future = executor.submit(repo.claim, operation_key, 60)
        assert claim_begin_started.wait(timeout=5)
        assert not claim_future.done()

        allow_transition_commit.set()
        transitioned = transition_future.result(timeout=5)
        claim = claim_future.result(timeout=5)
    finally:
        allow_transition_commit.set()
        executor.shutdown(wait=True)

    assert transitioned.status is AIOperationStatus.PENDING
    assert claim.status is AIOperationClaimStatus.CLAIMED


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
