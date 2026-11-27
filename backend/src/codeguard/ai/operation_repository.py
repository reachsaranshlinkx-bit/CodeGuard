from __future__ import annotations

import math
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import cast

from codeguard.ai.exceptions import AIRequestTransmissionState
from codeguard.ai.operation_identity import (
    AI_OPERATION_V1_FORMAT_VERSION,
    AI_OPERATION_V2_FORMAT_VERSION,
    AIOperationIdentity,
    LegacyAIOperationIdentity,
    build_ai_operation_key,
    build_legacy_ai_operation_key,
)

_DATABASE_TIMEOUT_SECONDS = 5.0
_OPERATION_KEY_PATTERN = re.compile(
    rf"(?:{re.escape(AI_OPERATION_V1_FORMAT_VERSION)}|"
    rf"{re.escape(AI_OPERATION_V2_FORMAT_VERSION)}):[0-9a-f]{{64}}\Z"
)


class AIOperationRepositoryError(RuntimeError):
    """Raised when persisted AI operation data cannot be safely used."""


class AIOperationStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    REQUEST_STARTED = "REQUEST_STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED_PERMANENT = "FAILED_PERMANENT"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class AIOperationClaimStatus(StrEnum):
    CLAIMED = "CLAIMED"
    ALREADY_CLAIMED = "ALREADY_CLAIMED"
    REQUEST_STARTED = "REQUEST_STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED_PERMANENT = "FAILED_PERMANENT"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


@dataclass(frozen=True, slots=True)
class AIOperationRecord:
    """A typed snapshot of one persisted AI operation."""

    operation_key: str
    identity: AIOperationIdentity | LegacyAIOperationIdentity
    status: AIOperationStatus
    lease_expires_at: str | None
    provider_request_started_at: str | None
    result_schema_version: int | None
    serialization_version: int | None
    result_json: str | None
    provider_request_id: str | None
    disposition: str | None
    safe_error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AIOperationClaimResult:
    """Outcome of an atomic attempt to acquire an AI operation lease."""

    status: AIOperationClaimStatus
    record: AIOperationRecord
    lease_token: str | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _validate_operation_key(operation_key: str) -> None:
    if not isinstance(operation_key, str) or not _OPERATION_KEY_PATTERN.fullmatch(
        operation_key
    ):
        raise ValueError("operation_key has an invalid format.")


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _identity_values_from_row(
    row: sqlite3.Row,
) -> tuple[str, str, int, str, str, str, str, str, str, str]:
    pull_number = row["pull_number"]
    if isinstance(pull_number, bool) or not isinstance(pull_number, int):
        raise ValueError("Stored pull number is invalid.")
    return (
        _identity_string_from_row(row, "owner"),
        _identity_string_from_row(row, "repo"),
        pull_number,
        _identity_string_from_row(row, "head_sha"),
        _identity_string_from_row(row, "provider_identity"),
        _identity_string_from_row(row, "model_identity"),
        _identity_string_from_row(row, "prompt_contract_version"),
        _identity_string_from_row(row, "context_contract_version"),
        _identity_string_from_row(row, "evidence_contract_version"),
        _identity_string_from_row(row, "review_contract_version"),
    )


def _identity_string_from_row(row: sqlite3.Row, field_name: str) -> str:
    value = row[field_name]
    if not isinstance(value, str):
        raise ValueError("Stored identity string is invalid.")
    return value


class AIOperationRepository:
    """Durable SQLite repository for AI review operations."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)

    @property
    def database_path(self) -> Path:
        """Return the shared webhook database path."""

        return self._database_path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(
                str(self._database_path),
                timeout=_DATABASE_TIMEOUT_SECONDS,
            )
        except sqlite3.Error:
            raise AIOperationRepositoryError(
                "AI operation database access failed."
            ) from None

        connection.row_factory = sqlite3.Row
        try:
            connection.execute(
                f"PRAGMA busy_timeout = {int(_DATABASE_TIMEOUT_SECONDS * 1000)}"
            )
            yield connection
            connection.commit()
        except sqlite3.Error:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise AIOperationRepositoryError(
                "AI operation database operation failed."
            ) from None
        except Exception:
            try:
                connection.rollback()
            except sqlite3.Error:
                raise AIOperationRepositoryError(
                    "AI operation database rollback failed."
                ) from None
            raise
        finally:
            try:
                connection.close()
            except sqlite3.Error:
                raise AIOperationRepositoryError(
                    "AI operation database connection close failed."
                ) from None

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> AIOperationRecord:
        operation_key = row["operation_key"]
        if not isinstance(operation_key, str):
            raise AIOperationRepositoryError("Stored AI operation key is invalid.")
        try:
            _validate_operation_key(operation_key)
        except ValueError:
            raise AIOperationRepositoryError(
                "Stored AI operation key is invalid."
            ) from None

        key_version = operation_key.partition(":")[0]
        fingerprint = row["input_fingerprint"]

        if key_version == AI_OPERATION_V1_FORMAT_VERSION:
            if fingerprint is not None:
                raise AIOperationRepositoryError(
                    "Legacy AI operation has an unexpected fingerprint."
                )
        else:
            if fingerprint is None:
                raise AIOperationRepositoryError(
                    "Version 2 AI operation is missing its input fingerprint."
                )
            if not isinstance(fingerprint, str):
                raise AIOperationRepositoryError(
                    "Stored AI operation fingerprint is invalid."
                )

        status_value = row["status"]
        if not isinstance(status_value, str):
            raise AIOperationRepositoryError("Stored AI operation status is invalid.")
        identity: AIOperationIdentity | LegacyAIOperationIdentity
        try:
            status = AIOperationStatus(status_value)
        except ValueError:
            raise AIOperationRepositoryError(
                "Stored AI operation status is invalid."
            ) from None

        try:
            (
                owner,
                repo,
                pull_number,
                head_sha,
                provider_identity,
                model_identity,
                prompt_contract_version,
                context_contract_version,
                evidence_contract_version,
                review_contract_version,
            ) = _identity_values_from_row(row)
            if key_version == AI_OPERATION_V2_FORMAT_VERSION:
                if not isinstance(fingerprint, str):
                    raise ValueError("Stored AI operation fingerprint is invalid.")
                identity = AIOperationIdentity(
                    owner=owner,
                    repo=repo,
                    pull_number=pull_number,
                    head_sha=head_sha,
                    provider_identity=provider_identity,
                    model_identity=model_identity,
                    prompt_contract_version=prompt_contract_version,
                    context_contract_version=context_contract_version,
                    evidence_contract_version=evidence_contract_version,
                    review_contract_version=review_contract_version,
                    input_fingerprint=fingerprint,
                )
                expected_key = build_ai_operation_key(identity)
            else:
                identity = LegacyAIOperationIdentity(
                    owner=owner,
                    repo=repo,
                    pull_number=pull_number,
                    head_sha=head_sha,
                    provider_identity=provider_identity,
                    model_identity=model_identity,
                    prompt_contract_version=prompt_contract_version,
                    context_contract_version=context_contract_version,
                    evidence_contract_version=evidence_contract_version,
                    review_contract_version=review_contract_version,
                )
                expected_key = build_legacy_ai_operation_key(identity)
        except (TypeError, ValueError, UnicodeError, OverflowError):
            raise AIOperationRepositoryError(
                "Stored AI operation identity is invalid."
            ) from None

        if expected_key != operation_key:
            raise AIOperationRepositoryError(
                "Stored AI operation key does not match its identity."
            )

        result_json = row["result_json"]
        if result_json is not None and not isinstance(result_json, str):
            raise AIOperationRepositoryError("Stored AI operation result is invalid.")

        return AIOperationRecord(
            operation_key=operation_key,
            identity=identity,
            status=status,
            lease_expires_at=_optional_string(row["lease_expires_at"]),
            provider_request_started_at=_optional_string(
                row["provider_request_started_at"]
            ),
            result_schema_version=(
                int(row["result_schema_version"])
                if row["result_schema_version"] is not None
                else None
            ),
            serialization_version=(
                int(row["serialization_version"])
                if row["serialization_version"] is not None
                else None
            ),
            result_json=result_json,
            provider_request_id=_optional_string(row["provider_request_id"]),
            disposition=_optional_string(row["disposition"]),
            safe_error=_optional_string(row["safe_error"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _identity_matches(
        record: AIOperationRecord,
        identity: AIOperationIdentity,
    ) -> bool:
        return (
            isinstance(record.identity, AIOperationIdentity)
            and record.identity == identity
        )

    def create_or_get(self, identity: AIOperationIdentity) -> AIOperationRecord:
        """Create a pending operation or return its existing matching record."""

        if not isinstance(identity, AIOperationIdentity):
            raise TypeError("New AI operations require an AIOperationIdentity v2.")

        operation_key = build_ai_operation_key(identity)
        now_text = _timestamp(_utc_now())

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO ai_operations (
                    operation_key, owner, repo, pull_number, head_sha,
                    provider_identity, model_identity, prompt_contract_version,
                    context_contract_version, evidence_contract_version,
                    review_contract_version, input_fingerprint, status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_key) DO NOTHING
                """,
                (
                    operation_key,
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
                    AIOperationStatus.PENDING.value,
                    now_text,
                    now_text,
                ),
            )
            row = connection.execute(
                "SELECT * FROM ai_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise AIOperationRepositoryError(
                    "AI operation could not be retrieved after creation."
                )
            record = self._record_from_row(row)
            if not self._identity_matches(record, identity):
                raise AIOperationRepositoryError(
                    "Stored AI operation identity does not match "
                    "the requested identity."
                )
            return record

    def get(self, operation_key: str) -> AIOperationRecord | None:
        """Retrieve an operation by its canonical key."""

        _validate_operation_key(operation_key)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM ai_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            return self._record_from_row(row) if row is not None else None

    def claim(
        self,
        operation_key: str,
        lease_duration_seconds: float,
    ) -> AIOperationClaimResult:
        """Atomically claim a pending or safely reclaimable operation."""

        _validate_operation_key(operation_key)
        if isinstance(lease_duration_seconds, bool) or not isinstance(
            lease_duration_seconds, Real
        ):
            raise ValueError("lease_duration_seconds must be finite and positive.")
        try:
            is_finite = math.isfinite(lease_duration_seconds)
        except OverflowError:
            is_finite = False
        if not is_finite or lease_duration_seconds <= 0:
            raise ValueError("lease_duration_seconds must be finite and positive.")

        try:
            lease_duration = timedelta(seconds=lease_duration_seconds)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(
                "lease_duration_seconds is outside the supported range."
            ) from None

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _utc_now()
            try:
                expiry = now + lease_duration
            except OverflowError:
                raise ValueError(
                    "lease_duration_seconds is outside the supported range."
                ) from None
            now_text = _timestamp(now)
            expiry_text = _timestamp(expiry)
            lease_token = uuid.uuid4().hex

            row = connection.execute(
                "SELECT * FROM ai_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise AIOperationRepositoryError("AI operation does not exist.")
            record = self._record_from_row(row)
            if not isinstance(record.identity, AIOperationIdentity):
                raise AIOperationRepositoryError(
                    "Legacy AI operations cannot be claimed."
                )

            if record.status is AIOperationStatus.PENDING:
                can_claim = True
            elif record.status is AIOperationStatus.CLAIMED:
                if record.lease_expires_at is None:
                    raise AIOperationRepositoryError(
                        "Stored AI operation lease is invalid."
                    )
                try:
                    lease_expiry = datetime.fromisoformat(record.lease_expires_at)
                except ValueError:
                    raise AIOperationRepositoryError(
                        "Stored AI operation lease is invalid."
                    ) from None
                if lease_expiry.tzinfo is None:
                    raise AIOperationRepositoryError(
                        "Stored AI operation lease is invalid."
                    )
                can_claim = lease_expiry <= now
            else:
                can_claim = False

            if can_claim:
                cursor = connection.execute(
                    """
                    UPDATE ai_operations
                    SET status = ?, lease_owner = ?, lease_expires_at = ?,
                        updated_at = ?
                    WHERE operation_key = ?
                      AND status IN (?, ?)
                      AND (
                          status = ?
                          OR (status = ? AND lease_expires_at <= ?)
                      )
                    """,
                    (
                        AIOperationStatus.CLAIMED.value,
                        lease_token,
                        expiry_text,
                        now_text,
                        operation_key,
                        AIOperationStatus.PENDING.value,
                        AIOperationStatus.CLAIMED.value,
                        AIOperationStatus.PENDING.value,
                        AIOperationStatus.CLAIMED.value,
                        now_text,
                    ),
                )
                if cursor.rowcount != 1:
                    raise AIOperationRepositoryError(
                        "AI operation claim could not be acquired."
                    )
                claimed_row = connection.execute(
                    "SELECT * FROM ai_operations WHERE operation_key = ?",
                    (operation_key,),
                ).fetchone()
                if claimed_row is None:
                    raise AIOperationRepositoryError(
                        "AI operation could not be retrieved after claiming."
                    )
                claimed_record = self._record_from_row(claimed_row)
                return AIOperationClaimResult(
                    status=AIOperationClaimStatus.CLAIMED,
                    record=claimed_record,
                    lease_token=lease_token,
                )

            claim_status = {
                AIOperationStatus.CLAIMED: AIOperationClaimStatus.ALREADY_CLAIMED,
                AIOperationStatus.REQUEST_STARTED: (
                    AIOperationClaimStatus.REQUEST_STARTED
                ),
                AIOperationStatus.SUCCEEDED: AIOperationClaimStatus.SUCCEEDED,
                AIOperationStatus.FAILED_PERMANENT: (
                    AIOperationClaimStatus.FAILED_PERMANENT
                ),
                AIOperationStatus.OUTCOME_UNKNOWN: (
                    AIOperationClaimStatus.OUTCOME_UNKNOWN
                ),
            }.get(record.status)
            if claim_status is None:
                raise AIOperationRepositoryError(
                    "AI operation cannot be claimed in its current state."
                )
            return AIOperationClaimResult(
                status=claim_status,
                record=record,
                lease_token=None,
            )

    def mark_request_started(
        self,
        operation_key: str,
        lease_token: str,
    ) -> AIOperationRecord:
        """Record provider request start for the current lease owner."""

        return self._transition(
            operation_key,
            lease_token,
            expected_statuses=(AIOperationStatus.CLAIMED,),
            new_status=AIOperationStatus.REQUEST_STARTED,
            assignments="",
            values=(),
            timestamp_assignment="provider_request_started_at = ?",
        )

    def mark_succeeded(
        self,
        operation_key: str,
        lease_token: str,
        *,
        result_json: str,
        result_schema_version: int,
        serialization_version: int,
        provider_request_id: str | None = None,
    ) -> AIOperationRecord:
        """Persist the opaque result and success state atomically."""

        self._validate_version(result_schema_version, "result_schema_version")
        self._validate_version(serialization_version, "serialization_version")
        if not isinstance(result_json, str):
            raise ValueError("result_json must be a string.")

        return self._transition(
            operation_key,
            lease_token,
            expected_statuses=(AIOperationStatus.REQUEST_STARTED,),
            new_status=AIOperationStatus.SUCCEEDED,
            assignments=(
                "result_json = ?, result_schema_version = ?, "
                "serialization_version = ?, provider_request_id = ?, "
                "lease_owner = NULL, lease_expires_at = NULL"
            ),
            values=(
                result_json,
                result_schema_version,
                serialization_version,
                provider_request_id,
            ),
        )

    def mark_safe_to_retry(
        self,
        operation_key: str,
        lease_token: str,
        *,
        transmission_state: AIRequestTransmissionState,
    ) -> AIOperationRecord:
        """Return to PENDING only when the provider request was not sent."""

        if transmission_state is not AIRequestTransmissionState.NOT_SENT:
            raise ValueError(
                "Safe retry requires an independently confirmed NOT_SENT state."
            )

        return self._transition(
            operation_key,
            lease_token,
            expected_statuses=(AIOperationStatus.REQUEST_STARTED,),
            new_status=AIOperationStatus.PENDING,
            assignments=(
                "lease_owner = NULL, lease_expires_at = NULL, "
                "provider_request_started_at = NULL"
            ),
            values=(),
        )

    def mark_failed_permanent(
        self,
        operation_key: str,
        lease_token: str,
        *,
        disposition: str,
        safe_error: str,
    ) -> AIOperationRecord:
        """Mark an owned active operation as permanently failed."""

        return self._mark_terminal_failure(
            operation_key,
            lease_token,
            AIOperationStatus.FAILED_PERMANENT,
            disposition,
            safe_error,
        )

    def mark_outcome_unknown(
        self,
        operation_key: str,
        lease_token: str,
        *,
        disposition: str,
        safe_error: str,
    ) -> AIOperationRecord:
        """Mark an owned active operation whose provider outcome is unknown."""

        return self._mark_terminal_failure(
            operation_key,
            lease_token,
            AIOperationStatus.OUTCOME_UNKNOWN,
            disposition,
            safe_error,
        )

    def _mark_terminal_failure(
        self,
        operation_key: str,
        lease_token: str,
        status: AIOperationStatus,
        disposition: str,
        safe_error: str,
    ) -> AIOperationRecord:
        if not isinstance(disposition, str) or not disposition.strip():
            raise ValueError("disposition must be a non-empty string.")
        if not isinstance(safe_error, str) or not safe_error.strip():
            raise ValueError("safe_error must be a non-empty string.")

        return self._transition(
            operation_key,
            lease_token,
            expected_statuses=(
                AIOperationStatus.CLAIMED,
                AIOperationStatus.REQUEST_STARTED,
            ),
            new_status=status,
            assignments=(
                "disposition = ?, safe_error = ?, "
                "lease_owner = NULL, lease_expires_at = NULL"
            ),
            values=(disposition.strip()[:100], safe_error.strip()[:1000]),
        )

    @staticmethod
    def _validate_version(value: int, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer.")

    def _transition(
        self,
        operation_key: str,
        lease_token: str,
        *,
        expected_statuses: tuple[AIOperationStatus, ...],
        new_status: AIOperationStatus,
        assignments: str,
        values: tuple[object, ...],
        timestamp_assignment: str | None = None,
    ) -> AIOperationRecord:
        _validate_operation_key(operation_key)
        if not isinstance(lease_token, str) or not lease_token:
            raise ValueError("lease_token must be a non-empty string.")

        set_assignments = assignments
        if timestamp_assignment is not None:
            set_assignments = (
                f"{timestamp_assignment}, {assignments}"
                if assignments
                else timestamp_assignment
            )
        expected_sql = ", ".join("?" for _ in expected_statuses)
        statement = (
            f"UPDATE ai_operations SET {set_assignments}, status = ?, updated_at = ? "
            f"WHERE operation_key = ? AND lease_owner = ? "
            f"AND status IN ({expected_sql}) "
            "AND (status != ? OR lease_expires_at > ?)"
        )

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                "SELECT * FROM ai_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if existing_row is not None:
                existing_record = self._record_from_row(existing_row)
                if not isinstance(existing_record.identity, AIOperationIdentity):
                    raise AIOperationRepositoryError(
                        "Legacy AI operations cannot be transitioned."
                    )

            now_text = _timestamp(_utc_now())
            parameters = (
                *((now_text,) if timestamp_assignment is not None else ()),
                *values,
                new_status.value,
                now_text,
                operation_key,
                lease_token,
                *(status.value for status in expected_statuses),
                AIOperationStatus.CLAIMED.value,
                now_text,
            )
            cursor = connection.execute(statement, parameters)
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "AI operation transition rejected: ownership or state changed."
                )
            row = connection.execute(
                "SELECT * FROM ai_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise AIOperationRepositoryError(
                    "AI operation could not be retrieved after transition."
                )
            return self._record_from_row(cast(sqlite3.Row, row))
