from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

JobStatus = Literal["queued", "processing", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class WebhookJob:
    """Persisted GitHub pull request review job."""

    delivery_id: str
    event: str
    action: str
    owner: str
    repo: str
    pull_number: int
    status: JobStatus
    attempts: int
    created_at: str
    updated_at: str
    available_at: str
    lease_expires_at: str | None
    last_error: str | None


class WebhookJobQueue:
    """SQLite storage for durable webhook review jobs."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection with transaction handling."""

        connection = sqlite3.connect(
            str(self._database_path),
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create durable job storage and its lookup index."""

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_jobs (
                    delivery_id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    action TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    pull_number INTEGER NOT NULL
                        CHECK(pull_number >= 1),
                    status TEXT NOT NULL
                        CHECK(status IN (
                            'queued',
                            'processing',
                            'completed',
                            'failed'
                        )),
                    attempts INTEGER NOT NULL DEFAULT 0
                        CHECK(attempts >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    lease_expires_at TEXT,
                    last_error TEXT
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_jobs_ready
                ON webhook_jobs(
                    status,
                    available_at,
                    lease_expires_at,
                    created_at
                )
                """
            )

    @property
    def database_path(self) -> Path:
        """Return the configured SQLite database path."""

        return self._database_path

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> WebhookJob:
        """Convert a SQLite row into a typed webhook job."""

        return WebhookJob(
            delivery_id=str(row["delivery_id"]),
            event=str(row["event"]),
            action=str(row["action"]),
            owner=str(row["owner"]),
            repo=str(row["repo"]),
            pull_number=int(row["pull_number"]),
            status=cast(JobStatus, str(row["status"])),
            attempts=int(row["attempts"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            available_at=str(row["available_at"]),
            lease_expires_at=(
                str(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            last_error=(
                str(row["last_error"]) if row["last_error"] is not None else None
            ),
        )

    def enqueue_job(
        self,
        *,
        delivery_id: str,
        event: str,
        action: str,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> WebhookJob:
        """Persist a new job without duplicating existing deliveries."""

        if not delivery_id.strip():
            raise ValueError("delivery_id must not be empty.")

        if not event.strip() or not action.strip():
            raise ValueError("Event and action must not be empty.")

        if not owner.strip() or not repo.strip():
            raise ValueError("Repository owner and name must not be empty.")

        if (
            isinstance(pull_number, bool)
            or not isinstance(pull_number, int)
            or pull_number < 1
        ):
            raise ValueError("pull_number must be a positive integer.")

        now_text = datetime.now(UTC).isoformat()

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                """
                SELECT *
                FROM webhook_jobs
                WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()

            if existing is not None:
                existing_data = (
                    existing["event"],
                    existing["action"],
                    existing["owner"],
                    existing["repo"],
                    int(existing["pull_number"]),
                )

                incoming_data = (
                    event,
                    action,
                    owner,
                    repo,
                    pull_number,
                )

                if existing_data != incoming_data:
                    raise ValueError("Delivery ID was reused with different job data.")

                return self._row_to_job(existing)

            connection.execute(
                """
                INSERT INTO webhook_jobs (
                    delivery_id,
                    event,
                    action,
                    owner,
                    repo,
                    pull_number,
                    status,
                    attempts,
                    created_at,
                    updated_at,
                    available_at,
                    lease_expires_at,
                    last_error
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?, NULL, NULL
                )
                """,
                (
                    delivery_id,
                    event,
                    action,
                    owner,
                    repo,
                    pull_number,
                    now_text,
                    now_text,
                    now_text,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM webhook_jobs
                WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()

            if row is None:
                raise RuntimeError("The newly created webhook job was not found.")

            return self._row_to_job(row)

    def get_job(self, delivery_id: str) -> WebhookJob | None:
        """Retrieve a persisted webhook job."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM webhook_jobs
                WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_job(row)

    def claim_next_job(
        self,
        *,
        lease_seconds: int = 300,
        max_attempts: int = 5,
    ) -> WebhookJob | None:
        """Atomically claim the next available job."""

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer.")

        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a positive integer.")

        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expiry = (now + timedelta(seconds=lease_seconds)).isoformat()

        with self._connection() as connection:
            # Serialize competing workers while selecting a job.
            connection.execute("BEGIN IMMEDIATE")

            # Permanently fail jobs that exhausted their attempts
            # before a worker could recover them.
            connection.execute(
                """
                UPDATE webhook_jobs
                SET
                    status = 'failed',
                    updated_at = ?,
                    lease_expires_at = NULL,
                    last_error = 'Maximum attempts exceeded before recovery.'
                WHERE attempts >= ?
                  AND (
                        (
                            status = 'queued'
                            AND available_at <= ?
                        )
                        OR
                        (
                            status = 'processing'
                            AND lease_expires_at IS NOT NULL
                            AND lease_expires_at <= ?
                        )
                  )
                """,
                (
                    now_text,
                    max_attempts,
                    now_text,
                    now_text,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM webhook_jobs
                WHERE
                    (
                        status = 'queued'
                        AND available_at <= ?
                        AND attempts < ?
                    )
                    OR
                    (
                        status = 'processing'
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= ?
                        AND attempts < ?
                    )
                ORDER BY created_at ASC, delivery_id ASC
                LIMIT 1
                """,
                (
                    now_text,
                    max_attempts,
                    now_text,
                    max_attempts,
                ),
            ).fetchone()

            if row is None:
                return None

            cursor = connection.execute(
                """
                UPDATE webhook_jobs
                SET
                    status = 'processing',
                    attempts = attempts + 1,
                    updated_at = ?,
                    lease_expires_at = ?,
                    last_error = NULL
                WHERE delivery_id = ?
                  AND (
                        (
                            status = 'queued'
                            AND available_at <= ?
                            AND attempts < ?
                        )
                        OR
                        (
                            status = 'processing'
                            AND lease_expires_at IS NOT NULL
                            AND lease_expires_at <= ?
                            AND attempts < ?
                        )
                  )
                """,
                (
                    now_text,
                    lease_expiry,
                    row["delivery_id"],
                    now_text,
                    max_attempts,
                    now_text,
                    max_attempts,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError("Failed to claim the selected webhook job.")

            claimed_row = connection.execute(
                """
                SELECT *
                FROM webhook_jobs
                WHERE delivery_id = ?
                """,
                (row["delivery_id"],),
            ).fetchone()

            if claimed_row is None:
                raise RuntimeError("The claimed webhook job was not found.")

            return self._row_to_job(claimed_row)

    def renew_job_lease(
        self,
        delivery_id: str,
        *,
        attempt: int,
        lease_seconds: int = 300,
    ) -> None:
        """Extend the lease for the current active job attempt."""

        if not delivery_id.strip():
            raise ValueError("delivery_id must not be empty.")

        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer.")

        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a positive integer.")

        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expiry = (now + timedelta(seconds=lease_seconds)).isoformat()

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_jobs
                SET
                    updated_at = ?,
                    lease_expires_at = ?
                WHERE delivery_id = ?
                  AND status = 'processing'
                  AND attempts = ?
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                """,
                (
                    now_text,
                    lease_expiry,
                    delivery_id,
                    attempt,
                    now_text,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Job does not exist, is not processing, "
                    "or this attempt is no longer current."
                )

    def complete_job(
        self,
        delivery_id: str,
        *,
        attempt: int,
    ) -> None:
        """Complete a job only for its current active attempt."""

        if not delivery_id.strip():
            raise ValueError("delivery_id must not be empty.")

        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer.")

        now_text = datetime.now(UTC).isoformat()

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_jobs
                SET
                    status = 'completed',
                    updated_at = ?,
                    lease_expires_at = NULL,
                    last_error = NULL
                WHERE delivery_id = ?
                  AND status = 'processing'
                  AND attempts = ?
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                """,
                (
                    now_text,
                    delivery_id,
                    attempt,
                    now_text,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Job does not exist, is not processing, "
                    "or this attempt is no longer current."
                )

    def fail_job(
        self,
        delivery_id: str,
        *,
        attempt: int,
        error: str,
        retry_delay_seconds: int = 30,
        max_attempts: int = 5,
    ) -> None:
        """Schedule a retry or permanently fail the current job attempt."""

        if not delivery_id.strip():
            raise ValueError("delivery_id must not be empty.")

        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer.")

        if (
            isinstance(retry_delay_seconds, bool)
            or not isinstance(retry_delay_seconds, int)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a non-negative integer.")

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer.")

        now = datetime.now(UTC)
        now_text = now.isoformat()
        safe_error = error.strip()[:1000] or "Unknown error"

        should_retry = attempt < max_attempts

        new_status = "queued" if should_retry else "failed"

        available_at = (
            (now + timedelta(seconds=retry_delay_seconds)).isoformat()
            if should_retry
            else now_text
        )

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_jobs
                SET
                    status = ?,
                    updated_at = ?,
                    available_at = ?,
                    lease_expires_at = NULL,
                    last_error = ?
                WHERE delivery_id = ?
                AND status = 'processing'
                AND attempts = ?
                AND lease_expires_at IS NOT NULL
                AND lease_expires_at > ?
                """,
                (
                    new_status,
                    now_text,
                    available_at,
                    safe_error,
                    delivery_id,
                    attempt,
                    now_text,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Job does not exist, is not processing, "
                    "or this attempt is no longer current."
                )
