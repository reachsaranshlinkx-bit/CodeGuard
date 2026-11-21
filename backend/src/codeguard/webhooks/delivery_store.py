from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

DeliveryStatus = Literal["processing", "completed", "failed"]

ClaimStatus = Literal["claimed", "in_progress", "completed"]


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    """Result of attempting to claim a webhook delivery."""

    status: ClaimStatus
    attempts: int


@dataclass(frozen=True, slots=True)
class WebhookDelivery:
    """Persisted information about a GitHub webhook delivery."""

    delivery_id: str
    event: str
    action: str
    status: DeliveryStatus
    attempts: int
    received_at: str
    updated_at: str
    lease_expires_at: str | None
    last_error: str | None


class WebhookDeliveryStore:
    """SQLite-backed, duplicate-safe webhook delivery storage."""

    def __init__(
        self,
        database_path: Path,
        lease_seconds: int = 300,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive.")

        self._database_path = Path(database_path)
        self._lease_seconds = lease_seconds

        self.initialize()

    @property
    def database_path(self) -> Path:
        """Return the configured SQLite database path."""

        return self._database_path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open and safely close a SQLite connection."""

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
        """Create the delivery table and its index if necessary."""

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('processing', 'completed', 'failed')),
                    attempts INTEGER NOT NULL CHECK(attempts >= 1),
                    received_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_expires_at TEXT,
                    last_error TEXT
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_webhook_deliveries_status
                ON webhook_deliveries(status, lease_expires_at)
                """
            )

    def claim_delivery(
        self,
        delivery_id: str,
        event: str,
        action: str,
    ) -> DeliveryClaim:
        """Atomically claim a new delivery or retry an eligible one."""

        if not delivery_id.strip():
            raise ValueError("delivery_id must not be empty.")

        if not event.strip():
            raise ValueError("event must not be empty.")

        if not action.strip():
            raise ValueError("action must not be empty.")

        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_expiry = (now + timedelta(seconds=self._lease_seconds)).isoformat()

        with self._connection() as connection:
            # Serialize competing claims to avoid duplicate processing.
            connection.execute("BEGIN IMMEDIATE")

            row = connection.execute(
                """
                SELECT event, action, status, attempts, lease_expires_at
                FROM webhook_deliveries
                WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()

            if row is None:
                connection.execute(
                    """
                    INSERT INTO webhook_deliveries (
                        delivery_id,
                        event,
                        action,
                        status,
                        attempts,
                        received_at,
                        updated_at,
                        lease_expires_at
                    )
                    VALUES (?, ?, ?, 'processing', 1, ?, ?, ?)
                    """,
                    (
                        delivery_id,
                        event,
                        action,
                        now_text,
                        now_text,
                        lease_expiry,
                    ),
                )

                return DeliveryClaim(
                    status="claimed",
                    attempts=1,
                )

            if row["event"] != event or row["action"] != action:
                raise ValueError("Delivery ID was reused with different event data.")

            attempts = int(row["attempts"])

            if row["status"] == "completed":
                return DeliveryClaim(
                    status="completed",
                    attempts=attempts,
                )

            if (
                row["status"] == "processing"
                and row["lease_expires_at"] is not None
                and row["lease_expires_at"] > now_text
            ):
                return DeliveryClaim(
                    status="in_progress",
                    attempts=attempts,
                )

            # Retry failed deliveries or deliveries with expired leases.
            attempts += 1

            connection.execute(
                """
                UPDATE webhook_deliveries
                SET status = 'processing',
                    attempts = ?,
                    updated_at = ?,
                    lease_expires_at = ?,
                    last_error = NULL
                WHERE delivery_id = ?
                """,
                (
                    attempts,
                    now_text,
                    lease_expiry,
                    delivery_id,
                ),
            )

            return DeliveryClaim(
                status="claimed",
                attempts=attempts,
            )

    def mark_completed(
        self,
        delivery_id: str,
        *,
        attempt: int,
    ) -> None:
        """Complete a delivery only for its current processing attempt."""

        now_text = datetime.now(UTC).isoformat()

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_deliveries
                SET status = 'completed',
                    updated_at = ?,
                    lease_expires_at = NULL,
                    last_error = NULL
                WHERE delivery_id = ?
                    AND status = 'processing'
                    AND attempts = ?
                    AND lease_expires_at > ?
                """,
                (now_text, delivery_id, attempt, now_text),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Delivery does not exist, is not processing, "
                    "or this attempt is no longer current."
                )

    def renew_lease(
        self,
        delivery_id: str,
        *,
        attempt: int,
    ) -> None:
        """Extend the lease for the current active processing attempt."""

        now = datetime.now(UTC)
        now_text = now.isoformat()

        lease_expiry = (now + timedelta(seconds=self._lease_seconds)).isoformat()

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_deliveries
                SET updated_at = ?,
                    lease_expires_at = ?
                WHERE delivery_id = ?
                AND status = 'processing'
                AND attempts = ?
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
                    "Delivery lease is expired or this attempt is no longer current."
                )

    def mark_failed(
        self,
        delivery_id: str,
        error: str,
        *,
        attempt: int,
    ) -> None:
        """Fail a delivery only for its current processing attempt."""

        now_text = datetime.now(UTC).isoformat()
        safe_error = error.strip()[:1000] or "Unknown error"

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_deliveries
                SET status = 'failed',
                    updated_at = ?,
                    lease_expires_at = NULL,
                    last_error = ?
                WHERE delivery_id = ?
                    AND status = 'processing'
                    AND attempts = ?
                    AND lease_expires_at > ?
                """,
                (now_text, safe_error, delivery_id, attempt, now_text),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Delivery does not exist, is not processing, "
                    "or this attempt is no longer current."
                )

    def get_delivery(
        self,
        delivery_id: str,
    ) -> WebhookDelivery | None:
        """Retrieve a delivery record for inspection."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM webhook_deliveries
                WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()

        if row is None:
            return None

        return WebhookDelivery(
            delivery_id=str(row["delivery_id"]),
            event=str(row["event"]),
            action=str(row["action"]),
            status=cast(DeliveryStatus, str(row["status"])),
            attempts=int(row["attempts"]),
            received_at=str(row["received_at"]),
            updated_at=str(row["updated_at"]),
            lease_expires_at=(
                str(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            last_error=(
                str(row["last_error"]) if row["last_error"] is not None else None
            ),
        )
