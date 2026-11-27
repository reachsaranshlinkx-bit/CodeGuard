from __future__ import annotations

import sqlite3
from pathlib import Path

CURRENT_SCHEMA_VERSION = 3


def _apply_migration_v1(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_operations (
            operation_key TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            repo TEXT NOT NULL,
            pull_number INTEGER NOT NULL CHECK(pull_number >= 1),
            head_sha TEXT NOT NULL,
            provider_identity TEXT NOT NULL,
            model_identity TEXT NOT NULL,
            prompt_contract_version TEXT NOT NULL,
            context_contract_version TEXT NOT NULL,
            evidence_contract_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'PENDING',
                'CLAIMED',
                'REQUEST_STARTED',
                'SUCCEEDED',
                'FAILED_PERMANENT',
                'OUTCOME_UNKNOWN'
            )),
            lease_owner TEXT,
            lease_expires_at TEXT,
            provider_request_started_at TEXT,
            result_schema_version INTEGER,
            serialization_version INTEGER,
            result_json TEXT,
            provider_request_id TEXT,
            disposition TEXT,
            safe_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ai_operations_status_lease
        ON ai_operations(status, lease_expires_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ai_operations_repository
        ON ai_operations(owner, repo, pull_number, head_sha)
        """
    )


def _apply_migration_v2(connection: sqlite3.Connection) -> None:
    """Add the review contract compatibility default to existing operations."""

    connection.execute(
        """
        ALTER TABLE ai_operations
        ADD COLUMN review_contract_version TEXT NOT NULL DEFAULT 'review-v1'
        """
    )


def _apply_migration_v3(connection: sqlite3.Connection) -> None:
    """Add the optional execution fingerprint without assigning legacy values."""

    connection.execute(
        """
        ALTER TABLE ai_operations
        ADD COLUMN input_fingerprint TEXT
        """
    )


def migrate_webhook_database(database_path: Path) -> None:
    """Migrate the shared webhook database to the latest supported version."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.execute("PRAGMA busy_timeout = 5000")

    try:
        connection.execute("BEGIN IMMEDIATE")

        version_row = connection.execute("PRAGMA user_version").fetchone()
        if version_row is None:
            raise RuntimeError("Could not read SQLite schema version.")

        current_version = int(version_row[0])

        if current_version > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported webhook database schema version "
                f"{current_version}; this application supports version "
                f"{CURRENT_SCHEMA_VERSION}."
            )

        if current_version == 0:
            _apply_migration_v1(connection)
            connection.execute("PRAGMA user_version = 1")
            current_version = 1

        if current_version == 1:
            _apply_migration_v2(connection)
            connection.execute("PRAGMA user_version = 2")
            current_version = 2

        if current_version == 2:
            _apply_migration_v3(connection)
            connection.execute("PRAGMA user_version = 3")

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
