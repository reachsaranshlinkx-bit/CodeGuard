import sqlite3
from pathlib import Path

import pytest

import codeguard.main as main_module
from codeguard.core.config import Settings
from codeguard.webhooks.schema_migrations import migrate_webhook_database


def read_user_version(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("PRAGMA user_version").fetchone()
    assert row is not None
    return int(row[0])


def test_fresh_database_migrates_to_version_one(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    migrate_webhook_database(database_path)

    assert read_user_version(database_path) == 1


def test_migration_creates_ai_operations_table(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'ai_operations'
            """
        ).fetchone()

    assert table == ("ai_operations",)


def test_migration_creates_required_indexes(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        indexes = {
            row[1]: [
                index_row[2]
                for index_row in connection.execute(f"PRAGMA index_info({row[1]})")
            ]
            for row in connection.execute("PRAGMA index_list('ai_operations')")
        }

    assert indexes["idx_ai_operations_status_lease"] == [
        "status",
        "lease_expires_at",
    ]
    assert indexes["idx_ai_operations_repository"] == [
        "owner",
        "repo",
        "pull_number",
        "head_sha",
    ]


def insert_operation(
    database_path: Path,
    *,
    pull_number: int,
    status: str = "PENDING",
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ai_operations (
                operation_key,
                owner,
                repo,
                pull_number,
                head_sha,
                provider_identity,
                model_identity,
                prompt_contract_version,
                context_contract_version,
                evidence_contract_version,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "operation-key",
                "octocat",
                "CodeGuard",
                pull_number,
                "a" * 40,
                "provider",
                "model",
                "prompt-v1",
                "context-v1",
                "evidence-v1",
                status,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )


@pytest.mark.parametrize(
    "status",
    [
        "PENDING",
        "CLAIMED",
        "REQUEST_STARTED",
        "SUCCEEDED",
        "FAILED_PERMANENT",
        "OUTCOME_UNKNOWN",
    ],
)
def test_allowed_operation_states_are_enforced(
    tmp_path: Path,
    status: str,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)

    insert_operation(database_path, pull_number=42, status=status)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT status FROM ai_operations WHERE operation_key = ?",
            ("operation-key",),
        ).fetchone()

    assert row == (status,)


def test_invalid_operation_state_is_rejected(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)

    with pytest.raises(sqlite3.IntegrityError):
        insert_operation(database_path, pull_number=42, status="INVALID")


@pytest.mark.parametrize("pull_number", [0, -1])
def test_invalid_pull_number_is_rejected(
    tmp_path: Path,
    pull_number: int,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)

    with pytest.raises(sqlite3.IntegrityError):
        insert_operation(database_path, pull_number=pull_number)


def test_existing_webhook_jobs_rows_remain_unchanged(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE webhook_jobs (
                delivery_id TEXT PRIMARY KEY,
                event TEXT NOT NULL,
                action TEXT NOT NULL,
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                pull_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
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
            INSERT INTO webhook_jobs (
                delivery_id, event, action, owner, repo, pull_number,
                status, attempts, created_at, updated_at, available_at,
                lease_expires_at, last_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "delivery-job",
                "pull_request",
                "opened",
                "octocat",
                "CodeGuard",
                42,
                "queued",
                0,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                None,
                None,
            ),
        )

    with sqlite3.connect(database_path) as connection:
        original_row = connection.execute(
            "SELECT * FROM webhook_jobs WHERE delivery_id = 'delivery-job'"
        ).fetchone()

    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM webhook_jobs WHERE delivery_id = 'delivery-job'"
        ).fetchone()

    assert original_row is not None
    assert row == original_row


def test_existing_webhook_deliveries_rows_remain_unchanged(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE webhook_deliveries (
                delivery_id TEXT PRIMARY KEY,
                event TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                received_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                lease_expires_at TEXT,
                last_error TEXT
            )
            """
        )

    # Correct the intentionally simple fixture schema through an executable
    # compatible row setup for this preservation test.
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO webhook_deliveries (
                delivery_id, event, action, status, attempts,
                received_at, updated_at, lease_expires_at, last_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "delivery-store",
                "pull_request",
                "opened",
                "processing",
                1,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                None,
                None,
            ),
        )

    with sqlite3.connect(database_path) as connection:
        original_row = connection.execute(
            """
            SELECT * FROM webhook_deliveries
            WHERE delivery_id = 'delivery-store'
            """
        ).fetchone()

    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT * FROM webhook_deliveries
            WHERE delivery_id = 'delivery-store'
            """
        ).fetchone()

    assert original_row is not None
    assert row == original_row


def test_migration_is_safe_when_called_twice(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    migrate_webhook_database(database_path)
    migrate_webhook_database(database_path)

    assert read_user_version(database_path) == 1


def test_supported_migration_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import codeguard.webhooks.schema_migrations as migrations

    database_path = tmp_path / "webhook.sqlite3"

    real_apply_migration_v1 = migrations._apply_migration_v1

    def fail_after_real_migration(connection: sqlite3.Connection) -> None:
        real_apply_migration_v1(connection)
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(
        migrations,
        "_apply_migration_v1",
        fail_after_real_migration,
    )

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'ai_operations'
            """
        ).fetchone()
        indexes = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index'
                  AND name IN (
                    'idx_ai_operations_status_lease',
                    'idx_ai_operations_repository'
                  )
                """
            )
        }

    assert table is None
    assert indexes == set()
    assert read_user_version(database_path) == 0


def test_future_schema_version_is_rejected_without_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 999")
        connection.execute(
            """
            CREATE TABLE webhook_jobs (
                delivery_id TEXT PRIMARY KEY
            )
            """
        )
        connection.execute(
            "INSERT INTO webhook_jobs (delivery_id) VALUES ('delivery-1')"
        )

    with pytest.raises(RuntimeError, match="(?i)unsupported"):
        migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        ai_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'ai_operations'
            """
        ).fetchone()
        job = connection.execute("SELECT delivery_id FROM webhook_jobs").fetchone()

    assert ai_table is None
    assert job == ("delivery-1",)
    assert read_user_version(database_path) == 999


def test_application_runs_migration_before_store_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    database_path = tmp_path / "webhook.sqlite3"

    def fake_migration(received_path: Path) -> None:
        events.append("migration")
        assert received_path == database_path

    def fake_delivery_store(*, database_path: Path) -> object:
        events.append("delivery_store")
        assert events[0] == "migration"
        return object()

    def fake_job_queue(*, database_path: Path) -> object:
        events.append("job_queue")
        assert events[0] == "migration"
        return object()

    monkeypatch.setattr(main_module, "migrate_webhook_database", fake_migration)
    monkeypatch.setattr(main_module, "WebhookDeliveryStore", fake_delivery_store)
    monkeypatch.setattr(main_module, "WebhookJobQueue", fake_job_queue)

    settings = Settings(
        _env_file=None,
        webhook_database_path=database_path,
    )

    main_module.create_app(settings=settings)

    assert events == ["migration", "delivery_store", "job_queue"]
