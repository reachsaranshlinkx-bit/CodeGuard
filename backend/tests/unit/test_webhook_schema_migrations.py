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


def test_fresh_database_migrates_to_version_three(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"

    migrate_webhook_database(database_path)

    assert read_user_version(database_path) == 3


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

    assert read_user_version(database_path) == 3


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

    def fake_operation_repository(*, database_path: Path) -> object:
        events.append("operation_repository")
        assert database_path == tmp_path / "webhook.sqlite3"
        assert events[0] == "migration"
        return object()

    def fake_delivery_store(*, database_path: Path) -> object:
        events.append("delivery_store")
        assert events[0] == "migration"
        return object()

    def fake_job_queue(*, database_path: Path) -> object:
        events.append("job_queue")
        assert events[0] == "migration"
        return object()

    monkeypatch.setattr(main_module, "migrate_webhook_database", fake_migration)
    monkeypatch.setattr(
        main_module,
        "AIOperationRepository",
        fake_operation_repository,
    )
    monkeypatch.setattr(main_module, "WebhookDeliveryStore", fake_delivery_store)
    monkeypatch.setattr(main_module, "WebhookJobQueue", fake_job_queue)

    settings = Settings(
        _env_file=None,
        webhook_database_path=database_path,
    )

    main_module.create_app(settings=settings)

    assert events == [
        "migration",
        "operation_repository",
        "delivery_store",
        "job_queue",
    ]


def test_application_propagates_database_migration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_construction_attempts = []

    def failing_migration(database_path: Path) -> None:
        raise RuntimeError("database migration failed")

    def record_repository_construction(*, database_path: Path) -> object:
        repository_construction_attempts.append(database_path)
        return object()

    monkeypatch.setattr(main_module, "migrate_webhook_database", failing_migration)
    monkeypatch.setattr(
        main_module,
        "AIOperationRepository",
        record_repository_construction,
    )

    with pytest.raises(RuntimeError, match="database migration failed"):
        main_module.create_app(
            settings=Settings(
                _env_file=None,
                webhook_database_path=tmp_path / "migration-failure.sqlite3",
            )
        )

    assert repository_construction_attempts == []


def create_version_one_database(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            CREATE TABLE ai_operations (
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
            CREATE INDEX idx_ai_operations_status_lease
            ON ai_operations(status, lease_expires_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_ai_operations_repository
            ON ai_operations(owner, repo, pull_number, head_sha)
            """
        )


def create_version_two_database(database_path: Path) -> None:
    create_version_one_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            ALTER TABLE ai_operations
            ADD COLUMN review_contract_version TEXT NOT NULL DEFAULT 'review-v1'
            """
        )
        connection.execute("PRAGMA user_version = 2")


def test_version_one_database_upgrades_to_version_three(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    create_version_one_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ai_operations (
                operation_key, owner, repo, pull_number, head_sha,
                provider_identity, model_identity, prompt_contract_version,
                context_contract_version, evidence_contract_version, status,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "existing-operation",
                "octocat",
                "CodeGuard",
                42,
                "a" * 40,
                "provider",
                "model",
                "prompt-v1",
                "context-v1",
                "evidence-v1",
                "SUCCEEDED",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    migrate_webhook_database(database_path)

    assert read_user_version(database_path) == 3

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ai_operations)")
        }
        row = connection.execute(
            """
            SELECT operation_key, status, updated_at, review_contract_version,
                   input_fingerprint
            FROM ai_operations
            WHERE operation_key = 'existing-operation'
            """
        ).fetchone()

    assert "review_contract_version" in columns
    assert row is not None
    assert row == (
        "existing-operation",
        "SUCCEEDED",
        "2026-01-01T00:00:00+00:00",
        "review-v1",
        None,
    )
    assert "input_fingerprint" in columns


def test_review_contract_version_column_exists_with_default(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(ai_operations)")
        }

    assert "review_contract_version" in columns
    column = columns["review_contract_version"]
    assert column[2] == "TEXT"
    assert column[3] == 1
    assert column[4] == "'review-v1'"


def test_existing_operation_row_preserves_data_and_receives_legacy_defaults(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    create_version_one_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ai_operations (
                operation_key, owner, repo, pull_number, head_sha,
                provider_identity, model_identity, prompt_contract_version,
                context_contract_version, evidence_contract_version, status,
                lease_owner, lease_expires_at, provider_request_started_at,
                result_schema_version, serialization_version, result_json,
                provider_request_id, disposition, safe_error,
                created_at, updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?
            )
            """,
            (
                "preserved-operation",
                "octocat",
                "CodeGuard",
                42,
                "b" * 40,
                "provider",
                "model",
                "prompt-v1",
                "context-v1",
                "evidence-v1",
                "SUCCEEDED",
                "worker-1",
                "2026-01-02T00:00:00+00:00",
                "2026-01-01T12:00:00+00:00",
                1,
                1,
                '{"summary":"preserved"}',
                "request-123",
                "OUTCOME_UNKNOWN",
                "safe diagnostic",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T12:00:00+00:00",
            ),
        )

        original_row = connection.execute(
            """
            SELECT *
            FROM ai_operations
            WHERE operation_key = 'preserved-operation'
            """
        ).fetchone()

    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM ai_operations
            WHERE operation_key = 'preserved-operation'
            """
        ).fetchone()
        columns = [
            column[1]
            for column in connection.execute("PRAGMA table_info(ai_operations)")
        ]

    assert original_row is not None
    assert row is not None
    assert row[:-2] == original_row
    assert columns[-2:] == ["review_contract_version", "input_fingerprint"]
    assert row[-2:] == ("review-v1", None)


def test_existing_indexes_remain_intact_after_v3(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    create_version_one_database(database_path)

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


def test_version_two_database_upgrades_without_fingerprinting_legacy_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    create_version_two_database(database_path)
    insert_operation(database_path, pull_number=42, status="SUCCEEDED")

    migrate_webhook_database(database_path)

    assert read_user_version(database_path) == 3
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT operation_key, status, input_fingerprint
            FROM ai_operations
            WHERE operation_key = 'operation-key'
            """
        ).fetchone()
        columns = {
            column[1]: column
            for column in connection.execute("PRAGMA table_info(ai_operations)")
        }

    assert row == ("operation-key", "SUCCEEDED", None)
    assert columns["input_fingerprint"][2] == "TEXT"
    assert columns["input_fingerprint"][3] == 0
    assert columns["input_fingerprint"][4] is None


def test_version_three_database_migration_is_a_no_op(tmp_path: Path) -> None:
    database_path = tmp_path / "webhook.sqlite3"
    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        original_columns = connection.execute(
            "PRAGMA table_info(ai_operations)"
        ).fetchall()

    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = connection.execute("PRAGMA table_info(ai_operations)").fetchall()

    assert columns == original_columns
    assert read_user_version(database_path) == 3


def test_v2_migration_failure_rolls_back_column_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import codeguard.webhooks.schema_migrations as migrations

    database_path = tmp_path / "webhook.sqlite3"
    create_version_one_database(database_path)

    real_apply_migration_v2 = migrations._apply_migration_v2

    def fail_after_real_migration(connection: sqlite3.Connection) -> None:
        real_apply_migration_v2(connection)
        raise RuntimeError("simulated v2 migration failure")

    monkeypatch.setattr(
        migrations,
        "_apply_migration_v2",
        fail_after_real_migration,
    )

    with pytest.raises(RuntimeError, match="simulated v2 migration failure"):
        migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ai_operations)")
        }

    assert "review_contract_version" not in columns
    assert read_user_version(database_path) == 1


def test_v3_migration_failure_rolls_back_column_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import codeguard.webhooks.schema_migrations as migrations

    database_path = tmp_path / "webhook.sqlite3"
    create_version_two_database(database_path)
    real_apply_migration_v3 = migrations._apply_migration_v3

    def fail_after_real_migration(connection: sqlite3.Connection) -> None:
        real_apply_migration_v3(connection)
        raise RuntimeError("simulated v3 migration failure")

    monkeypatch.setattr(
        migrations,
        "_apply_migration_v3",
        fail_after_real_migration,
    )

    with pytest.raises(RuntimeError, match="simulated v3 migration failure"):
        migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ai_operations)")
        }

    assert "input_fingerprint" not in columns
    assert read_user_version(database_path) == 2
