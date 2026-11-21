import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeguard.webhooks.job_queue import WebhookJobQueue


def test_queue_creates_database_and_job_table(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "data" / "webhook_jobs.sqlite3"

    queue = WebhookJobQueue(database_path)

    assert database_path.exists()
    assert queue.database_path == database_path

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(webhook_jobs)")
        }

    assert {
        "delivery_id",
        "event",
        "action",
        "owner",
        "repo",
        "pull_number",
        "status",
        "attempts",
        "created_at",
        "updated_at",
        "available_at",
        "lease_expires_at",
        "last_error",
    }.issubset(columns)


def test_queue_can_be_initialized_more_than_once(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook_jobs.sqlite3"

    WebhookJobQueue(database_path)
    WebhookJobQueue(database_path)

    with sqlite3.connect(database_path) as connection:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'webhook_jobs'
            """
        ).fetchone()

    assert result is not None


def test_enqueue_job_persists_pull_request_details(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    job = queue.enqueue_job(
        delivery_id="delivery-101",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    assert job.delivery_id == "delivery-101"
    assert job.owner == "octocat"
    assert job.repo == "CodeGuard"
    assert job.pull_number == 42
    assert job.status == "queued"
    assert job.attempts == 0
    assert job.lease_expires_at is None
    assert job.last_error is None

    saved = queue.get_job("delivery-101")

    assert saved == job


def test_enqueue_same_delivery_is_idempotent(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    arguments = {
        "delivery_id": "delivery-102",
        "event": "pull_request",
        "action": "opened",
        "owner": "octocat",
        "repo": "CodeGuard",
        "pull_number": 42,
    }

    first = queue.enqueue_job(**arguments)
    second = queue.enqueue_job(**arguments)

    assert first == second

    with sqlite3.connect(queue.database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM webhook_jobs").fetchone()[0]

    assert count == 1


def test_rejects_delivery_id_reused_for_different_job(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-103",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    with pytest.raises(
        ValueError,
        match="different job data",
    ):
        queue.enqueue_job(
            delivery_id="delivery-103",
            event="pull_request",
            action="opened",
            owner="octocat",
            repo="AnotherRepo",
            pull_number=42,
        )


def test_claim_next_job_returns_none_when_queue_is_empty(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    assert queue.claim_next_job() is None


def test_claim_next_job_marks_job_as_processing(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-201",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    job = queue.claim_next_job(lease_seconds=300)

    assert job is not None
    assert job.delivery_id == "delivery-201"
    assert job.status == "processing"
    assert job.attempts == 1
    assert job.lease_expires_at is not None
    assert job.last_error is None

    saved = queue.get_job("delivery-201")

    assert saved == job


def test_claim_next_job_reclaims_expired_processing_lease(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-202",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    original = queue.claim_next_job()

    assert original is not None
    assert original.attempts == 1

    with sqlite3.connect(queue.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_jobs
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-202"),
        )

    retry = queue.claim_next_job()

    assert retry is not None
    assert retry.delivery_id == "delivery-202"
    assert retry.status == "processing"
    assert retry.attempts == 2
    assert retry.lease_expires_at is not None


def test_only_one_concurrent_worker_claims_job(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "concurrent_jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-203",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    def claim_job(_: int) -> str | None:
        job = queue.claim_next_job()
        return job.delivery_id if job is not None else None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(claim_job, range(8)))

    assert results.count("delivery-203") == 1
    assert results.count(None) == 7

    saved = queue.get_job("delivery-203")

    assert saved is not None
    assert saved.status == "processing"
    assert saved.attempts == 1


def test_renew_job_lease_extends_current_attempt(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-301",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    job = queue.claim_next_job(lease_seconds=30)
    assert job is not None
    assert job.lease_expires_at is not None

    before = datetime.fromisoformat(job.lease_expires_at)

    queue.renew_job_lease(
        "delivery-301",
        attempt=job.attempts,
        lease_seconds=300,
    )

    updated = queue.get_job("delivery-301")
    assert updated is not None
    assert updated.lease_expires_at is not None

    after = datetime.fromisoformat(updated.lease_expires_at)

    assert after > before
    assert updated.attempts == 1
    assert updated.status == "processing"


def test_complete_job_marks_current_attempt_completed(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-302",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    job = queue.claim_next_job()
    assert job is not None

    queue.complete_job(
        "delivery-302",
        attempt=job.attempts,
    )

    completed = queue.get_job("delivery-302")

    assert completed is not None
    assert completed.status == "completed"
    assert completed.attempts == 1
    assert completed.lease_expires_at is None
    assert completed.last_error is None

    assert queue.claim_next_job() is None


def test_stale_worker_cannot_complete_retried_job(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-303",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    original = queue.claim_next_job()
    assert original is not None

    with sqlite3.connect(queue.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_jobs
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-303"),
        )

    retry = queue.claim_next_job()
    assert retry is not None
    assert retry.attempts == 2

    with pytest.raises(ValueError, match="no longer current"):
        queue.complete_job(
            "delivery-303",
            attempt=original.attempts,
        )

    queue.complete_job(
        "delivery-303",
        attempt=retry.attempts,
    )

    completed = queue.get_job("delivery-303")
    assert completed is not None
    assert completed.status == "completed"
    assert completed.attempts == 2


def test_failed_job_is_scheduled_for_retry(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-401",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    job = queue.claim_next_job()
    assert job is not None

    queue.fail_job(
        "delivery-401",
        attempt=job.attempts,
        error="Temporary GitHub API error",
        retry_delay_seconds=60,
        max_attempts=3,
    )

    failed = queue.get_job("delivery-401")

    assert failed is not None
    assert failed.status == "queued"
    assert failed.attempts == 1
    assert failed.lease_expires_at is None
    assert failed.last_error == "Temporary GitHub API error"
    assert datetime.fromisoformat(failed.available_at) > datetime.now(UTC)

    assert queue.claim_next_job() is None


def test_job_fails_after_maximum_attempts(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-402",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    job = queue.claim_next_job()
    assert job is not None

    queue.fail_job(
        "delivery-402",
        attempt=job.attempts,
        error="Permanent processing failure",
        max_attempts=1,
    )

    failed = queue.get_job("delivery-402")

    assert failed is not None
    assert failed.status == "failed"
    assert failed.attempts == 1
    assert failed.lease_expires_at is None
    assert failed.last_error == "Permanent processing failure"

    assert queue.claim_next_job() is None


def test_stale_worker_cannot_fail_retried_job(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-403",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    original = queue.claim_next_job()
    assert original is not None

    with sqlite3.connect(queue.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_jobs
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-403"),
        )

    retry = queue.claim_next_job()
    assert retry is not None
    assert retry.attempts == 2

    with pytest.raises(ValueError, match="no longer current"):
        queue.fail_job(
            "delivery-403",
            attempt=original.attempts,
            error="Late failure from old worker",
        )

    current = queue.get_job("delivery-403")

    assert current is not None
    assert current.status == "processing"
    assert current.attempts == 2


def test_expired_job_is_not_reclaimed_after_maximum_attempts(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    queue.enqueue_job(
        delivery_id="delivery-501",
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    # Simulate five worker attempts, each ending in a crash.
    for expected_attempt in range(1, 6):
        job = queue.claim_next_job(max_attempts=5)

        assert job is not None
        assert job.attempts == expected_attempt

        with sqlite3.connect(queue.database_path) as connection:
            connection.execute(
                """
                UPDATE webhook_jobs
                SET lease_expires_at = ?
                WHERE delivery_id = ?
                """,
                (
                    "2000-01-01T00:00:00+00:00",
                    "delivery-501",
                ),
            )

    # The exhausted job must not be reclaimed.
    assert queue.claim_next_job(max_attempts=5) is None

    failed = queue.get_job("delivery-501")

    assert failed is not None
    assert failed.status == "failed"
    assert failed.attempts == 5
    assert failed.lease_expires_at is None
    assert failed.last_error == ("Maximum attempts exceeded before recovery.")


def test_claim_next_job_rejects_invalid_max_attempts(
    tmp_path: Path,
) -> None:
    queue = WebhookJobQueue(tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match="max_attempts"):
        queue.claim_next_job(max_attempts=0)
