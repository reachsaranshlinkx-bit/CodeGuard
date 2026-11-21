from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from codeguard.services.webhook_job_worker import WebhookJobWorker
from codeguard.webhooks.job_queue import WebhookJobQueue


@pytest.fixture
def queue(tmp_path: Path) -> WebhookJobQueue:
    return WebhookJobQueue(tmp_path / "webhook_jobs.sqlite3")


def enqueue_test_job(
    queue: WebhookJobQueue,
    delivery_id: str = "delivery-worker-001",
) -> None:
    queue.enqueue_job(
        delivery_id=delivery_id,
        event="pull_request",
        action="opened",
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )


async def test_run_once_returns_false_when_queue_is_empty(
    queue: WebhookJobQueue,
) -> None:
    processor = AsyncMock()

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
    )

    result = await worker.run_once()

    assert result is False
    processor.process.assert_not_awaited()


async def test_run_once_processes_job_and_marks_it_completed(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
    )

    result = await worker.run_once()

    assert result is True

    processor.process.assert_awaited_once_with(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
    )

    saved = queue.get_job("delivery-worker-001")

    assert saved is not None
    assert saved.status == "completed"
    assert saved.attempts == 1
    assert saved.lease_expires_at is None


async def test_run_once_schedules_retry_when_processing_fails(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = RuntimeError("Temporary GitHub API error")

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=0,
        max_attempts=3,
    )

    result = await worker.run_once()

    assert result is True

    saved = queue.get_job("delivery-worker-001")

    assert saved is not None
    assert saved.status == "queued"
    assert saved.attempts == 1
    assert saved.lease_expires_at is None
    assert saved.last_error == "Temporary GitHub API error"


async def test_run_once_permanently_fails_after_maximum_attempts(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = RuntimeError("Permanent processing failure")

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        max_attempts=1,
    )

    result = await worker.run_once()

    assert result is True

    saved = queue.get_job("delivery-worker-001")

    assert saved is not None
    assert saved.status == "failed"
    assert saved.attempts == 1
    assert saved.lease_expires_at is None
    assert saved.last_error == "Permanent processing failure"
