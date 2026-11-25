import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from codeguard.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderHTTPError,
    AIProviderPermissionError,
    AIProviderRateLimitError,
    AIProviderRequestError,
    AIProviderResponseError,
    AIProviderServerError,
    AIRequestTransmissionState,
)
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


async def test_not_sent_ai_failure_uses_existing_bounded_retry(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = AIProviderRequestError(
        "pool unavailable",
        transmission_state=AIRequestTransmissionState.NOT_SENT,
    )

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=0,
        max_attempts=3,
    )

    await worker.run_once()
    saved = queue.get_job("delivery-worker-001")

    assert saved is not None
    assert saved.status == "queued"
    assert saved.attempts == 1
    assert "disposition=SAFE_TO_RETRY" in (saved.last_error or "")
    assert "AIProviderRequestError" in (saved.last_error or "")


@pytest.mark.parametrize(
    "provider_error",
    [
        AIProviderRequestError(
            "outcome unknown",
            transmission_state=AIRequestTransmissionState.UNKNOWN,
        ),
        AIProviderAuthenticationError("authentication failed", status_code=401),
        AIProviderPermissionError("permission denied", status_code=403),
        AIProviderResponseError("malformed response", status_code=200),
        AIProviderRateLimitError("rate limited", status_code=429),
        AIProviderServerError("server failed", status_code=503),
        AIProviderHTTPError("bad request", status_code=400),
    ],
)
async def test_uncertain_or_permanent_ai_failures_are_terminal(
    queue: WebhookJobQueue,
    provider_error: Exception,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = provider_error

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=0,
        max_attempts=5,
    )

    await worker.run_once()
    saved = queue.get_job("delivery-worker-001")

    assert saved is not None
    assert saved.status == "failed"
    assert saved.attempts == 1
    assert "disposition=" in (saved.last_error or "")
    assert type(provider_error).__name__ in (saved.last_error or "")


async def test_ai_diagnostic_preserves_safe_metadata_and_hides_sensitive_chain(
    queue: WebhookJobQueue,
    caplog: pytest.LogCaptureFixture,
) -> None:
    enqueue_test_job(queue)
    secret = "private-api-key-and-transport-detail"
    provider_error = AIProviderServerError(
        "safe public message",
        status_code=503,
        request_id="provider-request-456",
    )
    provider_error.__cause__ = RuntimeError(secret)
    processor = AsyncMock()
    processor.process.side_effect = provider_error

    worker = WebhookJobWorker(queue=queue, processor=processor)

    with caplog.at_level(logging.WARNING):
        await worker.run_once()

    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert "disposition=OUTCOME_UNKNOWN" in (saved.last_error or "")
    assert "AIProviderServerError" in (saved.last_error or "")
    assert "status=503" in (saved.last_error or "")
    assert "request_id=provider-request-456" in (saved.last_error or "")
    assert secret not in (saved.last_error or "")
    assert secret not in caplog.text


async def test_not_sent_ai_failure_respects_maximum_attempts(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = AIProviderRequestError(
        "pool unavailable",
        transmission_state=AIRequestTransmissionState.NOT_SENT,
    )

    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=0,
        max_attempts=1,
    )

    await worker.run_once()
    saved = queue.get_job("delivery-worker-001")

    assert saved is not None
    assert saved.status == "failed"
    assert saved.attempts == 1
