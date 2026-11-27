import asyncio
import logging
import sqlite3
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
from codeguard.ai.execution_coordinator import (
    AIExecutionInProgressError,
    AIExecutionOutcomeUnknownError,
    AIExecutionPermanentFailureError,
    AIExecutionRetryableError,
)
from codeguard.ai.operation_identity import AIOperationIdentity
from codeguard.ai.operation_repository import (
    AIOperationRecord,
    AIOperationStatus,
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


def make_permanent_execution_error(
    *,
    safe_error: str = "AIProviderAuthenticationError",
) -> AIExecutionPermanentFailureError:
    identity = AIOperationIdentity(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        head_sha="a" * 40,
        provider_identity="test-provider",
        model_identity="test-model",
        prompt_contract_version="prompt-v1",
        context_contract_version="context-v1",
        evidence_contract_version="evidence-v1",
        review_contract_version="review-v1",
        input_fingerprint=f"ai-execution-input-v1:{'b' * 64}",
    )
    record = AIOperationRecord(
        operation_key=f"ai-operation-v2:{'c' * 64}",
        identity=identity,
        status=AIOperationStatus.FAILED_PERMANENT,
        lease_expires_at=None,
        provider_request_started_at="2026-01-01T00:00:00+00:00",
        result_schema_version=None,
        serialization_version=None,
        result_json=None,
        provider_request_id=None,
        disposition="PERMANENT",
        safe_error=safe_error,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    return AIExecutionPermanentFailureError(record)


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


async def test_retryable_coordinator_error_uses_configured_bounded_retry(
    queue: WebhookJobQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = AIExecutionRetryableError(
        "ai-operation-v2:" + "a" * 64,
        "provider error containing a private credential",
    )
    original_fail_job = queue.fail_job
    fail_job_calls: list[tuple[str, dict[str, object]]] = []

    def record_fail_job(
        delivery_id: str,
        *,
        attempt: int,
        error: str,
        retry_delay_seconds: int = 30,
        max_attempts: int = 5,
    ) -> None:
        kwargs: dict[str, object] = {
            "attempt": attempt,
            "error": error,
            "retry_delay_seconds": retry_delay_seconds,
            "max_attempts": max_attempts,
        }
        fail_job_calls.append((delivery_id, kwargs))
        original_fail_job(
            delivery_id,
            attempt=attempt,
            error=error,
            retry_delay_seconds=retry_delay_seconds,
            max_attempts=max_attempts,
        )

    monkeypatch.setattr(queue, "fail_job", record_fail_job)
    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=17,
        max_attempts=4,
    )

    await worker.run_once()

    assert fail_job_calls == [
        (
            "delivery-worker-001",
            {
                "attempt": 1,
                "error": (
                    "AI_exception=AIExecutionRetryableError "
                    f"operation_key=ai-operation-v2:{'a' * 64} "
                    "operation_state=PENDING"
                ),
                "retry_delay_seconds": 17,
                "max_attempts": 4,
            },
        )
    ]
    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert saved.status == "queued"
    assert saved.attempts == 1
    assert "private credential" not in (saved.last_error or "")


@pytest.mark.parametrize(
    ("coordinator_error", "expected_state"),
    [
        (
            make_permanent_execution_error(
                safe_error="raw provider message with secret-token",
            ),
            "FAILED_PERMANENT",
        ),
        (
            AIExecutionOutcomeUnknownError(
                "ai-operation-v2:" + "d" * 64,
                "raw provider message with secret-token",
            ),
            "OUTCOME_UNKNOWN",
        ),
    ],
)
async def test_terminal_coordinator_errors_fail_webhook_job_immediately(
    queue: WebhookJobQueue,
    coordinator_error: Exception,
    expected_state: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = coordinator_error
    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=17,
        max_attempts=4,
    )

    with caplog.at_level(logging.ERROR):
        await worker.run_once()

    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert saved.status == "failed"
    assert saved.attempts == 1
    assert saved.last_error is not None
    assert f"operation_state={expected_state}" in saved.last_error
    assert "secret-token" not in saved.last_error
    assert "secret-token" not in caplog.text
    assert queue.claim_next_job(max_attempts=4) is None
    processor.process.assert_awaited_once()


async def test_coordinator_permanent_failure_preserves_safe_disposition(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = make_permanent_execution_error()
    worker = WebhookJobWorker(queue=queue, processor=processor)

    await worker.run_once()

    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert saved.status == "failed"
    assert "disposition=PERMANENT" in (saved.last_error or "")
    assert "safe_error=AIProviderAuthenticationError" in (saved.last_error or "")


async def test_in_progress_coordinator_error_uses_bounded_retry(
    queue: WebhookJobQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processor.process.side_effect = AIExecutionInProgressError(
        "ai-operation-v2:" + "e" * 64,
        "another worker owns the operation",
    )
    original_fail_job = queue.fail_job
    fail_job_calls: list[tuple[str, dict[str, object]]] = []

    def record_fail_job(
        delivery_id: str,
        *,
        attempt: int,
        error: str,
        retry_delay_seconds: int = 30,
        max_attempts: int = 5,
    ) -> None:
        kwargs: dict[str, object] = {
            "attempt": attempt,
            "error": error,
            "retry_delay_seconds": retry_delay_seconds,
            "max_attempts": max_attempts,
        }
        fail_job_calls.append((delivery_id, kwargs))
        original_fail_job(
            delivery_id,
            attempt=attempt,
            error=error,
            retry_delay_seconds=retry_delay_seconds,
            max_attempts=max_attempts,
        )

    monkeypatch.setattr(queue, "fail_job", record_fail_job)
    worker = WebhookJobWorker(
        queue=queue,
        processor=processor,
        retry_delay_seconds=23,
        max_attempts=3,
    )

    await worker.run_once()

    assert fail_job_calls == [
        (
            "delivery-worker-001",
            {
                "attempt": 1,
                "error": (
                    "AI_exception=AIExecutionInProgressError "
                    f"operation_key=ai-operation-v2:{'e' * 64} "
                    "operation_state=CLAIMED retry_policy=bounded "
                    "note=webhook_budget_exhaustion_does_not_change_"
                    "ai_operation_state"
                ),
                "retry_delay_seconds": 23,
                "max_attempts": 3,
            },
        )
    ]
    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert saved.status == "queued"
    assert saved.attempts == 1
    assert "operation_state=CLAIMED" in (saved.last_error or "")
    processor.process.assert_awaited_once()


async def test_cancellation_propagates_without_acknowledgement_or_retry(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()
    processing_started = asyncio.Event()

    async def cancel_while_processing(**_: object) -> None:
        processing_started.set()
        await asyncio.Future()

    processor.process.side_effect = cancel_while_processing
    worker = WebhookJobWorker(queue=queue, processor=processor)
    task = asyncio.create_task(worker.run_once())

    await asyncio.wait_for(processing_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert saved.status == "processing"
    assert saved.attempts == 1
    assert saved.last_error is None
    assert queue.claim_next_job() is None
    processor.process.assert_awaited_once()


async def test_stale_worker_failure_cannot_change_reclaimed_attempt(
    queue: WebhookJobQueue,
) -> None:
    enqueue_test_job(queue)
    processor = AsyncMock()

    async def expire_lease_then_fail(**_: object) -> None:
        with sqlite3.connect(queue.database_path) as connection:
            connection.execute(
                """
                UPDATE webhook_jobs
                SET lease_expires_at = ?
                WHERE delivery_id = ?
                """,
                ("2000-01-01T00:00:00+00:00", "delivery-worker-001"),
            )
        reclaimed = queue.claim_next_job(max_attempts=3)
        assert reclaimed is not None
        assert reclaimed.attempts == 2
        raise RuntimeError("stale worker failure")

    processor.process.side_effect = expire_lease_then_fail
    worker = WebhookJobWorker(queue=queue, processor=processor, max_attempts=3)

    await worker.run_once()

    saved = queue.get_job("delivery-worker-001")
    assert saved is not None
    assert saved.status == "processing"
    assert saved.attempts == 2
    assert saved.last_error is None


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
