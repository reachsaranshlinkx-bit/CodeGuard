from __future__ import annotations

import asyncio
import logging

from codeguard.ai.exceptions import (
    AIProviderError,
    AIProviderFailureDisposition,
    classify_ai_provider_error,
)
from codeguard.services.webhook_review_processor import (
    WebhookReviewProcessor,
)
from codeguard.webhooks.job_queue import (
    WebhookJob,
    WebhookJobQueue,
)

logger = logging.getLogger(__name__)


class WebhookJobWorker:
    """Process queued GitHub pull request review jobs."""

    def __init__(
        self,
        *,
        queue: WebhookJobQueue,
        processor: WebhookReviewProcessor,
        lease_seconds: int = 300,
        max_attempts: int = 5,
        retry_delay_seconds: int = 30,
    ) -> None:
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a positive integer.")

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer.")

        if (
            isinstance(retry_delay_seconds, bool)
            or not isinstance(retry_delay_seconds, int)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a non-negative integer.")

        self._queue = queue
        self._processor = processor
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

    async def run_once(self) -> bool:
        """Process one available job.

        Returns True if a job was claimed, otherwise False.
        """

        job = await asyncio.to_thread(
            self._queue.claim_next_job,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
        )

        if job is None:
            return False

        stop_event = asyncio.Event()
        lease_lost_event = asyncio.Event()

        heartbeat_task = asyncio.create_task(
            self._maintain_lease(
                job=job,
                stop_event=stop_event,
                lease_lost_event=lease_lost_event,
            )
        )

        try:
            try:
                await self._processor.process(
                    owner=job.owner,
                    repo=job.repo,
                    pull_number=job.pull_number,
                )
            except AIProviderError as exc:
                disposition = classify_ai_provider_error(exc)
                diagnostic = self._ai_failure_diagnostic(exc, disposition)

                logger.error(
                    "Webhook review job %s failed: %s",
                    job.delivery_id,
                    diagnostic,
                )

                stop_event.set()
                await heartbeat_task

                if not lease_lost_event.is_set():
                    try:
                        terminal_attempt_limit = (
                            job.attempts
                            if disposition != AIProviderFailureDisposition.SAFE_TO_RETRY
                            else self._max_attempts
                        )
                        await asyncio.to_thread(
                            self._queue.fail_job,
                            job.delivery_id,
                            attempt=job.attempts,
                            error=diagnostic,
                            retry_delay_seconds=self._retry_delay_seconds,
                            max_attempts=terminal_attempt_limit,
                        )
                    except ValueError:
                        logger.warning(
                            "Could not mark job %s as failed because "
                            "its lease is no longer current.",
                            job.delivery_id,
                        )

                return True
            except Exception as exc:
                logger.error(
                    "Webhook review job %s failed with an ordinary exception.",
                    job.delivery_id,
                )

                stop_event.set()
                await heartbeat_task

                if not lease_lost_event.is_set():
                    try:
                        await asyncio.to_thread(
                            self._queue.fail_job,
                            job.delivery_id,
                            attempt=job.attempts,
                            error=str(exc),
                            retry_delay_seconds=self._retry_delay_seconds,
                            max_attempts=self._max_attempts,
                        )
                    except ValueError:
                        logger.warning(
                            "Could not mark job %s as failed because "
                            "its lease is no longer current.",
                            job.delivery_id,
                        )

                return True

            stop_event.set()
            await heartbeat_task

            if lease_lost_event.is_set():
                logger.warning(
                    "Job %s finished processing after losing its lease. "
                    "It will not be marked completed by this worker.",
                    job.delivery_id,
                )
                return True

            try:
                await asyncio.to_thread(
                    self._queue.complete_job,
                    job.delivery_id,
                    attempt=job.attempts,
                )
            except ValueError:
                logger.warning(
                    "Could not complete job %s because its lease is no longer current.",
                    job.delivery_id,
                )

            return True

        finally:
            stop_event.set()
            await heartbeat_task

    @staticmethod
    def _ai_failure_diagnostic(
        error: AIProviderError,
        disposition: AIProviderFailureDisposition,
    ) -> str:
        """Build a bounded diagnostic without provider or transport details."""

        parts = [
            f"disposition={disposition.value}",
            f"AI_exception={type(error).__name__}",
        ]

        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int) and not isinstance(status_code, bool):
            parts.append(f"status={status_code}")

        request_id = getattr(error, "request_id", None)
        if isinstance(request_id, str) and request_id.strip():
            safe_request_id = "".join(
                character
                if character.isprintable() and character not in "\r\n"
                else "_"
                for character in request_id.strip()
            )[:200]
            parts.append(f"request_id={safe_request_id}")

        return " ".join(parts)

    async def _maintain_lease(
        self,
        *,
        job: WebhookJob,
        stop_event: asyncio.Event,
        lease_lost_event: asyncio.Event,
    ) -> None:
        """Periodically extend the active job lease."""

        interval = max(0.1, self._lease_seconds / 3)

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=interval,
                )
                return
            except TimeoutError:
                pass

            try:
                await asyncio.to_thread(
                    self._queue.renew_job_lease,
                    job.delivery_id,
                    attempt=job.attempts,
                    lease_seconds=self._lease_seconds,
                )
            except ValueError:
                lease_lost_event.set()
                logger.warning(
                    "Worker lost the lease for job %s.",
                    job.delivery_id,
                )
                return
            except Exception:
                lease_lost_event.set()
                logger.exception(
                    "Failed to renew the lease for job %s.",
                    job.delivery_id,
                )
                return
