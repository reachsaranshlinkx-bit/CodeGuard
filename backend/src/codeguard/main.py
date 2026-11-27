import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from codeguard.ai.execution_coordinator import AIExecutionCoordinator
from codeguard.ai.factory import (
    AIReviewProviderSelection,
    create_ai_review_provider_selection,
)
from codeguard.ai.operation_repository import AIOperationRepository
from codeguard.api.v1.health import router as health_router
from codeguard.api.v1.reviews import router as reviews_router
from codeguard.api.v1.webhooks import router as webhooks_router
from codeguard.context.repository_context_builder import (
    RepositoryContextBuilder,
)
from codeguard.core.config import Settings, get_settings
from codeguard.core.logging_config import configure_logging
from codeguard.github.client import GitHubClient
from codeguard.services.pull_request_analyzer import PullRequestAnalyzer
from codeguard.services.webhook_job_worker import WebhookJobWorker
from codeguard.services.webhook_review_processor import (
    WebhookReviewProcessor,
)
from codeguard.webhooks.delivery_store import WebhookDeliveryStore
from codeguard.webhooks.job_queue import WebhookJobQueue
from codeguard.webhooks.schema_migrations import migrate_webhook_database

logger = logging.getLogger(__name__)


async def run_webhook_worker(
    *,
    worker: WebhookJobWorker,
    poll_interval_seconds: int,
    stop_event: asyncio.Event,
) -> None:
    """Poll the queue and process available webhook jobs."""

    while not stop_event.is_set():
        try:
            job_claimed = await worker.run_once()
        except Exception:
            logger.exception("webhook_worker_iteration_failed")
            job_claimed = False

        if not job_claimed:
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=poll_interval_seconds,
                )
            except TimeoutError:
                continue


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the CodeGuard FastAPI application."""

    app_settings = settings or get_settings()

    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage background worker and GitHub client lifecycle."""

        if not app_settings.webhook_worker_enabled:
            logger.info("webhook_worker_disabled")
            yield
            return

        token = (
            app_settings.github_token.get_secret_value()
            if app_settings.github_token is not None
            else None
        )

        github_client = GitHubClient(token=token)
        worker_task: asyncio.Task[None] | None = None
        stop_event = asyncio.Event()
        provider_selection: AIReviewProviderSelection | None = None

        try:
            provider_selection = create_ai_review_provider_selection(app_settings)
            ai_execution_coordinator = (
                AIExecutionCoordinator(
                    provider_selection=provider_selection,
                    repository=app.state.ai_operation_repository,
                )
                if provider_selection is not None
                else None
            )

            analyzer = PullRequestAnalyzer(
                github_client=github_client,
                repository_context_builder=RepositoryContextBuilder(
                    github_client,
                ),
                ai_execution_coordinator=ai_execution_coordinator,
            )

            processor = WebhookReviewProcessor(
                analyzer=analyzer,
                github_client=github_client,
                publishing_enabled=app_settings.enable_review_publishing,
            )

            worker = WebhookJobWorker(
                queue=app.state.webhook_job_queue,
                processor=processor,
                max_attempts=app_settings.webhook_worker_max_attempts,
                retry_delay_seconds=app_settings.webhook_worker_retry_delay_seconds,
            )

            app.state.webhook_worker = worker

            worker_task = asyncio.create_task(
                run_webhook_worker(
                    worker=worker,
                    poll_interval_seconds=(
                        app_settings.webhook_worker_poll_interval_seconds
                    ),
                    stop_event=stop_event,
                ),
                name="codeguard-webhook-worker",
            )

            logger.info("webhook_worker_started")

            try:
                yield
            finally:
                stop_event.set()

                if worker_task is not None:
                    await worker_task

                logger.info("webhook_worker_stopped")

        finally:
            try:
                if provider_selection is not None:
                    await provider_selection.aclose()
            finally:
                await github_client.aclose()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=app_settings.app_description,
        debug=app_settings.debug,
        lifespan=lifespan,
    )

    migrate_webhook_database(app_settings.webhook_database_path)

    app.state.ai_operation_repository = AIOperationRepository(
        database_path=app_settings.webhook_database_path,
    )

    app.state.webhook_delivery_store = WebhookDeliveryStore(
        database_path=app_settings.webhook_database_path,
    )

    app.state.webhook_job_queue = WebhookJobQueue(
        database_path=app_settings.webhook_database_path,
    )

    app.state.webhook_worker = None

    app.dependency_overrides[get_settings] = lambda: app_settings

    app.include_router(health_router)
    app.include_router(reviews_router)
    app.include_router(webhooks_router)

    logger.info(
        "application_created",
        extra={
            "environment": app_settings.environment,
        },
    )

    return app


app = create_app()
