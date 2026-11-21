import json
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)

from codeguard.api.dependencies import get_webhook_job_queue
from codeguard.core.config import Settings, get_settings
from codeguard.github.webhook_signature import (
    verify_github_webhook_signature,
)
from codeguard.webhooks.job_queue import WebhookJobQueue

router = APIRouter(
    prefix="/api/v1/webhooks",
    tags=["webhooks"],
)


@router.post(
    "/github",
    status_code=status.HTTP_202_ACCEPTED,
)
async def receive_github_webhook(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    job_queue: Annotated[
        WebhookJobQueue,
        Depends(get_webhook_job_queue),
    ],
    signature: Annotated[
        str | None,
        Header(alias="X-Hub-Signature-256"),
    ] = None,
    event: Annotated[
        str | None,
        Header(alias="X-GitHub-Event"),
    ] = None,
    delivery: Annotated[
        str | None,
        Header(alias="X-GitHub-Delivery"),
    ] = None,
) -> dict[str, Any]:
    """Authenticate, classify, and deduplicate GitHub webhooks."""

    # Fail closed if the webhook secret is not configured.
    if (
        settings.github_webhook_secret is None
        or not settings.github_webhook_secret.get_secret_value().strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub webhook authentication is not configured.",
        )

    # Always verify the original request bytes.
    payload_body = await request.body()

    is_valid = verify_github_webhook_signature(
        payload_body=payload_body,
        secret=settings.github_webhook_secret.get_secret_value(),
        signature_header=signature,
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid GitHub webhook signature.",
        )

    if not event or not delivery:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing GitHub webhook headers.",
        )

    # Parse JSON only after authentication.
    try:
        payload: Any = json.loads(payload_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook JSON payload.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook payload must be a JSON object.",
        )

    # Ignore events CodeGuard does not process.
    if event != "pull_request":
        return {
            "status": "ignored",
            "reason": "unsupported_event",
            "delivery_id": delivery,
        }

    action = payload.get("action")

    if not isinstance(action, str) or action not in {
        "opened",
        "synchronize",
    }:
        return {
            "status": "ignored",
            "reason": "unsupported_pull_request_action",
            "delivery_id": delivery,
        }

    # Validate repository and pull request identifiers.
    repository = payload.get("repository")
    pull_request = payload.get("pull_request")

    if not isinstance(repository, dict) or not isinstance(pull_request, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pull request webhook payload.",
        )

    owner_data = repository.get("owner")
    owner = owner_data.get("login") if isinstance(owner_data, dict) else None
    repo_name = repository.get("name")
    pull_number = pull_request.get("number")

    if (
        not isinstance(owner, str)
        or not owner.strip()
        or not isinstance(repo_name, str)
        or not repo_name.strip()
        or isinstance(pull_number, bool)
        or not isinstance(pull_number, int)
        or pull_number < 1
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid repository or pull request identifiers.",
        )

    # Persist the validated webhook as a durable background job.
    try:
        job = job_queue.enqueue_job(
            delivery_id=delivery,
            event=event,
            action=action,
            owner=owner,
            repo=repo_name,
            pull_number=pull_number,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="GitHub delivery ID conflicts with existing job data.",
        ) from exc

    # Do not enqueue or process completed deliveries again.
    if job.status == "completed":
        return {
            "status": "duplicate",
            "reason": "already_completed",
            "delivery_id": delivery,
        }

    # A background worker already owns this job.
    if job.status == "processing":
        return {
            "status": "in_progress",
            "delivery_id": delivery,
        }

    # The maximum processing attempts have been exhausted.
    if job.status == "failed":
        return {
            "status": "failed",
            "reason": "maximum_attempts_exceeded",
            "delivery_id": delivery,
        }

    # The job is safely stored and ready for a worker.
    return {
        "status": "queued",
        "delivery_id": delivery,
        "event": event,
        "action": action,
        "owner": owner,
        "repo": repo_name,
        "pull_number": pull_number,
    }
