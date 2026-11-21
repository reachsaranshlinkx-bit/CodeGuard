import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from codeguard.webhooks.delivery_store import WebhookDeliveryStore


@pytest.fixture
def store(tmp_path: Path) -> WebhookDeliveryStore:
    return WebhookDeliveryStore(
        tmp_path / "webhook_deliveries.sqlite3",
    )


def test_claims_new_delivery(store: WebhookDeliveryStore) -> None:
    claim = store.claim_delivery(
        "delivery-1",
        "pull_request",
        "opened",
    )

    assert claim.status == "claimed"
    assert claim.attempts == 1

    delivery = store.get_delivery("delivery-1")

    assert delivery is not None
    assert delivery.status == "processing"
    assert delivery.attempts == 1


def test_prevents_duplicate_processing(
    store: WebhookDeliveryStore,
) -> None:
    store.claim_delivery("delivery-2", "pull_request", "opened")

    duplicate = store.claim_delivery(
        "delivery-2",
        "pull_request",
        "opened",
    )

    assert duplicate.status == "in_progress"
    assert duplicate.attempts == 1


def test_completed_delivery_is_not_claimed_again(
    store: WebhookDeliveryStore,
) -> None:
    claim = store.claim_delivery(
        "delivery-3",
        "pull_request",
        "opened",
    )

    store.mark_completed(
        "delivery-3",
        attempt=claim.attempts,
    )

    duplicate = store.claim_delivery(
        "delivery-3",
        "pull_request",
        "opened",
    )

    assert duplicate.status == "completed"
    assert duplicate.attempts == 1


def test_failed_delivery_can_be_retried(
    store: WebhookDeliveryStore,
) -> None:
    claim = store.claim_delivery(
        "delivery-4",
        "pull_request",
        "opened",
    )

    store.mark_failed(
        "delivery-4",
        "Temporary GitHub API error",
        attempt=claim.attempts,
    )

    retry = store.claim_delivery(
        "delivery-4",
        "pull_request",
        "opened",
    )

    assert retry.status == "claimed"
    assert retry.attempts == 2

    delivery = store.get_delivery("delivery-4")

    assert delivery is not None
    assert delivery.status == "processing"
    assert delivery.last_error is None


def test_expired_processing_lease_can_be_reclaimed(
    store: WebhookDeliveryStore,
) -> None:
    store.claim_delivery("delivery-5", "pull_request", "opened")

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_deliveries
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-5"),
        )

    retry = store.claim_delivery(
        "delivery-5",
        "pull_request",
        "opened",
    )

    assert retry.status == "claimed"
    assert retry.attempts == 2


def test_rejects_reused_delivery_id_with_different_event(
    store: WebhookDeliveryStore,
) -> None:
    store.claim_delivery("delivery-6", "pull_request", "opened")

    with pytest.raises(ValueError, match="different event data"):
        store.claim_delivery(
            "delivery-6",
            "issues",
            "opened",
        )


def test_cannot_complete_unknown_delivery(
    store: WebhookDeliveryStore,
) -> None:
    with pytest.raises(ValueError):
        store.mark_completed(
            "unknown-delivery",
            attempt=1,
        )


def test_cannot_fail_unknown_delivery(
    store: WebhookDeliveryStore,
) -> None:
    with pytest.raises(ValueError):
        store.mark_failed(
            "unknown-delivery",
            "Test error",
            attempt=1,
        )


@pytest.mark.parametrize(
    ("delivery_id", "event", "action"),
    [
        ("", "pull_request", "opened"),
        ("delivery-7", "", "opened"),
        ("delivery-7", "pull_request", ""),
    ],
)
def test_rejects_empty_delivery_information(
    store: WebhookDeliveryStore,
    delivery_id: str,
    event: str,
    action: str,
) -> None:
    with pytest.raises(ValueError):
        store.claim_delivery(delivery_id, event, action)


def test_store_creates_database_directory(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "data" / "deliveries.sqlite3"

    WebhookDeliveryStore(database_path)

    assert database_path.exists()


def test_delivery_state_persists_across_store_instances(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "webhook_deliveries.sqlite3"

    first_store = WebhookDeliveryStore(database_path)

    first_claim = first_store.claim_delivery(
        "delivery-persist",
        "pull_request",
        "opened",
    )

    first_store.mark_failed(
        "delivery-persist",
        "Temporary GitHub API error",
        attempt=first_claim.attempts,
    )

    second_store = WebhookDeliveryStore(database_path)

    retry = second_store.claim_delivery(
        "delivery-persist",
        "pull_request",
        "opened",
    )

    assert retry.status == "claimed"
    assert retry.attempts == 2

    second_store.mark_completed(
        "delivery-persist",
        attempt=retry.attempts,
    )

    third_store = WebhookDeliveryStore(database_path)

    duplicate = third_store.claim_delivery(
        "delivery-persist",
        "pull_request",
        "opened",
    )

    assert duplicate.status == "completed"
    assert duplicate.attempts == 2


def test_only_one_concurrent_claim_succeeds(
    tmp_path: Path,
) -> None:
    store = WebhookDeliveryStore(
        tmp_path / "concurrent_deliveries.sqlite3",
    )

    def claim_delivery(_: int) -> str:
        result = store.claim_delivery(
            "delivery-concurrent",
            "pull_request",
            "opened",
        )
        return result.status

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(claim_delivery, range(8)))

    assert statuses.count("claimed") == 1
    assert statuses.count("in_progress") == 7


def test_stale_attempt_cannot_modify_retried_delivery(
    tmp_path: Path,
) -> None:
    store = WebhookDeliveryStore(
        tmp_path / "stale_attempt.sqlite3",
    )

    original = store.claim_delivery(
        "delivery-stale",
        "pull_request",
        "opened",
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_deliveries
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-stale"),
        )

    retry = store.claim_delivery(
        "delivery-stale",
        "pull_request",
        "opened",
    )

    assert original.attempts == 1
    assert retry.attempts == 2
    assert retry.status == "claimed"

    with pytest.raises(ValueError, match="no longer current"):
        store.mark_completed(
            "delivery-stale",
            attempt=original.attempts,
        )

    with pytest.raises(ValueError, match="no longer current"):
        store.mark_failed(
            "delivery-stale",
            "Late failure from old worker",
            attempt=original.attempts,
        )

    store.mark_completed(
        "delivery-stale",
        attempt=retry.attempts,
    )

    delivery = store.get_delivery("delivery-stale")

    assert delivery is not None
    assert delivery.status == "completed"
    assert delivery.attempts == 2


def test_renew_lease_extends_current_attempt(
    store: WebhookDeliveryStore,
) -> None:
    claim = store.claim_delivery(
        "delivery-renew",
        "pull_request",
        "opened",
    )

    before = store.get_delivery("delivery-renew")
    assert before is not None
    assert before.lease_expires_at is not None

    store.renew_lease(
        "delivery-renew",
        attempt=claim.attempts,
    )

    after = store.get_delivery("delivery-renew")
    assert after is not None
    assert after.lease_expires_at is not None

    assert datetime.fromisoformat(after.lease_expires_at) > datetime.fromisoformat(
        before.lease_expires_at
    )


@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_expired_lease_cannot_be_finalized(
    tmp_path: Path,
    operation: str,
) -> None:
    store = WebhookDeliveryStore(
        tmp_path / f"expired-{operation}.sqlite3",
    )

    claim = store.claim_delivery(
        "delivery-expired",
        "pull_request",
        "opened",
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_deliveries
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-expired"),
        )

    with pytest.raises(ValueError, match="no longer current"):
        if operation == "complete":
            store.mark_completed(
                "delivery-expired",
                attempt=claim.attempts,
            )
        else:
            store.mark_failed(
                "delivery-expired",
                "Late failure",
                attempt=claim.attempts,
            )


def test_stale_attempt_cannot_renew_lease(
    tmp_path: Path,
) -> None:
    store = WebhookDeliveryStore(
        tmp_path / "stale_renewal.sqlite3",
    )

    original = store.claim_delivery(
        "delivery-renew-stale",
        "pull_request",
        "opened",
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            UPDATE webhook_deliveries
            SET lease_expires_at = ?
            WHERE delivery_id = ?
            """,
            ("2000-01-01T00:00:00+00:00", "delivery-renew-stale"),
        )

    retry = store.claim_delivery(
        "delivery-renew-stale",
        "pull_request",
        "opened",
    )

    assert retry.attempts == 2

    with pytest.raises(ValueError):
        store.renew_lease(
            "delivery-renew-stale",
            attempt=original.attempts,
        )
