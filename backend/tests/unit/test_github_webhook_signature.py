import pytest

from codeguard.github.webhook_signature import (
    verify_github_webhook_signature,
)

SECRET = "It's a Secret to Everybody"
PAYLOAD = b"Hello, World!"

VALID_SIGNATURE = (
    "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17"
)


def test_accepts_valid_github_signature() -> None:
    assert (
        verify_github_webhook_signature(
            PAYLOAD,
            SECRET,
            VALID_SIGNATURE,
        )
        is True
    )


def test_rejects_invalid_signature() -> None:
    invalid_signature = "sha256=" + ("0" * 64)

    assert (
        verify_github_webhook_signature(
            PAYLOAD,
            SECRET,
            invalid_signature,
        )
        is False
    )


def test_rejects_missing_signature() -> None:
    assert (
        verify_github_webhook_signature(
            PAYLOAD,
            SECRET,
            None,
        )
        is False
    )


@pytest.mark.parametrize(
    "signature",
    [
        "",
        "invalid-signature",
        "sha1=" + ("0" * 40),
        "sha256=invalid",
    ],
)
def test_rejects_malformed_signature(signature: str) -> None:
    assert (
        verify_github_webhook_signature(
            PAYLOAD,
            SECRET,
            signature,
        )
        is False
    )


def test_rejects_non_ascii_signature() -> None:
    assert (
        verify_github_webhook_signature(
            PAYLOAD,
            SECRET,
            "sha256=" + ("é" * 64),
        )
        is False
    )


def test_rejects_empty_secret() -> None:
    assert (
        verify_github_webhook_signature(
            PAYLOAD,
            "",
            VALID_SIGNATURE,
        )
        is False
    )


def test_rejects_tampered_payload() -> None:
    tampered_payload = b"Hello, CodeGuard!"

    assert (
        verify_github_webhook_signature(
            tampered_payload,
            SECRET,
            VALID_SIGNATURE,
        )
        is False
    )
