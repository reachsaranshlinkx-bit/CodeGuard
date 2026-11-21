import hashlib
import hmac


def verify_github_webhook_signature(
    payload_body: bytes,
    secret: str,
    signature_header: str | None,
) -> bool:
    """Verify a GitHub webhook using its HMAC-SHA256 signature."""

    # Fail closed if the configured secret or signature is missing.
    if not secret or not secret.strip():
        return False

    if not signature_header:
        return False

    # GitHub uses the sha256= signature format.
    if not signature_header.startswith("sha256="):
        return False

    # HTTP headers should contain ASCII signature characters.
    try:
        provided_signature = signature_header.encode("ascii")
    except UnicodeEncodeError:
        return False

    # Calculate the expected signature from the original raw payload.
    expected_digest = hmac.new(
        secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    expected_signature = f"sha256={expected_digest}".encode("ascii")

    # Constant-time comparison helps mitigate timing attacks.
    return hmac.compare_digest(
        expected_signature,
        provided_signature,
    )
