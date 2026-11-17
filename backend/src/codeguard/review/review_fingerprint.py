from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from codeguard.review.comment_models import InlineReviewComment
from codeguard.review.finding_models import ReviewReport


def create_review_fingerprint(
    report: ReviewReport,
    comments: Sequence[InlineReviewComment],
) -> str:
    """Create a deterministic SHA-256 identity for a GitHub review."""

    comment_payloads = [comment.to_github_payload() for comment in comments]

    # Comment order must not affect the review identity.
    comment_payloads.sort(
        key=lambda comment: (
            str(comment["path"]),
            int(comment["line"]),
            str(comment["side"]),
            str(comment["body"]),
        )
    )

    identity = {
        "version": 1,
        "owner": report.owner.casefold(),
        "repo": report.repo.casefold(),
        "pull_number": report.pull_number,
        "head_sha": report.head_sha.lower(),
        "comments": comment_payloads,
    }

    # Canonical JSON ensures identical data produces identical bytes.
    canonical_json = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def create_review_marker(fingerprint: str) -> str:
    """Create a validated hidden marker for a GitHub review."""

    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise ValueError("Fingerprint must be a valid lowercase SHA-256 hex digest.")

    return f"<!-- codeguard-review:v1:{fingerprint} -->"
