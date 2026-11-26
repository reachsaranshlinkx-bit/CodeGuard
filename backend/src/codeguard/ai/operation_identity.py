from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

AI_OPERATION_V1_FORMAT_VERSION = "ai-operation-v1"
AI_OPERATION_V2_FORMAT_VERSION = "ai-operation-v2"
AI_OPERATION_FORMAT_VERSION = AI_OPERATION_V2_FORMAT_VERSION

_INPUT_FINGERPRINT_PATTERN = re.compile(r"ai-execution-input-v1:[0-9a-f]{64}\Z")


def _normalize_repository_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")

    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return normalized


def _normalize_required_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return normalized


def _normalize_identity_fields(identity: LegacyAIOperationIdentity) -> None:
    object.__setattr__(
        identity,
        "owner",
        _normalize_repository_identifier(identity.owner, "owner"),
    )
    object.__setattr__(
        identity,
        "repo",
        _normalize_repository_identifier(identity.repo, "repo"),
    )

    if (
        isinstance(identity.pull_number, bool)
        or not isinstance(identity.pull_number, int)
        or identity.pull_number <= 0
    ):
        raise ValueError("pull_number must be a positive integer.")

    object.__setattr__(
        identity,
        "head_sha",
        _normalize_required_string(identity.head_sha, "head_sha").lower(),
    )
    for field_name in (
        "provider_identity",
        "model_identity",
        "prompt_contract_version",
        "context_contract_version",
        "evidence_contract_version",
        "review_contract_version",
    ):
        object.__setattr__(
            identity,
            field_name,
            _normalize_required_string(
                getattr(identity, field_name),
                field_name,
            ),
        )


@dataclass(frozen=True, slots=True)
class LegacyAIOperationIdentity:
    """The existing v1 identity fields, without an execution input fingerprint."""

    owner: str
    repo: str
    pull_number: int
    head_sha: str
    provider_identity: str
    model_identity: str
    prompt_contract_version: str
    context_contract_version: str
    evidence_contract_version: str
    review_contract_version: str

    def __post_init__(self) -> None:
        _normalize_identity_fields(self)


@dataclass(frozen=True, slots=True)
class AIOperationIdentity(LegacyAIOperationIdentity):
    """Stable v2 identity inputs for one logical AI review operation."""

    input_fingerprint: str

    def __post_init__(self) -> None:
        _normalize_identity_fields(self)

        if (
            not isinstance(self.input_fingerprint, str)
            or _INPUT_FINGERPRINT_PATTERN.fullmatch(self.input_fingerprint) is None
        ):
            raise ValueError(
                "input_fingerprint must be a versioned lowercase SHA-256 fingerprint."
            )


def _build_key(canonical: dict[str, object], format_version: str) -> str:
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{format_version}:{digest}"


def build_legacy_ai_operation_key(identity: LegacyAIOperationIdentity) -> str:
    """Rebuild the exact historical v1 key for validating stored legacy rows."""

    if type(identity) is not LegacyAIOperationIdentity:
        raise TypeError("Legacy key construction requires a v1 identity.")

    canonical = {
        "format_version": AI_OPERATION_V1_FORMAT_VERSION,
        "owner": identity.owner,
        "repo": identity.repo,
        "pull_number": identity.pull_number,
        "head_sha": identity.head_sha,
        "provider_identity": identity.provider_identity,
        "model_identity": identity.model_identity,
        "prompt_contract_version": identity.prompt_contract_version,
        "context_contract_version": identity.context_contract_version,
        "evidence_contract_version": identity.evidence_contract_version,
        "review_contract_version": identity.review_contract_version,
    }
    return _build_key(canonical, AI_OPERATION_V1_FORMAT_VERSION)


def build_ai_operation_key(identity: AIOperationIdentity) -> str:
    """Build the canonical v2 key for a fingerprinted AI operation."""

    if not isinstance(identity, AIOperationIdentity):
        raise TypeError("New operation keys require a v2 identity.")

    canonical = {
        "format_version": AI_OPERATION_V2_FORMAT_VERSION,
        "owner": identity.owner,
        "repo": identity.repo,
        "pull_number": identity.pull_number,
        "head_sha": identity.head_sha,
        "provider_identity": identity.provider_identity,
        "model_identity": identity.model_identity,
        "prompt_contract_version": identity.prompt_contract_version,
        "context_contract_version": identity.context_contract_version,
        "evidence_contract_version": identity.evidence_contract_version,
        "review_contract_version": identity.review_contract_version,
        "input_fingerprint": identity.input_fingerprint,
    }
    return _build_key(canonical, AI_OPERATION_V2_FORMAT_VERSION)
