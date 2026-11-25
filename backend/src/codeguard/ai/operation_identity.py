from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

AI_OPERATION_FORMAT_VERSION = "ai-operation-v1"


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


@dataclass(frozen=True, slots=True)
class AIOperationIdentity:
    """Stable identity inputs for one logical AI review operation."""

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
        object.__setattr__(
            self,
            "owner",
            _normalize_repository_identifier(self.owner, "owner"),
        )
        object.__setattr__(
            self,
            "repo",
            _normalize_repository_identifier(self.repo, "repo"),
        )

        if (
            isinstance(self.pull_number, bool)
            or not isinstance(self.pull_number, int)
            or self.pull_number <= 0
        ):
            raise ValueError("pull_number must be a positive integer.")

        object.__setattr__(
            self,
            "head_sha",
            _normalize_required_string(self.head_sha, "head_sha").lower(),
        )
        object.__setattr__(
            self,
            "provider_identity",
            _normalize_required_string(
                self.provider_identity,
                "provider_identity",
            ),
        )
        object.__setattr__(
            self,
            "model_identity",
            _normalize_required_string(
                self.model_identity,
                "model_identity",
            ),
        )
        object.__setattr__(
            self,
            "prompt_contract_version",
            _normalize_required_string(
                self.prompt_contract_version,
                "prompt_contract_version",
            ),
        )
        object.__setattr__(
            self,
            "context_contract_version",
            _normalize_required_string(
                self.context_contract_version,
                "context_contract_version",
            ),
        )
        object.__setattr__(
            self,
            "evidence_contract_version",
            _normalize_required_string(
                self.evidence_contract_version,
                "evidence_contract_version",
            ),
        )
        object.__setattr__(
            self,
            "review_contract_version",
            _normalize_required_string(
                self.review_contract_version,
                "review_contract_version",
            ),
        )


def build_ai_operation_key(identity: AIOperationIdentity) -> str:
    """Build a canonical, explicitly versioned AI operation key."""

    canonical = {
        "format_version": AI_OPERATION_FORMAT_VERSION,
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
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{AI_OPERATION_FORMAT_VERSION}:{digest}"
