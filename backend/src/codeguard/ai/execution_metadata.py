from dataclasses import dataclass
from typing import TypedDict


class AIExecutionIdentityComponents(TypedDict):
    """Identity components accepted by the durable AI operation identity."""

    provider_identity: str
    model_identity: str
    prompt_contract_version: str
    context_contract_version: str
    evidence_contract_version: str
    review_contract_version: str


@dataclass(frozen=True, slots=True)
class AIExecutionMetadata:
    """Immutable provider and contract identity for one AI execution."""

    provider_identity: str
    model_identity: str
    prompt_contract_version: str
    context_contract_version: str
    evidence_contract_version: str
    review_contract_version: str

    def __post_init__(self) -> None:
        """Normalize and validate every identity component."""

        for field_name in (
            "provider_identity",
            "model_identity",
            "prompt_contract_version",
            "context_contract_version",
            "evidence_contract_version",
            "review_contract_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")

            object.__setattr__(self, field_name, value.strip())

    def to_identity_components(self) -> AIExecutionIdentityComponents:
        """Return the six components for a future operation-identity call."""

        return {
            "provider_identity": self.provider_identity,
            "model_identity": self.model_identity,
            "prompt_contract_version": self.prompt_contract_version,
            "context_contract_version": self.context_contract_version,
            "evidence_contract_version": self.evidence_contract_version,
            "review_contract_version": self.review_contract_version,
        }
