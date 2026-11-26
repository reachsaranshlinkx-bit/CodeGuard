from dataclasses import dataclass

from codeguard.ai.context_builder import CONTEXT_CONTRACT_VERSION
from codeguard.ai.contracts import REVIEW_CONTRACT_VERSION, AIReviewResult
from codeguard.ai.evidence_matching import EVIDENCE_CONTRACT_VERSION
from codeguard.ai.exceptions import AIProviderConfigurationError
from codeguard.ai.execution_metadata import AIExecutionMetadata
from codeguard.ai.mock_provider import (
    DEFAULT_MOCK_PROFILE_IDENTITY,
    MOCK_PROMPT_CONTRACT_VERSION,
    MOCK_PROVIDER_IDENTITY,
    MockAIReviewProvider,
)
from codeguard.ai.provider import AIReviewProvider
from codeguard.core.config import Settings


@dataclass(frozen=True, slots=True)
class AIReviewProviderSelection:
    """A provider paired with the metadata for that exact selection."""

    provider: AIReviewProvider
    metadata: AIExecutionMetadata

    def __post_init__(self) -> None:
        """Reject provider/metadata combinations that do not describe each other."""

        if not isinstance(self.provider, MockAIReviewProvider):
            raise AIProviderConfigurationError(
                "Unsupported provider selection for durable execution metadata."
            )

        profile_identity = self.provider.profile_identity
        if profile_identity is None:
            raise AIProviderConfigurationError(
                "Mock provider requires a stable profile identity for durable reuse."
            )

        expected_metadata = _mock_execution_metadata(profile_identity)
        if self.metadata != expected_metadata:
            raise AIProviderConfigurationError(
                "Provider selection metadata does not match its provider."
            )


def _mock_execution_metadata(profile_identity: str) -> AIExecutionMetadata:
    return AIExecutionMetadata(
        provider_identity=MOCK_PROVIDER_IDENTITY,
        model_identity=profile_identity,
        prompt_contract_version=MOCK_PROMPT_CONTRACT_VERSION,
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )


def create_ai_review_provider_selection(
    settings: Settings,
    *,
    mock_result: AIReviewResult | None = None,
    mock_profile_identity: str | None = None,
) -> AIReviewProviderSelection | None:
    """Select a provider and construct its matching durable execution metadata."""

    if settings.ai_provider == "disabled":
        if mock_result is not None or mock_profile_identity is not None:
            raise AIProviderConfigurationError(
                "Mock profiles cannot be configured while the AI provider is disabled."
            )
        return None

    if settings.ai_provider == "mock":
        if settings.environment not in ("development", "testing"):
            raise AIProviderConfigurationError(
                "Mock AI provider is only permitted in development and testing."
            )

        if mock_result is not None and mock_profile_identity is None:
            raise AIProviderConfigurationError(
                "Custom mock results require a stable profile identity "
                "for durable reuse."
            )

        if (
            mock_result is not None
            and isinstance(mock_profile_identity, str)
            and mock_profile_identity.strip() == DEFAULT_MOCK_PROFILE_IDENTITY
        ):
            raise AIProviderConfigurationError(
                "Custom mock results cannot reuse the default mock profile identity."
            )

        profile_identity = (
            DEFAULT_MOCK_PROFILE_IDENTITY
            if mock_profile_identity is None
            else mock_profile_identity
        )

        try:
            provider = MockAIReviewProvider(
                result=mock_result,
                profile_identity=profile_identity,
            )
        except ValueError:
            raise AIProviderConfigurationError(
                "Mock profile identity must be a non-empty string."
            ) from None

        if provider.profile_identity is None:
            raise AIProviderConfigurationError(
                "Mock provider requires a stable profile identity for durable reuse."
            )

        metadata = _mock_execution_metadata(provider.profile_identity)
        return AIReviewProviderSelection(
            provider=provider,
            metadata=metadata,
        )

    raise AIProviderConfigurationError("Unsupported AI provider selection.")


def create_ai_review_provider(
    settings: Settings,
) -> AIReviewProvider | None:
    """Create the configured AI review provider, if enabled."""

    selection = create_ai_review_provider_selection(settings)
    return None if selection is None else selection.provider
