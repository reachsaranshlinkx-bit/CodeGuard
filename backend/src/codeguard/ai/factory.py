from dataclasses import dataclass

from codeguard.ai.contracts import AIReviewResult
from codeguard.ai.exceptions import AIProviderConfigurationError
from codeguard.ai.execution_metadata import AIExecutionMetadata
from codeguard.ai.mock_provider import (
    DEFAULT_MOCK_PROFILE_IDENTITY,
    MockAIReviewProvider,
)
from codeguard.ai.openai_provider import OpenAIReviewProvider
from codeguard.ai.provider import (
    AIReviewProvider,
    AIReviewProviderExecutionIdentity,
    ClosableAIReviewProvider,
)
from codeguard.core.config import Settings


@dataclass(frozen=True, slots=True)
class AIReviewProviderSelection:
    """A provider paired with the metadata for that exact selection."""

    provider: AIReviewProviderExecutionIdentity
    metadata: AIExecutionMetadata

    def __post_init__(self) -> None:
        """Reject provider/metadata combinations that do not describe each other."""

        try:
            provider_metadata = self.provider.execution_metadata
        except (AttributeError, TypeError, ValueError):
            raise AIProviderConfigurationError(
                "Provider selection has invalid durable execution metadata."
            ) from None

        if not isinstance(provider_metadata, AIExecutionMetadata):
            raise AIProviderConfigurationError(
                "Provider selection must expose valid durable execution metadata."
            )

        if self.metadata != provider_metadata:
            raise AIProviderConfigurationError(
                "Provider selection metadata does not match its provider."
            )

    async def aclose(self) -> None:
        """Close the selected provider when it owns a closable client."""

        if isinstance(self.provider, ClosableAIReviewProvider):
            await self.provider.aclose()


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
            mock_provider = MockAIReviewProvider(
                result=mock_result,
                profile_identity=profile_identity,
            )
        except ValueError:
            raise AIProviderConfigurationError(
                "Mock profile identity must be a non-empty string."
            ) from None

        if mock_provider.profile_identity is None:
            raise AIProviderConfigurationError(
                "Mock provider requires a stable profile identity for durable reuse."
            )

        metadata = mock_provider.execution_metadata
        return AIReviewProviderSelection(
            provider=mock_provider,
            metadata=metadata,
        )

    if settings.ai_provider == "openai":
        if settings.openai_api_key is None or settings.openai_model is None:
            raise AIProviderConfigurationError(
                "OpenAI provider requires an API key and model identity."
            )

        openai_provider = OpenAIReviewProvider(
            model_identity=settings.openai_model,
            api_key=settings.openai_api_key,
        )
        return AIReviewProviderSelection(
            provider=openai_provider,
            metadata=openai_provider.execution_metadata,
        )

    raise AIProviderConfigurationError("Unsupported AI provider selection.")


def create_ai_review_provider(
    settings: Settings,
) -> AIReviewProvider | None:
    """Create the configured AI review provider, if enabled."""

    selection = create_ai_review_provider_selection(settings)
    return None if selection is None else selection.provider
