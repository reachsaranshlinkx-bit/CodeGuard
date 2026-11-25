from codeguard.ai.exceptions import AIProviderConfigurationError
from codeguard.ai.mock_provider import MockAIReviewProvider
from codeguard.ai.provider import AIReviewProvider
from codeguard.core.config import Settings


def create_ai_review_provider(
    settings: Settings,
) -> AIReviewProvider | None:
    """Create the configured AI review provider, if enabled."""

    if settings.ai_provider == "disabled":
        return None

    if settings.ai_provider == "mock":
        if settings.environment not in ("development", "testing"):
            raise AIProviderConfigurationError(
                "Mock AI provider is only permitted in development and testing."
            )

        return MockAIReviewProvider()

    raise AIProviderConfigurationError("Unsupported AI provider selection.")
