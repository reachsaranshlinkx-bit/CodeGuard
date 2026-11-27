import pytest

from codeguard.ai.exceptions import AIProviderConfigurationError
from codeguard.ai.factory import (
    create_ai_review_provider,
    create_ai_review_provider_selection,
)
from codeguard.ai.mock_provider import MockAIReviewProvider
from codeguard.core.config import Settings


def test_disabled_provider_selection_returns_none() -> None:
    settings = Settings(
        _env_file=None,
        environment="development",
        ai_provider="disabled",
    )

    assert create_ai_review_provider(settings) is None


@pytest.mark.parametrize("environment", ["development", "testing"])
def test_mock_provider_selection_returns_a_new_mock_provider(
    environment: str,
) -> None:
    settings = Settings(
        _env_file=None,
        environment=environment,
        ai_provider="mock",
    )

    provider = create_ai_review_provider(settings)

    assert isinstance(provider, MockAIReviewProvider)
    assert provider is not create_ai_review_provider(settings)


@pytest.mark.parametrize(
    ("environment", "provider_selection"),
    [
        ("production", "mock"),
        ("staging", "mock"),
        ("development", "unsupported"),
    ],
)
def test_factory_rejects_incompatible_settings(
    environment: str,
    provider_selection: str,
) -> None:
    settings = Settings.model_construct(
        environment=environment,
        ai_provider=provider_selection,
    )

    with pytest.raises(AIProviderConfigurationError):
        create_ai_review_provider(settings)


def test_openai_factory_fails_closed_until_request_adapter_exists() -> None:
    settings = Settings(
        _env_file=None,
        environment="testing",
        ai_provider="openai",
        openai_api_key="test-openai-secret",
        openai_model="gpt-test-model",
    )

    with pytest.raises(
        AIProviderConfigurationError,
        match="request adapter is implemented",
    ) as exc_info:
        create_ai_review_provider_selection(settings)

    assert "test-openai-secret" not in str(exc_info.value)
