import sqlite3
from dataclasses import FrozenInstanceError, fields

import pytest

from codeguard.ai.context_builder import CONTEXT_CONTRACT_VERSION
from codeguard.ai.contracts import (
    REVIEW_CONTRACT_VERSION,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.evidence_matching import EVIDENCE_CONTRACT_VERSION
from codeguard.ai.exceptions import AIProviderConfigurationError
from codeguard.ai.execution_metadata import AIExecutionMetadata
from codeguard.ai.factory import (
    AIReviewProviderSelection,
    create_ai_review_provider,
    create_ai_review_provider_selection,
)
from codeguard.ai.mock_provider import (
    DEFAULT_MOCK_PROFILE_IDENTITY,
    MOCK_PROMPT_CONTRACT_VERSION,
    MOCK_PROVIDER_IDENTITY,
    MockAIReviewProvider,
)
from codeguard.core.config import Settings
from codeguard.webhooks.schema_migrations import migrate_webhook_database


def make_settings(
    *,
    environment: str = "testing",
    ai_provider: str = "mock",
) -> Settings:
    return Settings(
        _env_file=None,
        environment=environment,  # type: ignore[arg-type]
        ai_provider=ai_provider,  # type: ignore[arg-type]
    )


def make_metadata(**updates: str) -> AIExecutionMetadata:
    values = {
        "provider_identity": MOCK_PROVIDER_IDENTITY,
        "model_identity": DEFAULT_MOCK_PROFILE_IDENTITY,
        "prompt_contract_version": MOCK_PROMPT_CONTRACT_VERSION,
        "context_contract_version": CONTEXT_CONTRACT_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "review_contract_version": REVIEW_CONTRACT_VERSION,
    }
    values.update(updates)
    return AIExecutionMetadata(**values)


def make_result(summary: str) -> AIReviewResult:
    return AIReviewResult(summary=summary, findings=())


class GenericTestProvider:
    def __init__(self, execution_metadata: AIExecutionMetadata) -> None:
        self.execution_metadata = execution_metadata

    async def review(self, review_input: AIReviewInput) -> AIReviewResult:
        return make_result("Generic provider test result.")


def test_metadata_descriptor_has_exact_fields_and_identity_conversion() -> None:
    metadata = make_metadata()
    expected_fields = (
        "provider_identity",
        "model_identity",
        "prompt_contract_version",
        "context_contract_version",
        "evidence_contract_version",
        "review_contract_version",
    )

    assert tuple(field.name for field in fields(AIExecutionMetadata)) == expected_fields
    assert metadata.to_identity_components() == {
        name: getattr(metadata, name) for name in expected_fields
    }
    assert not hasattr(metadata, "__dict__")


def test_metadata_descriptor_is_immutable() -> None:
    metadata = make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.model_identity = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "provider_identity",
        "model_identity",
        "prompt_contract_version",
        "context_contract_version",
        "evidence_contract_version",
        "review_contract_version",
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_metadata_rejects_empty_identity_components(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        make_metadata(**{field_name: value})


@pytest.mark.parametrize("value", [None, 42])
def test_metadata_rejects_non_string_identity_components(value: object) -> None:
    with pytest.raises(ValueError, match="provider_identity"):
        make_metadata(provider_identity=value)  # type: ignore[arg-type]


def test_factory_default_mock_selection_has_stable_identity() -> None:
    settings = make_settings()

    first = create_ai_review_provider_selection(settings)
    second = create_ai_review_provider_selection(settings)

    assert first is not None
    assert second is not None
    assert isinstance(first.provider, MockAIReviewProvider)
    assert first.provider.profile_identity == DEFAULT_MOCK_PROFILE_IDENTITY
    assert first.metadata == second.metadata
    assert first.provider is not second.provider
    assert first.metadata.provider_identity == "codeguard.mock"
    assert first.metadata.model_identity == "mock-profile:default-empty-result-v1"


def test_custom_mock_result_requires_and_uses_explicit_profile_identity() -> None:
    result = make_result("Custom profile A.")
    selection = create_ai_review_provider_selection(
        make_settings(),
        mock_result=result,
        mock_profile_identity="mock-profile:custom-a-v1",
    )

    assert selection is not None
    assert isinstance(selection.provider, MockAIReviewProvider)
    assert selection.provider.profile_identity == "mock-profile:custom-a-v1"
    assert selection.metadata.model_identity == "mock-profile:custom-a-v1"


def test_custom_mock_result_cannot_reuse_default_profile_identity() -> None:
    with pytest.raises(AIProviderConfigurationError, match="cannot reuse"):
        create_ai_review_provider_selection(
            make_settings(),
            mock_result=make_result("Different result."),
            mock_profile_identity=DEFAULT_MOCK_PROFILE_IDENTITY,
        )


def test_different_custom_profiles_produce_different_metadata() -> None:
    first = create_ai_review_provider_selection(
        make_settings(),
        mock_result=make_result("Custom profile A."),
        mock_profile_identity="mock-profile:custom-a-v1",
    )
    second = create_ai_review_provider_selection(
        make_settings(),
        mock_result=make_result("Custom profile B."),
        mock_profile_identity="mock-profile:custom-b-v1",
    )

    assert first is not None
    assert second is not None
    assert first.metadata != second.metadata
    assert first.metadata.model_identity != second.metadata.model_identity


def test_custom_mock_without_profile_is_not_eligible_for_durable_reuse() -> None:
    with pytest.raises(AIProviderConfigurationError, match="stable profile"):
        create_ai_review_provider_selection(
            make_settings(),
            mock_result=make_result("Ad hoc result."),
        )

    ad_hoc_provider = MockAIReviewProvider(result=make_result("Ad hoc result."))
    assert ad_hoc_provider.profile_identity is None


def test_contract_versions_have_the_initial_owned_values() -> None:
    assert MOCK_PROMPT_CONTRACT_VERSION == "not-applicable:mock-no-prompt-v1"
    assert CONTEXT_CONTRACT_VERSION == "context-v1"
    assert EVIDENCE_CONTRACT_VERSION == "evidence-v1"
    assert REVIEW_CONTRACT_VERSION == "review-v1"


def test_review_contract_version_matches_database_migration_default(
    tmp_path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    migrate_webhook_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = connection.execute("PRAGMA table_info(ai_operations)").fetchall()

    review_column = next(
        column for column in columns if column[1] == "review_contract_version"
    )
    assert review_column[4] == f"'{REVIEW_CONTRACT_VERSION}'"


def test_metadata_has_no_credential_or_execution_state_fields() -> None:
    field_names = {field.name for field in fields(AIExecutionMetadata)}

    assert not field_names & {
        "api_key",
        "credential",
        "secret",
        "delivery_id",
        "attempt",
        "lease_token",
        "timestamp",
        "result_codec_version",
        "input_fingerprint",
    }


def test_provider_selection_rejects_divergent_metadata() -> None:
    selection = create_ai_review_provider_selection(make_settings())
    assert selection is not None

    mismatched_metadata = make_metadata(
        model_identity="mock-profile:other-v1",
    )

    with pytest.raises(AIProviderConfigurationError, match="does not match"):
        AIReviewProviderSelection(
            provider=selection.provider,
            metadata=mismatched_metadata,
        )

    mismatched_versions = make_metadata(context_contract_version="context-v2")
    with pytest.raises(AIProviderConfigurationError, match="does not match"):
        AIReviewProviderSelection(
            provider=selection.provider,
            metadata=mismatched_versions,
        )


def test_provider_selection_accepts_non_mock_provider_identity() -> None:
    provider_metadata = AIExecutionMetadata(
        provider_identity="openai",
        model_identity="gpt-test-model",
        prompt_contract_version="openai-prompt-v1",
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )
    provider = GenericTestProvider(provider_metadata)

    selection = AIReviewProviderSelection(
        provider=provider,
        metadata=provider_metadata,
    )

    assert selection.provider is provider
    assert selection.metadata.provider_identity == "openai"
    assert selection.metadata.model_identity == "gpt-test-model"


def test_provider_selection_rejects_generic_provider_model_mismatch() -> None:
    provider_metadata = AIExecutionMetadata(
        provider_identity="openai",
        model_identity="gpt-test-model",
        prompt_contract_version="openai-prompt-v1",
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )
    provider = GenericTestProvider(provider_metadata)
    mismatched_metadata = AIExecutionMetadata(
        provider_identity="openai",
        model_identity="gpt-other-model",
        prompt_contract_version="openai-prompt-v1",
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )

    with pytest.raises(AIProviderConfigurationError, match="does not match"):
        AIReviewProviderSelection(
            provider=provider,
            metadata=mismatched_metadata,
        )


def test_provider_selection_rejects_generic_provider_identity_mismatch() -> None:
    provider_metadata = AIExecutionMetadata(
        provider_identity="openai",
        model_identity="gpt-test-model",
        prompt_contract_version="openai-prompt-v1",
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )
    provider = GenericTestProvider(provider_metadata)
    mismatched_metadata = AIExecutionMetadata(
        provider_identity="other-provider",
        model_identity="gpt-test-model",
        prompt_contract_version="openai-prompt-v1",
        context_contract_version=CONTEXT_CONTRACT_VERSION,
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        review_contract_version=REVIEW_CONTRACT_VERSION,
    )

    with pytest.raises(AIProviderConfigurationError, match="does not match"):
        AIReviewProviderSelection(
            provider=provider,
            metadata=mismatched_metadata,
        )


@pytest.mark.asyncio
async def test_provider_selection_closes_optional_provider_client() -> None:
    class ClosableTestProvider(GenericTestProvider):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    metadata = make_metadata()
    provider = ClosableTestProvider(metadata)
    selection = AIReviewProviderSelection(provider=provider, metadata=metadata)

    await selection.aclose()

    assert provider.closed is True


def test_existing_provider_factory_interface_remains_compatible() -> None:
    provider = create_ai_review_provider(make_settings())

    assert isinstance(provider, MockAIReviewProvider)
    assert create_ai_review_provider(make_settings(ai_provider="disabled")) is None


@pytest.mark.asyncio
async def test_ad_hoc_mock_without_profile_remains_usable() -> None:
    provider = MockAIReviewProvider(result=make_result("Ad hoc result."))

    result = await provider.review(
        AIReviewInput(
            owner="octocat",
            repo="CodeGuard",
            pull_number=1,
            head_sha="a" * 40,
            context="Test context.",
        )
    )

    assert result.summary == "Ad hoc result."
    assert provider.profile_identity is None
