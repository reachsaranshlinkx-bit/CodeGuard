from codeguard.ai.context_builder import CONTEXT_CONTRACT_VERSION
from codeguard.ai.contracts import (
    REVIEW_CONTRACT_VERSION,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.evidence_matching import EVIDENCE_CONTRACT_VERSION
from codeguard.ai.execution_metadata import AIExecutionMetadata

MOCK_PROVIDER_IDENTITY = "codeguard.mock"
# Change the profile identity whenever the default mock result semantics change.
DEFAULT_MOCK_PROFILE_IDENTITY = "mock-profile:default-empty-result-v1"

# No prompt is used by this mock. Change this if mock prompt behavior is added.
MOCK_PROMPT_CONTRACT_VERSION = "not-applicable:mock-no-prompt-v1"


class MockAIReviewProvider:
    """
    Deterministic provider implementation for development and testing.

    It performs no network requests and never invokes an LLM.
    Custom result profiles require caller-supplied stable identities. Assign a
    new profile identity whenever a custom result's semantics change.
    """

    def __init__(
        self,
        result: AIReviewResult | None = None,
        error: Exception | None = None,
        *,
        profile_identity: str | None = None,
    ) -> None:
        if result is not None and error is not None:
            raise ValueError("A mock provider cannot have both a result and an error.")

        if profile_identity is not None and (
            not isinstance(profile_identity, str) or not profile_identity.strip()
        ):
            raise ValueError("Mock profile identity must be a non-empty string.")

        self._result = (
            result.model_copy(deep=True)
            if result is not None
            else AIReviewResult(
                summary="Mock AI review completed with no findings.",
                findings=(),
            )
        )

        self._error = error
        self._profile_identity = (
            profile_identity.strip() if profile_identity is not None else None
        )
        self._reviewed_inputs: list[AIReviewInput] = []

    @property
    def profile_identity(self) -> str | None:
        """Return the stable result profile, if this mock is durable-reuse ready."""

        return self._profile_identity

    @property
    def execution_metadata(self) -> AIExecutionMetadata:
        """Return the durable identity associated with this mock profile."""

        if self._profile_identity is None:
            raise ValueError(
                "Mock provider requires a stable profile identity for durable reuse."
            )

        return AIExecutionMetadata(
            provider_identity=MOCK_PROVIDER_IDENTITY,
            model_identity=self._profile_identity,
            prompt_contract_version=MOCK_PROMPT_CONTRACT_VERSION,
            context_contract_version=CONTEXT_CONTRACT_VERSION,
            evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
            review_contract_version=REVIEW_CONTRACT_VERSION,
        )

    @property
    def reviewed_inputs(self) -> tuple[AIReviewInput, ...]:
        """Return a read-only snapshot of received review inputs."""

        return tuple(self._reviewed_inputs)

    async def review(
        self,
        review_input: AIReviewInput,
    ) -> AIReviewResult:
        """Return a configured result or simulate a provider failure."""

        self._reviewed_inputs.append(review_input)

        if self._error is not None:
            raise self._error

        return self._result.model_copy(deep=True)
