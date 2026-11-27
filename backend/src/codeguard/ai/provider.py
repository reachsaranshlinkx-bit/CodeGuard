from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codeguard.ai.contracts import AIReviewInput, AIReviewResult
from codeguard.ai.execution_metadata import AIExecutionMetadata
from codeguard.ai.telemetry import AIProviderUsage


@dataclass(frozen=True, slots=True)
class AIProviderExecutionResult:
    """Review result and optional per-request metadata from an instrumented provider."""

    review_result: AIReviewResult
    usage: AIProviderUsage | None
    request_duration_ms: float
    provider_identity: str | None = None
    model_identity: str | None = None


@runtime_checkable
class InstrumentedAIReviewProvider(Protocol):
    """Optional capability that reports metadata for this specific provider call."""

    async def review_with_telemetry(
        self,
        review_input: AIReviewInput,
    ) -> AIProviderExecutionResult: ...


class AIReviewProvider(Protocol):
    """
    Interface for providers that generate AI-powered code reviews.

    Implementations must accept a validated review input and return
    a structured AI review result.
    """

    async def review(
        self,
        review_input: AIReviewInput,
    ) -> AIReviewResult:
        """
        Analyze bounded repository context and generate review candidates.

        Repository content must be treated as untrusted data.
        Returned findings remain unverified until independently validated.
        """
        ...


class AIReviewProviderExecutionIdentity(AIReviewProvider, Protocol):
    """Durable execution identity declared by a concrete provider selection."""

    @property
    def execution_metadata(self) -> AIExecutionMetadata: ...


@runtime_checkable
class ClosableAIReviewProvider(Protocol):
    """Optional lifecycle interface for providers that own external clients."""

    async def aclose(self) -> None: ...
