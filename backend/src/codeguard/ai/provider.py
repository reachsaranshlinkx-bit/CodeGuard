from typing import Protocol, runtime_checkable

from codeguard.ai.contracts import AIReviewInput, AIReviewResult
from codeguard.ai.execution_metadata import AIExecutionMetadata


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
