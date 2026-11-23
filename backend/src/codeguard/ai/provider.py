from typing import Protocol

from codeguard.ai.contracts import AIReviewInput, AIReviewResult


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
