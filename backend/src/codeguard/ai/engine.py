from codeguard.ai.contracts import AIReviewInput, AIReviewResult
from codeguard.ai.provider import AIReviewProvider


class AIReviewEngine:
    """
    Coordinate structured AI code review execution.

    The engine delegates review generation to an injected provider
    and independently validates its structured response.

    Provider errors are allowed to propagate. Retry policies and
    failure recovery belong to a separate reliability layer.
    """

    def __init__(self, provider: AIReviewProvider) -> None:
        self._provider = provider

    async def review(self, review_input: AIReviewInput) -> AIReviewResult:
        """
        Execute a review and validate the provider's structured result.
        """

        provider_result = await self._provider.review(review_input)

        if not isinstance(provider_result, AIReviewResult):
            raise TypeError(
                "AI review provider must return an AIReviewResult instance."
            )

        return AIReviewResult.model_validate(provider_result.model_dump(mode="python"))
