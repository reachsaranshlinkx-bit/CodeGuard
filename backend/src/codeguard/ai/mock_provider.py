from codeguard.ai.contracts import AIReviewInput, AIReviewResult


class MockAIReviewProvider:
    """
    Deterministic provider implementation for development and testing.

    It performs no network requests and never invokes an LLM.
    """

    def __init__(
        self,
        result: AIReviewResult | None = None,
        error: Exception | None = None,
    ) -> None:
        if result is not None and error is not None:
            raise ValueError("A mock provider cannot have both a result and an error.")

        self._result = (
            result
            if result is not None
            else AIReviewResult(
                summary="Mock AI review completed with no findings.",
                findings=(),
            )
        )

        self._error = error
        self._reviewed_inputs: list[AIReviewInput] = []

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

        return self._result
