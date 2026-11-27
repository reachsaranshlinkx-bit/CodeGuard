from dataclasses import dataclass
from time import perf_counter_ns
from typing import cast

from codeguard.ai.contracts import (
    AIFindingCandidate,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatcher,
    EvidenceMatchResult,
)
from codeguard.ai.provider import (
    AIProviderExecutionResult,
    AIReviewProvider,
    InstrumentedAIReviewProvider,
)
from codeguard.ai.telemetry import AIExecutionTelemetry
from codeguard.context.source_snapshot import RepositorySourceSnapshot


@dataclass(frozen=True, slots=True)
class AIReviewFindingAssessment:
    """Original AI finding paired with its independent evidence assessment."""

    candidate: AIFindingCandidate
    evidence_match: EvidenceMatchResult


@dataclass(frozen=True, slots=True)
class AIReviewExecutionResult:
    """Validated provider result and independently assessed findings."""

    provider_result: AIReviewResult
    assessments: tuple[AIReviewFindingAssessment, ...]
    telemetry: AIExecutionTelemetry | None = None


class AIReviewEngine:
    """
    Coordinate AI review generation and independent evidence verification.

    The engine delegates review generation to an injected provider,
    validates its structured response, and independently verifies
    every finding against a repository source snapshot.

    Provider errors are allowed to propagate. Retry policies and
    failure recovery belong to a separate reliability layer.
    """

    def __init__(
        self,
        provider: AIReviewProvider,
        *,
        evidence_matcher: EvidenceMatcher | None = None,
    ) -> None:
        self._provider = provider
        self._evidence_matcher = (
            evidence_matcher if evidence_matcher is not None else EvidenceMatcher()
        )

    async def review(
        self,
        review_input: AIReviewInput,
        source_snapshot: RepositorySourceSnapshot,
    ) -> AIReviewExecutionResult:
        """
        Generate a review and independently assess every finding.

        The expected commit SHA comes from the validated review input.
        Findings are preserved even when their evidence cannot be verified.
        """

        execution_started_ns = perf_counter_ns()
        instrumented = isinstance(self._provider, InstrumentedAIReviewProvider)
        provider_execution: AIProviderExecutionResult | None = None
        if instrumented:
            instrumented_provider = cast(
                InstrumentedAIReviewProvider,
                self._provider,
            )
            provider_execution = await instrumented_provider.review_with_telemetry(
                review_input
            )
            if not isinstance(provider_execution, AIProviderExecutionResult):
                raise TypeError(
                    "Instrumented AI provider must return an "
                    "AIProviderExecutionResult instance."
                )
            provider_result = provider_execution.review_result
        else:
            provider_result = await self._provider.review(review_input)

        if not isinstance(provider_result, AIReviewResult):
            raise TypeError(
                "AI review provider must return an AIReviewResult instance."
            )

        validated_result = AIReviewResult.model_validate(
            provider_result.model_dump(mode="python")
        )

        assessments = tuple(
            AIReviewFindingAssessment(
                candidate=candidate,
                evidence_match=self._evidence_matcher.match(
                    candidate,
                    source_snapshot,
                    expected_commit_sha=review_input.head_sha,
                ),
            )
            for candidate in validated_result.findings
        )

        execution_finished_ns = perf_counter_ns()
        telemetry = AIExecutionTelemetry(
            provider_identity=(
                provider_execution.provider_identity
                if provider_execution is not None
                else None
            ),
            model_identity=(
                provider_execution.model_identity
                if provider_execution is not None
                else None
            ),
            provider_usage=(
                provider_execution.usage if provider_execution is not None else None
            ),
            provider_request_duration_ms=(
                provider_execution.request_duration_ms
                if provider_execution is not None
                else None
            ),
            total_ai_execution_duration_ms=(
                execution_finished_ns - execution_started_ns
            )
            / 1_000_000,
            measurement_source="provider" if instrumented else "runtime",
        )
        return AIReviewExecutionResult(
            provider_result=validated_result,
            assessments=assessments,
            telemetry=telemetry,
        )
