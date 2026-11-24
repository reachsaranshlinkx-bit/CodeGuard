from dataclasses import dataclass

from codeguard.ai.contracts import (
    AIFindingCandidate,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatcher,
    EvidenceMatchResult,
)
from codeguard.ai.provider import AIReviewProvider
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

        return AIReviewExecutionResult(
            provider_result=validated_result,
            assessments=assessments,
        )
