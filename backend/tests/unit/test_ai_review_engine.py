from typing import cast

import pytest
from pydantic import ValidationError

from codeguard.ai.contracts import (
    AIFindingCandidate,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.engine import AIReviewEngine
from codeguard.ai.evidence_matching import EvidenceMatchStatus
from codeguard.ai.evidence_validation import EvidenceLocationReason
from codeguard.ai.mock_provider import MockAIReviewProvider
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)

COMMIT_SHA = "a" * 40


def make_review_input() -> AIReviewInput:
    return AIReviewInput(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        head_sha=COMMIT_SHA,
        context="Changed authentication function and related callers.",
    )


def make_review_result(
    evidence: str = "def get_account(account_id):",
) -> AIReviewResult:
    candidate = AIFindingCandidate(
        title="Missing authentication check",
        severity="high",
        confidence="medium",
        file_path="src/auth.py",
        line_number=25,
        problem="The endpoint does not verify user authorization.",
        evidence=evidence,
        why_it_matters="Unauthorized users may access another account.",
        suggested_fix="Check account ownership before returning the account.",
        recommended_validation="Test authorized and unauthorized users.",
    )

    return AIReviewResult(
        summary="One potential authorization issue was identified.",
        findings=(candidate,),
    )


def make_source_snapshot(
    commit_sha: str = COMMIT_SHA,
) -> RepositorySourceSnapshot:
    source_lines = ["# context"] * 24 + [
        "def get_account(account_id):",
        "    return account_id",
    ]
    source = "\n".join(source_lines) + "\n"

    discovered_file = DiscoveredPythonFile(
        path="src/auth.py",
        sha="b" * 40,
        size_bytes=len(source.encode("utf-8")),
    )

    return RepositorySourceSnapshot(
        commit_sha=commit_sha,
        files=(RepositorySourceFile(discovered_file, source),),
    )


@pytest.mark.asyncio
async def test_engine_returns_verified_provider_result() -> None:
    review_input = make_review_input()
    configured_result = make_review_result()

    provider = MockAIReviewProvider(result=configured_result)
    engine = AIReviewEngine(provider)

    result = await engine.review(
        review_input,
        make_source_snapshot(),
    )

    assert result.provider_result == configured_result
    assert result.provider_result is not configured_result
    assert len(result.assessments) == 1
    assert result.assessments[0].candidate == configured_result.findings[0]
    assert result.assessments[0].evidence_match.status == EvidenceMatchStatus.MATCHED
    assert provider.reviewed_inputs == (review_input,)


@pytest.mark.asyncio
async def test_engine_accepts_empty_provider_result() -> None:
    provider = MockAIReviewProvider()
    engine = AIReviewEngine(provider)

    result = await engine.review(
        make_review_input(),
        make_source_snapshot(),
    )

    assert result.provider_result.findings == ()
    assert result.assessments == ()


@pytest.mark.asyncio
async def test_engine_propagates_provider_errors() -> None:
    error = RuntimeError("Mock provider unavailable.")

    provider = MockAIReviewProvider(error=error)
    engine = AIReviewEngine(provider)

    with pytest.raises(RuntimeError, match="Mock provider unavailable"):
        await engine.review(
            make_review_input(),
            make_source_snapshot(),
        )


@pytest.mark.asyncio
async def test_engine_rejects_invalid_provider_return_type() -> None:
    class InvalidProvider:
        async def review(self, review_input: AIReviewInput) -> AIReviewResult:
            return cast(
                AIReviewResult,
                {"summary": "Invalid result", "findings": []},
            )

    engine = AIReviewEngine(InvalidProvider())

    with pytest.raises(TypeError, match="AIReviewResult"):
        await engine.review(
            make_review_input(),
            make_source_snapshot(),
        )


@pytest.mark.asyncio
async def test_engine_revalidates_provider_result_contents() -> None:
    invalid_result = AIReviewResult.model_construct(
        summary="",
        findings=(),
    )

    provider = MockAIReviewProvider(result=invalid_result)
    engine = AIReviewEngine(provider)

    with pytest.raises(ValidationError):
        await engine.review(
            make_review_input(),
            make_source_snapshot(),
        )


@pytest.mark.asyncio
async def test_engine_marks_incorrect_evidence_as_mismatched() -> None:
    provider = MockAIReviewProvider(
        result=make_review_result(evidence="    return False"),
    )
    engine = AIReviewEngine(provider)

    result = await engine.review(
        make_review_input(),
        make_source_snapshot(),
    )

    assert len(result.assessments) == 1
    assert result.assessments[0].evidence_match.status == EvidenceMatchStatus.MISMATCHED


@pytest.mark.asyncio
async def test_engine_marks_different_snapshot_commit_unverifiable() -> None:
    provider = MockAIReviewProvider(result=make_review_result())
    engine = AIReviewEngine(provider)

    result = await engine.review(
        make_review_input(),
        make_source_snapshot(commit_sha="c" * 40),
    )

    assert len(result.assessments) == 1

    assessment = result.assessments[0]

    assert assessment.evidence_match.status == EvidenceMatchStatus.UNVERIFIABLE
    assert assessment.evidence_match.location_reason == (
        EvidenceLocationReason.COMMIT_MISMATCH
    )


@pytest.mark.asyncio
async def test_engine_assesses_each_provider_finding() -> None:
    matched_candidate = make_review_result().findings[0]

    mismatched_candidate = make_review_result(
        evidence="    return False",
    ).findings[0]

    configured_result = AIReviewResult(
        summary="Two potential findings were identified.",
        findings=(matched_candidate, mismatched_candidate),
    )

    provider = MockAIReviewProvider(result=configured_result)
    engine = AIReviewEngine(provider)

    result = await engine.review(
        make_review_input(),
        make_source_snapshot(),
    )

    assert len(result.provider_result.findings) == 2
    assert len(result.assessments) == 2

    assert result.assessments[0].candidate == matched_candidate
    assert result.assessments[0].evidence_match.status == (EvidenceMatchStatus.MATCHED)

    assert result.assessments[1].candidate == mismatched_candidate
    assert result.assessments[1].evidence_match.status == (
        EvidenceMatchStatus.MISMATCHED
    )
