from typing import cast

import pytest
from pydantic import ValidationError

from codeguard.ai.contracts import (
    AIFindingCandidate,
    AIReviewInput,
    AIReviewResult,
)
from codeguard.ai.engine import AIReviewEngine
from codeguard.ai.mock_provider import MockAIReviewProvider


def make_review_input() -> AIReviewInput:
    return AIReviewInput(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        head_sha="abc123",
        context="Changed authentication function and related callers.",
    )


def make_review_result() -> AIReviewResult:
    candidate = AIFindingCandidate(
        title="Missing authentication check",
        severity="high",
        confidence="medium",
        file_path="src/auth.py",
        line_number=25,
        problem="The endpoint does not verify user authorization.",
        evidence="def get_account(account_id):",
        why_it_matters="Unauthorized users may access another account.",
        suggested_fix="Check account ownership before returning the account.",
        recommended_validation="Test authorized and unauthorized users.",
    )

    return AIReviewResult(
        summary="One potential authorization issue was identified.",
        findings=(candidate,),
    )


@pytest.mark.asyncio
async def test_engine_returns_validated_provider_result() -> None:
    review_input = make_review_input()
    configured_result = make_review_result()

    provider = MockAIReviewProvider(result=configured_result)
    engine = AIReviewEngine(provider)

    result = await engine.review(review_input)

    assert result == configured_result
    assert result is not configured_result
    assert provider.reviewed_inputs == (review_input,)


@pytest.mark.asyncio
async def test_engine_accepts_empty_provider_result() -> None:
    provider = MockAIReviewProvider()
    engine = AIReviewEngine(provider)

    result = await engine.review(make_review_input())

    assert isinstance(result, AIReviewResult)
    assert result.findings == ()


@pytest.mark.asyncio
async def test_engine_propagates_provider_errors() -> None:
    error = RuntimeError("Mock provider unavailable.")

    provider = MockAIReviewProvider(error=error)
    engine = AIReviewEngine(provider)

    with pytest.raises(RuntimeError, match="Mock provider unavailable"):
        await engine.review(make_review_input())


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
        await engine.review(make_review_input())


@pytest.mark.asyncio
async def test_engine_revalidates_provider_result_contents() -> None:
    invalid_result = AIReviewResult.model_construct(
        summary="",
        findings=(),
    )

    provider = MockAIReviewProvider(result=invalid_result)
    engine = AIReviewEngine(provider)

    with pytest.raises(ValidationError):
        await engine.review(make_review_input())
