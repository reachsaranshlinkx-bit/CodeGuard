import pytest

from codeguard.ai.contracts import (
    AIFindingCandidate,
    AIReviewInput,
    AIReviewResult,
)
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
async def test_mock_provider_returns_default_empty_result() -> None:
    provider = MockAIReviewProvider()
    review_input = make_review_input()

    result = await provider.review(review_input)

    assert isinstance(result, AIReviewResult)
    assert result.summary == "Mock AI review completed with no findings."
    assert result.findings == ()
    assert provider.reviewed_inputs == (review_input,)


@pytest.mark.asyncio
async def test_mock_provider_returns_configured_result() -> None:
    configured_result = make_review_result()
    provider = MockAIReviewProvider(result=configured_result)

    result = await provider.review(make_review_input())

    assert result is configured_result
    assert len(result.findings) == 1


@pytest.mark.asyncio
async def test_mock_provider_records_multiple_review_inputs() -> None:
    provider = MockAIReviewProvider()
    review_input = make_review_input()

    await provider.review(review_input)
    await provider.review(review_input)

    assert provider.reviewed_inputs == (review_input, review_input)


@pytest.mark.asyncio
async def test_mock_provider_raises_configured_error() -> None:
    error = RuntimeError("Simulated provider failure.")
    provider = MockAIReviewProvider(error=error)
    review_input = make_review_input()

    with pytest.raises(RuntimeError, match="Simulated provider failure"):
        await provider.review(review_input)

    assert provider.reviewed_inputs == (review_input,)


def test_mock_provider_rejects_result_and_error_together() -> None:
    with pytest.raises(ValueError, match="both a result and an error"):
        MockAIReviewProvider(
            result=make_review_result(),
            error=RuntimeError("Simulated failure."),
        )
