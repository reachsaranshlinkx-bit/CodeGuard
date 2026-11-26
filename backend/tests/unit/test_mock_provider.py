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

    assert result == configured_result
    assert result is not configured_result
    assert len(result.findings) == 1


@pytest.mark.asyncio
async def test_original_result_mutation_does_not_change_provider_output() -> None:
    configured_result = make_review_result()
    provider = MockAIReviewProvider(result=configured_result)

    object.__setattr__(configured_result, "summary", "Mutated original result.")
    object.__setattr__(configured_result.findings[0], "title", "Mutated finding.")

    result = await provider.review(make_review_input())

    assert result.summary == "One potential authorization issue was identified."
    assert result.findings[0].title == "Missing authentication check"


@pytest.mark.asyncio
async def test_mutating_returned_result_does_not_change_subsequent_reviews() -> None:
    provider = MockAIReviewProvider(result=make_review_result())

    first_result = await provider.review(make_review_input())
    object.__setattr__(first_result, "summary", "Mutated returned result.")
    object.__setattr__(first_result.findings[0], "title", "Mutated finding.")

    second_result = await provider.review(make_review_input())

    assert second_result.summary == "One potential authorization issue was identified."
    assert second_result.findings[0].title == "Missing authentication check"
    assert second_result is not first_result


@pytest.mark.asyncio
async def test_configured_profiles_remain_independent() -> None:
    first_provider = MockAIReviewProvider(
        result=AIReviewResult(summary="Profile one result.", findings=()),
        profile_identity="mock-profile:profile-one-v1",
    )
    second_provider = MockAIReviewProvider(
        result=AIReviewResult(summary="Profile two result.", findings=()),
        profile_identity="mock-profile:profile-two-v1",
    )

    first_result = await first_provider.review(make_review_input())
    second_result = await second_provider.review(make_review_input())

    assert first_provider.profile_identity == "mock-profile:profile-one-v1"
    assert second_provider.profile_identity == "mock-profile:profile-two-v1"
    assert first_result.summary == "Profile one result."
    assert second_result.summary == "Profile two result."


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
