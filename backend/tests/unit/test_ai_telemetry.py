from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from codeguard.ai.telemetry import (
    AIExecutionTelemetry,
    AIPricingConfiguration,
    AIProviderUsage,
    aggregate_ai_telemetry,
)


def _pricing(
    *,
    input_price: str = "2",
    output_price: str = "4",
    cached_input_price: str | None = "0.5",
) -> AIPricingConfiguration:
    return AIPricingConfiguration(
        provider_identity="provider-x",
        model_identity="model-exact",
        input_price_per_million_tokens=Decimal(input_price),
        output_price_per_million_tokens=Decimal(output_price),
        cached_input_price_per_million_tokens=(
            Decimal(cached_input_price) if cached_input_price is not None else None
        ),
        currency="USD",
        pricing_version="fixture-v1",
        effective_date=date(2026, 1, 1),
        pricing_source="test fixture",
    )


def _telemetry(
    *,
    duration: float | None,
    usage: AIProviderUsage | None = None,
    source: str = "provider",
) -> AIExecutionTelemetry:
    return AIExecutionTelemetry(
        provider_identity="provider-x",
        model_identity="model-exact",
        provider_usage=usage,
        provider_request_duration_ms=duration,
        total_ai_execution_duration_ms=duration,
        measurement_source=source,
    )


def test_provider_usage_requires_non_negative_integer_counts() -> None:
    usage = AIProviderUsage(
        input_tokens=12,
        output_tokens=7,
        total_tokens=19,
        cached_input_tokens=3,
    )

    assert usage.total_tokens == 19
    with pytest.raises(ValidationError):
        AIProviderUsage(
            input_tokens=-1,
            output_tokens=1,
            total_tokens=0,
        )
    with pytest.raises(ValidationError):
        AIProviderUsage(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cached_input_tokens=2,
        )
    with pytest.raises(ValidationError):
        AIProviderUsage(
            input_tokens=True,
            output_tokens=1,
            total_tokens=2,
        )


@pytest.mark.parametrize("duration", [-0.1, float("nan"), float("inf"), True])
def test_invalid_durations_are_rejected(duration: float) -> None:
    with pytest.raises(ValidationError):
        AIExecutionTelemetry(provider_request_duration_ms=duration)


def test_missing_measurements_remain_unavailable() -> None:
    telemetry = AIExecutionTelemetry()

    assert telemetry.provider_usage is None
    assert telemetry.provider_request_duration_ms is None
    assert telemetry.total_ai_execution_duration_ms is None
    assert telemetry.queue_waiting_duration_ms is None


def test_aggregation_excludes_missing_durations_and_reports_sample_counts() -> None:
    performance, tokens, cost = aggregate_ai_telemetry(
        (
            _telemetry(duration=100, source="provider"),
            _telemetry(duration=200, source="synthetic"),
            _telemetry(duration=300, source="runtime"),
            _telemetry(duration=None),
        ),
        pricing=None,
        reviewed_pr_count=4,
        verified_finding_count=0,
    )

    assert performance.provider_request.sample_count == 3
    assert performance.provider_request.mean_duration_ms == 200
    assert performance.provider_request.p50_duration_ms == 200
    assert performance.provider_request.p95_duration_ms == 300
    assert performance.provider_request.maximum_duration_ms == 300
    assert performance.total_execution.sample_count == 3
    assert performance.measurement_sources == ("provider", "runtime", "synthetic")
    assert performance.queue_waiting.sample_count == 0
    assert tokens.sample_count == 0
    assert tokens.total_tokens is None
    assert tokens.average_tokens_per_pr is None
    assert cost.estimated_total_cost is None
    assert cost.unavailable_reason == "pricing_not_configured"


def test_token_and_cost_aggregation_uses_exact_model_pricing() -> None:
    usage = AIProviderUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        total_tokens=1_500_000,
        cached_input_tokens=200_000,
    )

    _, token_metrics, cost = aggregate_ai_telemetry(
        (_telemetry(duration=10, usage=usage),),
        pricing=_pricing(),
        reviewed_pr_count=1,
        verified_finding_count=2,
    )

    assert token_metrics.total_input_tokens == 1_000_000
    assert token_metrics.total_output_tokens == 500_000
    assert token_metrics.total_tokens == 1_500_000
    assert token_metrics.average_tokens_per_pr == 1_500_000
    assert cost.sample_count == 1
    assert cost.pricing_version == "fixture-v1"
    assert cost.estimated_input_cost == Decimal("1.60000000")
    assert cost.estimated_cached_input_cost == Decimal("0.10000000")
    assert cost.estimated_output_cost == Decimal("2.00000000")
    assert cost.estimated_total_cost == Decimal("3.70000000")
    assert cost.average_cost_per_reviewed_pr == Decimal("3.70000000")
    assert cost.estimated_cost_per_verified_finding == Decimal("1.85000000")


def test_unknown_model_pricing_and_missing_cache_price_are_unavailable() -> None:
    usage = AIProviderUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=1,
    )
    wrong_model = _telemetry(duration=1, usage=usage).model_copy(
        update={"model_identity": "another-model"}
    )
    _, _, unmatched = aggregate_ai_telemetry(
        (wrong_model,),
        pricing=_pricing(),
        reviewed_pr_count=1,
        verified_finding_count=1,
    )
    _, _, missing_cache_price = aggregate_ai_telemetry(
        (_telemetry(duration=1, usage=usage),),
        pricing=_pricing(cached_input_price=None),
        reviewed_pr_count=1,
        verified_finding_count=1,
    )

    assert unmatched.estimated_total_cost is None
    assert unmatched.unavailable_reason == "model_unmatched"
    assert missing_cache_price.estimated_total_cost is None
    assert missing_cache_price.unavailable_reason == "cached_pricing_unavailable"


def test_cost_rounding_uses_decimal_half_up_and_zero_findings_is_undefined() -> None:
    usage = AIProviderUsage(input_tokens=1, output_tokens=0, total_tokens=1)
    _, _, cost = aggregate_ai_telemetry(
        (_telemetry(duration=1, usage=usage),),
        pricing=_pricing(input_price="0.005", output_price="0"),
        reviewed_pr_count=1,
        verified_finding_count=0,
    )

    assert cost.estimated_input_cost == Decimal("0.00000001")
    assert cost.estimated_total_cost == Decimal("0.00000001")
    assert cost.estimated_cost_per_verified_finding is None


def test_partial_token_usage_does_not_invent_full_run_cost() -> None:
    usage = AIProviderUsage(input_tokens=3, output_tokens=2, total_tokens=5)

    _, _, cost = aggregate_ai_telemetry(
        (_telemetry(duration=1, usage=usage), None),
        pricing=_pricing(),
        reviewed_pr_count=2,
        verified_finding_count=1,
    )

    assert cost.estimated_total_cost is None
    assert cost.unavailable_reason == "usage_unavailable"


def test_provider_usage_rejects_inconsistent_total_tokens() -> None:
    with pytest.raises(
        ValidationError,
        match="Total tokens must equal input plus output tokens",
    ):
        AIProviderUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1400,
        )
