"""Provider-neutral AI execution telemetry and deterministic cost aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from math import ceil, isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MeasurementSource = Literal["provider", "synthetic", "runtime", "unknown"]
_COST_QUANTUM = Decimal("0.00000001")


class _TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AIProviderUsage(_TelemetryModel):
    """Token counts returned by a provider for one completed request."""

    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)
    cached_input_tokens: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_cached_tokens(
        self: AIProviderUsage,
    ) -> AIProviderUsage:
        if (
            self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("Cached input tokens cannot exceed input tokens.")

        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Total tokens must equal input plus output tokens.")

        return self


@model_validator(mode="after")
def validate_cached_tokens(
    self: AIProviderUsage,
) -> AIProviderUsage:
    if (
        self.cached_input_tokens is not None
        and self.cached_input_tokens > self.input_tokens
    ):
        raise ValueError("Cached input tokens cannot exceed input tokens.")

    if self.total_tokens != self.input_tokens + self.output_tokens:
        raise ValueError("Total tokens must equal input plus output tokens.")

    return self


class AIExecutionTelemetry(_TelemetryModel):
    """Measured values for one execution; absent values remain explicitly unknown.

    Provider request duration covers only the provider HTTP request/response.
    Total execution duration covers provider review plus evidence assessment.
    Queue waiting is populated only when enqueue and worker-start boundaries are
    available from a trustworthy lifecycle clock.
    """

    provider_identity: str | None = None
    model_identity: str | None = None
    provider_usage: AIProviderUsage | None = None
    provider_request_duration_ms: float | None = None
    total_ai_execution_duration_ms: float | None = None
    queue_waiting_duration_ms: float | None = None
    measurement_source: MeasurementSource = "unknown"

    @field_validator(
        "provider_request_duration_ms",
        "total_ai_execution_duration_ms",
        "queue_waiting_duration_ms",
        mode="before",
    )
    @classmethod
    def validate_duration(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Durations must be finite non-negative numbers.")
        if not isfinite(value) or value < 0:
            raise ValueError("Durations must be finite non-negative numbers.")
        return float(value)

    @field_validator("provider_identity", "model_identity")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Provider and model identities must not be blank.")
        return value.strip() if value is not None else None


class AIPricingConfiguration(_TelemetryModel):
    """Explicit, versioned per-million-token prices for an exact provider/model."""

    provider_identity: str = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    input_price_per_million_tokens: Decimal = Field(ge=0)
    output_price_per_million_tokens: Decimal = Field(ge=0)
    cached_input_price_per_million_tokens: Decimal | None = Field(
        default=None,
        ge=0,
    )
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    pricing_version: str = Field(min_length=1)
    effective_date: date
    pricing_source: str = Field(min_length=1)

    @field_validator(
        "input_price_per_million_tokens",
        "output_price_per_million_tokens",
        "cached_input_price_per_million_tokens",
    )
    @classmethod
    def validate_prices(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("Token prices must be finite and non-negative.")
        return value

    @field_validator(
        "provider_identity",
        "model_identity",
        "pricing_version",
        "pricing_source",
    )
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pricing identity and source fields must not be blank.")
        return value.strip()


class AIDurationMetrics(_TelemetryModel):
    """Distribution summary for the measured duration sample only."""

    sample_count: int = Field(ge=0, strict=True)
    mean_duration_ms: float | None = Field(default=None, ge=0)
    p50_duration_ms: float | None = Field(default=None, ge=0)
    p95_duration_ms: float | None = Field(default=None, ge=0)
    maximum_duration_ms: float | None = Field(default=None, ge=0)

    @field_validator(
        "mean_duration_ms",
        "p50_duration_ms",
        "p95_duration_ms",
        "maximum_duration_ms",
    )
    @classmethod
    def validate_finite_duration(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("Aggregated durations must be finite.")
        return value

    @model_validator(mode="after")
    def validate_sample_statistics(self) -> AIDurationMetrics:
        values = (
            self.mean_duration_ms,
            self.p50_duration_ms,
            self.p95_duration_ms,
            self.maximum_duration_ms,
        )
        if (self.sample_count == 0 and any(value is not None for value in values)) or (
            self.sample_count > 0 and any(value is None for value in values)
        ):
            raise ValueError("Duration statistics must match their sample count.")
        return self


class AIPerformanceMetrics(_TelemetryModel):
    """Timing aggregates with separate provider and full-execution boundaries."""

    provider_identities: tuple[str, ...]
    model_identities: tuple[str, ...]
    provider_request: AIDurationMetrics
    total_execution: AIDurationMetrics
    queue_waiting: AIDurationMetrics
    measurement_sources: tuple[MeasurementSource, ...]


class AITokenUsageMetrics(_TelemetryModel):
    """Token totals and averages across PRs with actual token measurements."""

    sample_count: int = Field(ge=0, strict=True)
    total_input_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_output_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, strict=True)
    average_input_tokens_per_pr: float | None = Field(default=None, ge=0)
    average_output_tokens_per_pr: float | None = Field(default=None, ge=0)
    average_tokens_per_pr: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_sample_statistics(self) -> AITokenUsageMetrics:
        values = (
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_tokens,
            self.average_input_tokens_per_pr,
            self.average_output_tokens_per_pr,
            self.average_tokens_per_pr,
        )
        if (self.sample_count == 0 and any(value is not None for value in values)) or (
            self.sample_count > 0 and any(value is None for value in values)
        ):
            raise ValueError("Token statistics must match their sample count.")
        return self


class AICostMetrics(_TelemetryModel):
    """Estimated token costs; values are unavailable without matching pricing."""

    pricing_version: str | None
    currency: str | None
    sample_count: int = Field(ge=0, strict=True)
    estimated_input_cost: Decimal | None
    estimated_cached_input_cost: Decimal | None
    estimated_output_cost: Decimal | None
    estimated_total_cost: Decimal | None
    average_cost_per_reviewed_pr: Decimal | None
    estimated_cost_per_verified_finding: Decimal | None
    unavailable_reason: (
        Literal[
            "pricing_not_configured",
            "model_unmatched",
            "usage_unavailable",
            "cached_pricing_unavailable",
        ]
        | None
    )

    @field_validator(
        "estimated_input_cost",
        "estimated_cached_input_cost",
        "estimated_output_cost",
        "estimated_total_cost",
        "average_cost_per_reviewed_pr",
        "estimated_cost_per_verified_finding",
    )
    @classmethod
    def validate_costs(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("Cost values must be finite and non-negative.")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> AICostMetrics:
        values = (
            self.estimated_input_cost,
            self.estimated_cached_input_cost,
            self.estimated_output_cost,
            self.estimated_total_cost,
            self.average_cost_per_reviewed_pr,
            self.estimated_cost_per_verified_finding,
        )
        if self.unavailable_reason is None:
            if self.sample_count == 0 or any(value is None for value in values[:5]):
                raise ValueError("Available costs require measured samples and totals.")
        elif self.sample_count != 0 or any(value is not None for value in values):
            raise ValueError("Unavailable costs cannot contain fabricated values.")
        return self


def _duration_metrics(values: Iterable[float]) -> AIDurationMetrics:
    ordered = sorted(values)
    if not ordered:
        return AIDurationMetrics(
            sample_count=0,
            mean_duration_ms=None,
            p50_duration_ms=None,
            p95_duration_ms=None,
            maximum_duration_ms=None,
        )

    def nearest_rank(percentile: float) -> float:
        return ordered[ceil(percentile * len(ordered)) - 1]

    return AIDurationMetrics(
        sample_count=len(ordered),
        mean_duration_ms=sum(value / len(ordered) for value in ordered),
        p50_duration_ms=nearest_rank(0.50),
        p95_duration_ms=nearest_rank(0.95),
        maximum_duration_ms=ordered[-1],
    )


def _rounded_cost(value: Decimal) -> Decimal:
    return value.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)


def aggregate_ai_telemetry(
    telemetry: Iterable[AIExecutionTelemetry | None],
    *,
    pricing: AIPricingConfiguration | None,
    reviewed_pr_count: int,
    verified_finding_count: int,
) -> tuple[AIPerformanceMetrics, AITokenUsageMetrics, AICostMetrics]:
    """Aggregate telemetry deterministically; duration percentiles use nearest rank.

    Provider measurements and synthetic/runtime samples retain their source labels.
    Costs are calculated only for token measurements whose exact provider and
    model identities match the supplied pricing configuration.
    """

    for label, count in (
        ("reviewed_pr_count", reviewed_pr_count),
        ("verified_finding_count", verified_finding_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label} must be a non-negative integer.")

    records = tuple(item for item in telemetry if item is not None)
    provider_durations = tuple(
        item.provider_request_duration_ms
        for item in records
        if item.provider_request_duration_ms is not None
    )
    execution_durations = tuple(
        item.total_ai_execution_duration_ms
        for item in records
        if item.total_ai_execution_duration_ms is not None
    )
    queue_durations = tuple(
        item.queue_waiting_duration_ms
        for item in records
        if item.queue_waiting_duration_ms is not None
    )
    performance = AIPerformanceMetrics(
        provider_identities=tuple(
            sorted(
                {
                    item.provider_identity
                    for item in records
                    if item.provider_identity is not None
                }
            )
        ),
        model_identities=tuple(
            sorted(
                {
                    item.model_identity
                    for item in records
                    if item.model_identity is not None
                }
            )
        ),
        provider_request=_duration_metrics(provider_durations),
        total_execution=_duration_metrics(execution_durations),
        queue_waiting=_duration_metrics(queue_durations),
        measurement_sources=tuple(
            sorted({item.measurement_source for item in records})
        ),
    )

    usages = tuple(item.provider_usage for item in records if item.provider_usage)
    input_total = sum(usage.input_tokens for usage in usages)
    output_total = sum(usage.output_tokens for usage in usages)
    total_tokens = sum(usage.total_tokens for usage in usages)
    token_count = len(usages)
    token_metrics = AITokenUsageMetrics(
        sample_count=token_count,
        total_input_tokens=input_total if token_count else None,
        total_output_tokens=output_total if token_count else None,
        total_tokens=total_tokens if token_count else None,
        average_input_tokens_per_pr=input_total / token_count if token_count else None,
        average_output_tokens_per_pr=(
            output_total / token_count if token_count else None
        ),
        average_tokens_per_pr=total_tokens / token_count if token_count else None,
    )

    if pricing is None:
        return (
            performance,
            token_metrics,
            _unavailable_cost(None, "pricing_not_configured"),
        )

    priced_records = tuple(
        item
        for item in records
        if item.provider_usage is not None
        and item.provider_identity == pricing.provider_identity
        and item.model_identity == pricing.model_identity
    )
    unmatched_usage = any(
        item.provider_usage is not None
        and (
            item.provider_identity != pricing.provider_identity
            or item.model_identity != pricing.model_identity
        )
        for item in records
    )
    if unmatched_usage:
        return performance, token_metrics, _unavailable_cost(pricing, "model_unmatched")
    if not priced_records or len(priced_records) != reviewed_pr_count:
        return (
            performance,
            token_metrics,
            _unavailable_cost(pricing, "usage_unavailable"),
        )
    if pricing.cached_input_price_per_million_tokens is None and any(
        item.provider_usage is not None
        and item.provider_usage.cached_input_tokens not in (None, 0)
        for item in priced_records
    ):
        return (
            performance,
            token_metrics,
            _unavailable_cost(pricing, "cached_pricing_unavailable"),
        )

    million = Decimal(1_000_000)
    input_cost = Decimal(0)
    cached_cost = Decimal(0)
    output_cost = Decimal(0)
    for item in priced_records:
        usage = item.provider_usage
        if usage is None:
            continue
        cached_tokens = usage.cached_input_tokens or 0
        noncached_tokens = usage.input_tokens - cached_tokens
        input_cost += (
            Decimal(noncached_tokens) * pricing.input_price_per_million_tokens / million
        )
        if cached_tokens and pricing.cached_input_price_per_million_tokens is not None:
            cached_cost += (
                Decimal(cached_tokens)
                * pricing.cached_input_price_per_million_tokens
                / million
            )
        output_cost += (
            Decimal(usage.output_tokens)
            * pricing.output_price_per_million_tokens
            / million
        )

    total_cost = _rounded_cost(input_cost + cached_cost + output_cost)
    return (
        performance,
        token_metrics,
        AICostMetrics(
            pricing_version=pricing.pricing_version,
            currency=pricing.currency,
            sample_count=len(priced_records),
            estimated_input_cost=_rounded_cost(input_cost),
            estimated_cached_input_cost=_rounded_cost(cached_cost),
            estimated_output_cost=_rounded_cost(output_cost),
            estimated_total_cost=total_cost,
            average_cost_per_reviewed_pr=(
                _rounded_cost(total_cost / Decimal(reviewed_pr_count))
                if reviewed_pr_count
                else None
            ),
            estimated_cost_per_verified_finding=(
                _rounded_cost(total_cost / Decimal(verified_finding_count))
                if verified_finding_count > 0
                else None
            ),
            unavailable_reason=None,
        ),
    )


def _unavailable_cost(
    pricing: AIPricingConfiguration | None,
    reason: Literal[
        "pricing_not_configured",
        "model_unmatched",
        "usage_unavailable",
        "cached_pricing_unavailable",
    ],
) -> AICostMetrics:
    return AICostMetrics(
        pricing_version=pricing.pricing_version if pricing else None,
        currency=pricing.currency if pricing else None,
        sample_count=0,
        estimated_input_cost=None,
        estimated_cached_input_cost=None,
        estimated_output_cost=None,
        estimated_total_cost=None,
        average_cost_per_reviewed_pr=None,
        estimated_cost_per_verified_finding=None,
        unavailable_reason=reason,
    )
