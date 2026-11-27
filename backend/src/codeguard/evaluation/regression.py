"""Policy-driven comparisons between compatible evaluation reports.

Every enforced metric must have explicit policy thresholds. A minimum applies
to the candidate value; a maximum absolute drop applies only when the candidate
is lower than the baseline. ``absolute_change`` is signed as candidate minus
baseline. Undefined values are either reported as inconclusive or rejected,
according to the policy; they are never converted to zero.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codeguard.evaluation.contracts import CaseIdentifier, NonBlankText
from codeguard.evaluation.reports import EvaluationReport

RegressionMetricName = Literal[
    "finding.precision",
    "finding.recall",
    "finding.f1",
    "finding.severity_accuracy",
    "finding.evidence_accuracy",
    "finding.line_accuracy",
    "recommendation.precision",
    "recommendation.recall",
    "recommendation.f1",
]
RegressionStatus = Literal[
    "regression",
    "improved",
    "unchanged",
    "within_threshold",
    "inconclusive",
]
RegressionOutcome = Literal["pass", "fail", "inconclusive"]


class _RegressionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class MetricThreshold(_RegressionModel):
    """Explicit minimum and/or maximum tolerated drop for one metric."""

    metric: RegressionMetricName
    minimum_value: float | None = Field(default=None, ge=0, le=1)
    maximum_absolute_drop: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_a_threshold(self) -> MetricThreshold:
        if self.minimum_value is None and self.maximum_absolute_drop is None:
            raise ValueError(
                "A metric threshold must define a minimum value or maximum drop."
            )
        return self


class RegressionPolicy(_RegressionModel):
    """Approved dataset, metric-definition and degradation requirements.

    The case ID set fixes coverage. ``undefined_metric_behavior="inconclusive"``
    preserves undefined comparisons in the result; ``"reject"`` raises
    ``UndefinedMetricError`` instead of producing a result.
    """

    dataset_id: CaseIdentifier
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    case_ids: tuple[CaseIdentifier, ...] = Field(min_length=1)
    metric_definitions_version: NonBlankText = Field(max_length=100)
    thresholds: tuple[MetricThreshold, ...] = Field(min_length=1, max_length=9)
    undefined_metric_behavior: Literal["inconclusive", "reject"]

    @model_validator(mode="after")
    def validate_unique_policy_entries(self) -> RegressionPolicy:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Regression policy case IDs must be unique.")
        metric_names = tuple(threshold.metric for threshold in self.thresholds)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Regression policy metric thresholds must be unique.")
        return self


class MetricComparison(_RegressionModel):
    """One baseline/candidate metric comparison and its configured threshold."""

    metric: RegressionMetricName
    baseline_value: float | None = Field(ge=0, le=1)
    candidate_value: float | None = Field(ge=0, le=1)
    absolute_change: float | None = Field(ge=-1, le=1)
    threshold: MetricThreshold
    status: RegressionStatus

    @model_validator(mode="after")
    def validate_comparison_values(self) -> MetricComparison:
        if self.metric != self.threshold.metric:
            raise ValueError("Comparison metric must match its configured threshold.")
        if self.baseline_value is None or self.candidate_value is None:
            if self.absolute_change is not None or self.status != "inconclusive":
                raise ValueError(
                    "Undefined metric values must have no change and be inconclusive."
                )
            return self
        expected_change = self.candidate_value - self.baseline_value
        if self.absolute_change != expected_change:
            raise ValueError("Absolute change must equal candidate minus baseline.")
        below_minimum = (
            self.threshold.minimum_value is not None
            and self.candidate_value < self.threshold.minimum_value
        )
        exceeds_drop = (
            self.threshold.maximum_absolute_drop is not None
            and self.baseline_value - self.candidate_value
            > self.threshold.maximum_absolute_drop
        )
        if below_minimum or exceeds_drop:
            expected_status: RegressionStatus = "regression"
        elif expected_change > 0:
            expected_status = "improved"
        elif expected_change == 0:
            expected_status = "unchanged"
        else:
            expected_status = "within_threshold"
        if self.status != expected_status:
            raise ValueError(
                "Metric comparison status does not match values and threshold."
            )
        return self


class RegressionResult(_RegressionModel):
    """Machine-readable outcome for an approved baseline comparison."""

    baseline_run_id: CaseIdentifier
    candidate_run_id: CaseIdentifier
    baseline_system_name: NonBlankText
    baseline_system_version: NonBlankText
    candidate_system_name: NonBlankText
    candidate_system_version: NonBlankText
    dataset_id: CaseIdentifier
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    metric_definitions_version: NonBlankText = Field(max_length=100)
    overall_outcome: RegressionOutcome
    comparisons: tuple[MetricComparison, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_overall_outcome(self) -> RegressionResult:
        statuses = tuple(comparison.status for comparison in self.comparisons)
        metric_names = tuple(comparison.metric for comparison in self.comparisons)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Regression result metric comparisons must be unique.")

        if "regression" in statuses:
            expected_outcome = "fail"
        elif "inconclusive" in statuses:
            expected_outcome = "inconclusive"
        else:
            expected_outcome = "pass"
        if self.overall_outcome != expected_outcome:
            raise ValueError(
                "Overall regression outcome does not match metric comparison statuses."
            )
        return self


class IncompatibleEvaluationReportsError(ValueError):
    """Raised when reports do not match the policy's comparison requirements."""


class UndefinedMetricError(ValueError):
    """Raised when policy rejects an undefined baseline or candidate metric."""


def compare_evaluation_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    policy: RegressionPolicy,
) -> RegressionResult:
    """Compare configured metrics without inferring dataset or match equivalence."""

    _validate_compatibility(baseline, candidate, policy)
    comparisons: list[MetricComparison] = []
    for threshold in policy.thresholds:
        baseline_value = _metric_value(baseline, threshold.metric)
        candidate_value = _metric_value(candidate, threshold.metric)
        if baseline_value is None or candidate_value is None:
            if policy.undefined_metric_behavior == "reject":
                raise UndefinedMetricError(
                    f"Metric {threshold.metric!r} is undefined in a report."
                )
            comparisons.append(
                MetricComparison(
                    metric=threshold.metric,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    absolute_change=None,
                    threshold=threshold,
                    status="inconclusive",
                )
            )
            continue

        change = candidate_value - baseline_value
        below_minimum = (
            threshold.minimum_value is not None
            and candidate_value < threshold.minimum_value
        )
        exceeds_drop = (
            threshold.maximum_absolute_drop is not None
            and baseline_value - candidate_value > threshold.maximum_absolute_drop
        )
        if below_minimum or exceeds_drop:
            status: RegressionStatus = "regression"
        elif change > 0:
            status = "improved"
        elif change == 0:
            status = "unchanged"
        else:
            status = "within_threshold"
        comparisons.append(
            MetricComparison(
                metric=threshold.metric,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                absolute_change=change,
                threshold=threshold,
                status=status,
            )
        )

    statuses = tuple(comparison.status for comparison in comparisons)
    if "regression" in statuses:
        overall_outcome: RegressionOutcome = "fail"
    elif "inconclusive" in statuses:
        overall_outcome = "inconclusive"
    else:
        overall_outcome = "pass"
    return RegressionResult(
        baseline_run_id=baseline.evaluation_run_id,
        candidate_run_id=candidate.evaluation_run_id,
        baseline_system_name=baseline.system_name,
        baseline_system_version=baseline.system_version,
        candidate_system_name=candidate.system_name,
        candidate_system_version=candidate.system_version,
        dataset_id=baseline.dataset_id,
        dataset_version=baseline.dataset_version,
        metric_definitions_version=baseline.metric_definitions_version,
        overall_outcome=overall_outcome,
        comparisons=tuple(comparisons),
    )


def _validate_compatibility(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    policy: RegressionPolicy,
) -> None:
    if baseline.dataset_id != candidate.dataset_id:
        raise IncompatibleEvaluationReportsError(
            "Baseline and candidate dataset IDs differ."
        )
    if baseline.dataset_version != candidate.dataset_version:
        raise IncompatibleEvaluationReportsError(
            "Baseline and candidate dataset versions differ."
        )
    if baseline.dataset_id != policy.dataset_id:
        raise IncompatibleEvaluationReportsError(
            "Report dataset ID does not match the regression policy."
        )
    if baseline.dataset_version != policy.dataset_version:
        raise IncompatibleEvaluationReportsError(
            "Report dataset version does not match the regression policy."
        )
    if baseline.metric_definitions_version != candidate.metric_definitions_version:
        raise IncompatibleEvaluationReportsError(
            "Baseline and candidate metric definitions differ."
        )
    if baseline.metric_definitions_version != policy.metric_definitions_version:
        raise IncompatibleEvaluationReportsError(
            "Report metric definitions do not match the regression policy."
        )

    approved_case_ids = set(policy.case_ids)
    for label, report in (("Baseline", baseline), ("Candidate", candidate)):
        finding_case_ids = tuple(case.case_id for case in report.metrics.per_case)
        recommendation_case_ids = tuple(
            case.case_id for case in report.metrics.test_recommendation_metrics.per_case
        )
        if (
            len(finding_case_ids) != report.metrics.evaluated_case_count
            or len(recommendation_case_ids)
            != report.metrics.test_recommendation_metrics.evaluated_case_count
            or len(finding_case_ids) != len(set(finding_case_ids))
            or len(recommendation_case_ids) != len(set(recommendation_case_ids))
            or set(finding_case_ids) != set(recommendation_case_ids)
            or set(finding_case_ids) != approved_case_ids
        ):
            raise IncompatibleEvaluationReportsError(
                f"{label} case coverage does not match the regression policy."
            )


def _metric_value(
    report: EvaluationReport,
    metric: RegressionMetricName,
) -> float | None:
    finding_metrics = report.metrics
    recommendation_metrics = finding_metrics.test_recommendation_metrics
    match metric:
        case "finding.precision":
            return finding_metrics.precision
        case "finding.recall":
            return finding_metrics.recall
        case "finding.f1":
            return finding_metrics.f1
        case "finding.severity_accuracy":
            return finding_metrics.severity_accuracy
        case "finding.evidence_accuracy":
            return finding_metrics.evidence_accuracy
        case "finding.line_accuracy":
            return finding_metrics.line_accuracy
        case "recommendation.precision":
            return recommendation_metrics.precision
        case "recommendation.recall":
            return recommendation_metrics.recall
        case "recommendation.f1":
            return recommendation_metrics.f1
    raise AssertionError(f"Unhandled regression metric {metric!r}.")


def render_regression_json(result: RegressionResult) -> str:
    """Serialize a regression result deterministically for machine consumers."""

    return f"{result.model_dump_json(indent=2)}\n"
