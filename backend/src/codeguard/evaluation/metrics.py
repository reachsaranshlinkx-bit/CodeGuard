"""Deterministic metrics for complete, provider-neutral evaluation runs.

For each case, TP is the number of explicit prediction/label pairs, FP is the
number of predictions without a pair, and FN is the number of labels without
a pair. Precision = TP / (TP + FP), recall = TP / (TP + FN), and
F1 = 2 * TP / (2 * TP + FP + FN). A zero denominator produces ``None``.

Micro metrics calculate these formulas after summing counts across cases.
Macro metrics take the arithmetic mean of each per-case metric, omitting
undefined (``None``) values; a metric with no defined case values is ``None``.
Severity, evidence, and line accuracy each equal exact correct matched pairs
divided by TP, and are ``None`` when TP is zero. The negative-case false
positive rate is the number of evaluated negative cases with at least one FP
divided by the number of evaluated negative cases; it is not the classical
statistical false-positive rate. Test recommendation counts use only explicit
scenario matches: unmatched predictions are FP and unmatched expected
scenarios are FN. Recommendation metrics use the same precision, recall, F1,
micro, and macro formulas, independently of finding metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codeguard.evaluation.contracts import (
    CaseIdentifier,
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
)


class _MetricsModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class CoreMetrics(_MetricsModel):
    """Precision, recall, and F1, with undefined values represented by None."""

    precision: float | None = Field(ge=0, le=1)
    recall: float | None = Field(ge=0, le=1)
    f1: float | None = Field(ge=0, le=1)


class CaseEvaluationMetrics(_MetricsModel):
    """Finding classification and quality metrics for one evaluated case."""

    case_id: CaseIdentifier
    case_type: Literal["positive", "negative"]
    true_positives: int = Field(ge=0, strict=True)
    false_positives: int = Field(ge=0, strict=True)
    false_negatives: int = Field(ge=0, strict=True)
    precision: float | None = Field(ge=0, le=1)
    recall: float | None = Field(ge=0, le=1)
    f1: float | None = Field(ge=0, le=1)
    severity_accuracy: float | None = Field(ge=0, le=1)
    evidence_accuracy: float | None = Field(ge=0, le=1)
    line_accuracy: float | None = Field(ge=0, le=1)


class TestRecommendationCaseMetrics(_MetricsModel):
    """Recommendation classification metrics for one evaluated case."""

    case_id: CaseIdentifier
    true_positives: int = Field(ge=0, strict=True)
    false_positives: int = Field(ge=0, strict=True)
    false_negatives: int = Field(ge=0, strict=True)
    precision: float | None = Field(ge=0, le=1)
    recall: float | None = Field(ge=0, le=1)
    f1: float | None = Field(ge=0, le=1)


class TestRecommendationMetrics(_MetricsModel):
    """Aggregate and per-case metrics for test recommendations only."""

    evaluated_case_count: int = Field(ge=1, strict=True)
    true_positives: int = Field(ge=0, strict=True)
    false_positives: int = Field(ge=0, strict=True)
    false_negatives: int = Field(ge=0, strict=True)
    precision: float | None = Field(ge=0, le=1)
    recall: float | None = Field(ge=0, le=1)
    f1: float | None = Field(ge=0, le=1)
    micro_averaged: CoreMetrics
    macro_averaged: CoreMetrics
    per_case: tuple[TestRecommendationCaseMetrics, ...]


class EvaluationMetrics(_MetricsModel):
    """Validated aggregate metrics and dataset-ordered per-case breakdown."""

    dataset_id: CaseIdentifier
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    evaluation_run_id: CaseIdentifier
    evaluated_case_count: int = Field(ge=1, strict=True)
    true_positives: int = Field(ge=0, strict=True)
    false_positives: int = Field(ge=0, strict=True)
    false_negatives: int = Field(ge=0, strict=True)
    precision: float | None = Field(ge=0, le=1)
    recall: float | None = Field(ge=0, le=1)
    f1: float | None = Field(ge=0, le=1)
    micro_averaged: CoreMetrics
    macro_averaged: CoreMetrics
    severity_accuracy: float | None = Field(ge=0, le=1)
    evidence_accuracy: float | None = Field(ge=0, le=1)
    line_accuracy: float | None = Field(ge=0, le=1)
    negative_case_false_positive_rate: float | None = Field(ge=0, le=1)
    per_case: tuple[CaseEvaluationMetrics, ...]
    test_recommendation_metrics: TestRecommendationMetrics


def calculate_evaluation_metrics(run: EvaluationRun) -> EvaluationMetrics:
    """Calculate metrics from explicit matches in a complete evaluation run."""

    if not run.is_complete:
        raise ValueError("Metrics cannot be calculated for an incomplete run.")

    results_by_case_id = {result.case_id: result for result in run.results}
    if set(results_by_case_id) != {case.case_id for case in run.dataset.cases}:
        raise ValueError("Metrics require a result for every dataset case.")

    per_case = tuple(
        _calculate_case_metrics(case, results_by_case_id[case.case_id])
        for case in run.dataset.cases
    )
    true_positives = sum(metrics.true_positives for metrics in per_case)
    false_positives = sum(metrics.false_positives for metrics in per_case)
    false_negatives = sum(metrics.false_negatives for metrics in per_case)
    micro = _core_metrics(true_positives, false_positives, false_negatives)
    macro = CoreMetrics(
        precision=_mean_defined(metrics.precision for metrics in per_case),
        recall=_mean_defined(metrics.recall for metrics in per_case),
        f1=_mean_defined(metrics.f1 for metrics in per_case),
    )
    test_recommendation_metrics = _calculate_test_recommendation_metrics(
        run,
        results_by_case_id,
    )

    matched_count = true_positives
    severity_correct = 0
    evidence_correct = 0
    line_correct = 0
    negative_case_count = 0
    negative_cases_with_false_positives = 0
    for case in run.dataset.cases:
        result = results_by_case_id[case.case_id]
        false_positive_count = len(result.predicted_findings) - len(
            result.finding_matches
        )
        if case.case_type == "negative":
            negative_case_count += 1
            if false_positive_count:
                negative_cases_with_false_positives += 1
        expected_by_id = {
            expected.finding_id: expected.finding for expected in case.expected_findings
        }
        for match in result.finding_matches:
            predicted = result.predicted_findings[match.predicted_finding_index]
            expected = expected_by_id[match.ground_truth_finding_id]
            severity_correct += predicted.severity == expected.severity
            evidence_correct += predicted.evidence == expected.evidence
            line_correct += predicted.line_number == expected.line_number

    return EvaluationMetrics(
        dataset_id=run.dataset.dataset_id,
        dataset_version=run.dataset.dataset_version,
        evaluation_run_id=run.run_id,
        evaluated_case_count=len(per_case),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=micro.precision,
        recall=micro.recall,
        f1=micro.f1,
        micro_averaged=micro,
        macro_averaged=macro,
        severity_accuracy=_ratio(severity_correct, matched_count),
        evidence_accuracy=_ratio(evidence_correct, matched_count),
        line_accuracy=_ratio(line_correct, matched_count),
        negative_case_false_positive_rate=_ratio(
            negative_cases_with_false_positives,
            negative_case_count,
        ),
        per_case=per_case,
        test_recommendation_metrics=test_recommendation_metrics,
    )


def _calculate_case_metrics(
    case: EvaluationCase,
    result: EvaluationResult,
) -> CaseEvaluationMetrics:
    true_positives = len(result.finding_matches)
    false_positives = len(result.predicted_findings) - true_positives
    false_negatives = len(case.expected_findings) - true_positives
    core = _core_metrics(true_positives, false_positives, false_negatives)
    expected_by_id = {
        expected.finding_id: expected.finding for expected in case.expected_findings
    }
    severity_correct = 0
    evidence_correct = 0
    line_correct = 0
    for match in result.finding_matches:
        predicted = result.predicted_findings[match.predicted_finding_index]
        expected = expected_by_id[match.ground_truth_finding_id]
        severity_correct += predicted.severity == expected.severity
        evidence_correct += predicted.evidence == expected.evidence
        line_correct += predicted.line_number == expected.line_number

    return CaseEvaluationMetrics(
        case_id=case.case_id,
        case_type=case.case_type,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=core.precision,
        recall=core.recall,
        f1=core.f1,
        severity_accuracy=_ratio(severity_correct, true_positives),
        evidence_accuracy=_ratio(evidence_correct, true_positives),
        line_accuracy=_ratio(line_correct, true_positives),
    )


def _calculate_test_recommendation_metrics(
    run: EvaluationRun,
    results_by_case_id: dict[str, EvaluationResult],
) -> TestRecommendationMetrics:
    per_case = tuple(
        _calculate_test_recommendation_case_metrics(
            case,
            results_by_case_id[case.case_id],
        )
        for case in run.dataset.cases
    )
    true_positives = sum(metrics.true_positives for metrics in per_case)
    false_positives = sum(metrics.false_positives for metrics in per_case)
    false_negatives = sum(metrics.false_negatives for metrics in per_case)
    micro = _core_metrics(true_positives, false_positives, false_negatives)
    macro = CoreMetrics(
        precision=_mean_defined(metrics.precision for metrics in per_case),
        recall=_mean_defined(metrics.recall for metrics in per_case),
        f1=_mean_defined(metrics.f1 for metrics in per_case),
    )
    return TestRecommendationMetrics(
        evaluated_case_count=len(per_case),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=micro.precision,
        recall=micro.recall,
        f1=micro.f1,
        micro_averaged=micro,
        macro_averaged=macro,
        per_case=per_case,
    )


def _calculate_test_recommendation_case_metrics(
    case: EvaluationCase,
    result: EvaluationResult,
) -> TestRecommendationCaseMetrics:
    true_positives = len(result.test_scenario_matches)
    false_positives = len(result.predicted_test_scenarios) - true_positives
    false_negatives = len(case.recommended_test_scenarios) - true_positives
    core = _core_metrics(true_positives, false_positives, false_negatives)
    return TestRecommendationCaseMetrics(
        case_id=case.case_id,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=core.precision,
        recall=core.recall,
        f1=core.f1,
    )


def _core_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> CoreMetrics:
    return CoreMetrics(
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, true_positives + false_negatives),
        f1=_ratio(
            2 * true_positives,
            2 * true_positives + false_positives + false_negatives,
        ),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined_values = tuple(value for value in values if value is not None)
    if not defined_values:
        return None
    return sum(defined_values) / len(defined_values)
