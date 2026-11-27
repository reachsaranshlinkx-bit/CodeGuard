from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.evaluation.contracts import (
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
    FindingMatch,
    load_evaluation_dataset,
)
from codeguard.evaluation.metrics import calculate_evaluation_metrics


def _result_for(
    case: EvaluationCase,
    predictions: tuple[AIFindingCandidate, ...],
    match_pairs: tuple[tuple[int, str], ...] = (),
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case.case_id,
        predicted_findings=predictions,
        finding_matches=tuple(
            FindingMatch(
                predicted_finding_index=prediction_index,
                ground_truth_finding_id=finding_id,
            )
            for prediction_index, finding_id in match_pairs
        ),
    )


def _perfect_result(case: EvaluationCase) -> EvaluationResult:
    predictions = tuple(expected.finding for expected in case.expected_findings)
    matches = tuple(
        (index, expected.finding_id)
        for index, expected in enumerate(case.expected_findings)
    )
    return _result_for(case, predictions, matches)


def _run(
    replacements: dict[str, EvaluationResult] | None = None,
    *,
    is_complete: bool = True,
) -> EvaluationRun:
    dataset = load_evaluation_dataset()
    replacement_results = replacements or {}
    results = tuple(
        replacement_results.get(case.case_id, _perfect_result(case))
        for case in dataset.cases
    )
    return EvaluationRun(
        run_id="synthetic-metrics-run",
        dataset=dataset,
        started_at=datetime(2026, 11, 27, tzinfo=UTC),
        system_name="synthetic-test-system",
        system_version="1.0.0",
        is_complete=is_complete,
        results=results,
    )


def _case(run: EvaluationRun, case_id: str) -> EvaluationCase:
    return next(case for case in run.dataset.cases if case.case_id == case_id)


def test_perfect_predictions_have_perfect_classification_and_quality_metrics() -> None:
    run = _run()

    metrics = calculate_evaluation_metrics(run)

    expected_true_positives = sum(
        len(case.expected_findings) for case in run.dataset.cases
    )
    assert metrics.evaluated_case_count == len(run.dataset.cases)
    assert metrics.true_positives == expected_true_positives
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0
    assert metrics.severity_accuracy == 1.0
    assert metrics.evidence_accuracy == 1.0
    assert metrics.line_accuracy == 1.0
    assert metrics.negative_case_false_positive_rate == 0.0
    assert len(metrics.per_case) == len(run.dataset.cases)


def test_missing_findings_are_false_negatives_not_correct_matches() -> None:
    run = _run()
    case = _case(run, "untrusted-eval-input")
    missing_result = _result_for(case, ())
    metrics = calculate_evaluation_metrics(_run({case.case_id: missing_result}))
    case_metrics = next(
        item for item in metrics.per_case if item.case_id == case.case_id
    )

    assert case_metrics.true_positives == 0
    assert case_metrics.false_negatives == 1
    assert case_metrics.recall == 0.0
    assert case_metrics.severity_accuracy is None
    assert metrics.false_negatives == 1


def test_mixed_positive_and_negative_cases_measure_negative_case_fp_rate() -> None:
    run = _run()
    negative_case = next(
        case for case in run.dataset.cases if case.case_type == "negative"
    )
    false_positive = run.dataset.cases[0].expected_findings[0].finding
    result = _result_for(negative_case, (false_positive,))

    metrics = calculate_evaluation_metrics(_run({negative_case.case_id: result}))

    negative_case_metrics = next(
        item for item in metrics.per_case if item.case_id == negative_case.case_id
    )
    negative_case_count = sum(
        case.case_type == "negative" for case in run.dataset.cases
    )
    assert metrics.true_positives > 0
    assert negative_case_metrics.false_positives == 1
    assert metrics.negative_case_false_positive_rate == 1 / negative_case_count


def test_zero_predictions_make_precision_undefined_and_recall_zero() -> None:
    dataset = load_evaluation_dataset()
    results = tuple(_result_for(case, ()) for case in dataset.cases)
    run = EvaluationRun(
        run_id="zero-predictions-run",
        dataset=dataset,
        started_at=datetime(2026, 11, 27, tzinfo=UTC),
        system_name="synthetic-test-system",
        system_version="1.0.0",
        results=results,
    )

    metrics = calculate_evaluation_metrics(run)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.precision is None
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert all(
        item.precision is None
        for item in metrics.per_case
        if item.case_type == "negative"
    )


def test_zero_ground_truth_findings_have_undefined_recall() -> None:
    run = _run()
    negative_case = next(
        case for case in run.dataset.cases if case.case_type == "negative"
    )
    false_positive = run.dataset.cases[0].expected_findings[0].finding
    metrics = calculate_evaluation_metrics(
        _run({negative_case.case_id: _result_for(negative_case, (false_positive,))})
    )
    negative_metrics = next(
        item for item in metrics.per_case if item.case_id == negative_case.case_id
    )

    assert negative_metrics.true_positives == 0
    assert negative_metrics.false_positives == 1
    assert negative_metrics.false_negatives == 0
    assert negative_metrics.recall is None
    assert negative_metrics.precision == 0.0
    assert negative_metrics.f1 == 0.0


def test_duplicate_prediction_or_ground_truth_matches_are_rejected() -> None:
    case = load_evaluation_dataset().cases[0]
    finding_id = case.expected_findings[0].finding_id
    prediction = case.expected_findings[0].finding

    with pytest.raises(ValidationError, match="predicted finding cannot match"):
        EvaluationResult(
            case_id=case.case_id,
            predicted_findings=(prediction, prediction),
            finding_matches=(
                FindingMatch(
                    predicted_finding_index=0,
                    ground_truth_finding_id=finding_id,
                ),
                FindingMatch(
                    predicted_finding_index=0,
                    ground_truth_finding_id=finding_id,
                ),
            ),
        )

    with pytest.raises(ValidationError, match="ground-truth finding cannot match"):
        EvaluationResult(
            case_id=case.case_id,
            predicted_findings=(prediction, prediction),
            finding_matches=(
                FindingMatch(
                    predicted_finding_index=0,
                    ground_truth_finding_id=finding_id,
                ),
                FindingMatch(
                    predicted_finding_index=1,
                    ground_truth_finding_id=finding_id,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "metric_field"),
    [
        ("severity", "low", "severity_accuracy"),
        ("evidence", "    return unrelated_value", "evidence_accuracy"),
        ("line_number", 3, "line_accuracy"),
    ],
)
def test_quality_accuracy_uses_only_explicit_matches(
    field: str,
    replacement: str | int,
    metric_field: str,
) -> None:
    run = _run()
    case = _case(run, "untrusted-eval-input")
    original = case.expected_findings[0].finding
    prediction_data = original.model_dump()
    prediction_data[field] = replacement
    prediction = AIFindingCandidate.model_validate(prediction_data)
    result = _result_for(case, (prediction,), ((0, "dynamic-execution"),))

    metrics = calculate_evaluation_metrics(_run({case.case_id: result}))
    case_metrics = next(
        item for item in metrics.per_case if item.case_id == case.case_id
    )

    assert getattr(case_metrics, metric_field) == 0.0
    assert getattr(metrics, metric_field) < 1.0


def test_micro_and_macro_averages_use_different_aggregation_rules() -> None:
    run = _run()
    case = _case(run, "untrusted-eval-input")
    prediction = case.expected_findings[0].finding
    result = _result_for(
        case,
        (prediction, prediction),
        ((0, "dynamic-execution"),),
    )

    metrics = calculate_evaluation_metrics(_run({case.case_id: result}))

    assert metrics.micro_averaged.precision == metrics.precision
    assert metrics.micro_averaged.precision == metrics.true_positives / (
        metrics.true_positives + metrics.false_positives
    )
    assert metrics.macro_averaged.precision > metrics.micro_averaged.precision


def test_incomplete_runs_are_explicit_and_metrics_reject_them() -> None:
    dataset = load_evaluation_dataset()
    partial_result = _perfect_result(dataset.cases[0])

    with pytest.raises(ValidationError, match="must include every dataset case"):
        EvaluationRun(
            run_id="implicit-partial-run",
            dataset=dataset,
            started_at=datetime(2026, 11, 27, tzinfo=UTC),
            system_name="synthetic-test-system",
            system_version="1.0.0",
            results=(partial_result,),
        )

    incomplete_run = EvaluationRun(
        run_id="explicit-partial-run",
        dataset=dataset,
        started_at=datetime(2026, 11, 27, tzinfo=UTC),
        system_name="synthetic-test-system",
        system_version="1.0.0",
        is_complete=False,
        results=(partial_result,),
    )
    with pytest.raises(ValueError, match="incomplete run"):
        calculate_evaluation_metrics(incomplete_run)


def test_metric_output_is_reproducible() -> None:
    run = _run()

    first = calculate_evaluation_metrics(run).model_dump_json()
    second = calculate_evaluation_metrics(run).model_dump_json()

    assert first == second
