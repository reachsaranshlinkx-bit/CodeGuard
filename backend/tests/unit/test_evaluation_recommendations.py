import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from codeguard.evaluation.contracts import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    load_evaluation_dataset,
)
from codeguard.evaluation.contracts import (
    TestScenarioMatch as ScenarioMatch,
)
from codeguard.evaluation.metrics import calculate_evaluation_metrics


def _result_for(
    case_id: str,
    predictions: tuple[str, ...],
    match_pairs: tuple[tuple[int, int], ...] = (),
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        predicted_findings=(),
        predicted_test_scenarios=predictions,
        test_scenario_matches=tuple(
            ScenarioMatch(
                predicted_scenario_index=prediction_index,
                expected_scenario_index=expected_index,
            )
            for prediction_index, expected_index in match_pairs
        ),
    )


def _perfect_result(case: EvaluationCase) -> EvaluationResult:
    scenarios = case.recommended_test_scenarios
    return _result_for(
        case.case_id,
        scenarios,
        tuple((index, index) for index in range(len(scenarios))),
    )


def _run(
    replacements: dict[str, EvaluationResult] | None = None,
    *,
    dataset: EvaluationDataset | None = None,
    is_complete: bool = True,
) -> EvaluationRun:
    evaluation_dataset = dataset or load_evaluation_dataset()
    replacement_results = replacements or {}
    results = tuple(
        replacement_results.get(case.case_id, _perfect_result(case))
        for case in evaluation_dataset.cases
    )
    return EvaluationRun(
        run_id="recommendation-metrics-run",
        dataset=evaluation_dataset,
        started_at=datetime(2026, 11, 27, tzinfo=UTC),
        system_name="synthetic-test-system",
        system_version="1.0.0",
        is_complete=is_complete,
        results=results,
    )


def _dataset_with_no_scenarios(case_id: str) -> EvaluationDataset:
    payload: dict[str, Any] = json.loads(load_evaluation_dataset().model_dump_json())
    for case in payload["cases"]:
        if case["case_id"] == case_id:
            case["recommended_test_scenarios"] = []
            break
    return EvaluationDataset.model_validate_json(json.dumps(payload))


def test_perfect_recommendations_have_perfect_metrics() -> None:
    run = _run()

    metrics = calculate_evaluation_metrics(run).test_recommendation_metrics
    expected_count = sum(
        len(case.recommended_test_scenarios) for case in run.dataset.cases
    )

    assert metrics.evaluated_case_count == 10
    assert metrics.true_positives == expected_count
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0
    assert metrics.micro_averaged == metrics.macro_averaged
    assert len(metrics.per_case) == 10


def test_missing_recommendations_are_false_negatives() -> None:
    case = load_evaluation_dataset().cases[0]
    metrics = calculate_evaluation_metrics(
        _run({case.case_id: _result_for(case.case_id, ())})
    ).test_recommendation_metrics
    case_metrics = next(
        item for item in metrics.per_case if item.case_id == case.case_id
    )

    assert case_metrics.true_positives == 0
    assert case_metrics.false_positives == 0
    assert case_metrics.false_negatives == len(case.recommended_test_scenarios)
    assert case_metrics.precision is None
    assert case_metrics.recall == 0.0


def test_irrelevant_recommendation_is_false_positive() -> None:
    case = load_evaluation_dataset().cases[0]
    metrics = calculate_evaluation_metrics(
        _run({case.case_id: _result_for(case.case_id, ("Test unrelated behavior.",))})
    ).test_recommendation_metrics
    case_metrics = next(
        item for item in metrics.per_case if item.case_id == case.case_id
    )

    assert case_metrics.true_positives == 0
    assert case_metrics.false_positives == 1
    assert case_metrics.false_negatives == len(case.recommended_test_scenarios)


def test_mixed_correct_and_incorrect_recommendations_are_classified() -> None:
    case = load_evaluation_dataset().cases[0]
    predictions = (case.recommended_test_scenarios[0], "Test unrelated behavior.")
    result = _result_for(case.case_id, predictions, ((0, 0),))

    metrics = calculate_evaluation_metrics(_run({case.case_id: result}))
    case_metrics = next(
        item
        for item in metrics.test_recommendation_metrics.per_case
        if item.case_id == case.case_id
    )

    assert case_metrics.true_positives == 1
    assert case_metrics.false_positives == 1
    assert case_metrics.false_negatives == 1


def test_duplicate_prediction_or_expected_scenario_matches_are_rejected() -> None:
    case = load_evaluation_dataset().cases[0]

    with pytest.raises(ValidationError, match="predicted test scenario cannot match"):
        _result_for(
            case.case_id,
            ("First predicted recommendation.", "Second predicted recommendation."),
            ((0, 0), (0, 1)),
        )

    with pytest.raises(ValidationError, match="expected test scenario cannot match"):
        _result_for(
            case.case_id,
            ("First predicted recommendation.", "Second predicted recommendation."),
            ((0, 0), (1, 0)),
        )


@pytest.mark.parametrize(
    ("predictions", "match", "message"),
    [
        (
            ("Only prediction.",),
            ScenarioMatch(predicted_scenario_index=1, expected_scenario_index=0),
            "must reference a predicted scenario",
        ),
        (
            ("Only prediction.",),
            ScenarioMatch(predicted_scenario_index=0, expected_scenario_index=2),
            "must reference an expected scenario",
        ),
    ],
)
def test_out_of_range_scenario_indexes_are_rejected_by_run_validation(
    predictions: tuple[str, ...],
    match: ScenarioMatch,
    message: str,
) -> None:
    case = load_evaluation_dataset().cases[0]
    result = EvaluationResult(
        case_id=case.case_id,
        predicted_findings=(),
        predicted_test_scenarios=predictions,
        test_scenario_matches=(match,),
    )

    with pytest.raises(ValidationError, match=message):
        EvaluationRun(
            run_id="invalid-scenario-index-run",
            dataset=load_evaluation_dataset(),
            started_at=datetime(2026, 11, 27, tzinfo=UTC),
            system_name="synthetic-test-system",
            system_version="1.0.0",
            is_complete=False,
            results=(result,),
        )


def test_negative_cases_are_evaluated_for_recommendations_independently() -> None:
    run = _run()
    negative_case = next(
        case for case in run.dataset.cases if case.case_type == "negative"
    )
    metrics = calculate_evaluation_metrics(run)
    recommendation_case = next(
        item
        for item in metrics.test_recommendation_metrics.per_case
        if item.case_id == negative_case.case_id
    )
    finding_case = next(
        item for item in metrics.per_case if item.case_id == negative_case.case_id
    )

    assert recommendation_case.true_positives == len(
        negative_case.recommended_test_scenarios
    )
    assert finding_case.true_positives == 0
    assert finding_case.false_negatives == 0


def test_zero_predictions_has_undefined_precision_and_zero_recall() -> None:
    dataset = load_evaluation_dataset()
    results = tuple(_result_for(case.case_id, ()) for case in dataset.cases)
    run = EvaluationRun(
        run_id="zero-recommendations-run",
        dataset=dataset,
        started_at=datetime(2026, 11, 27, tzinfo=UTC),
        system_name="synthetic-test-system",
        system_version="1.0.0",
        results=results,
    )

    metrics = calculate_evaluation_metrics(run).test_recommendation_metrics

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.precision is None
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_zero_expected_scenarios_have_undefined_case_metrics() -> None:
    case_id = "safe-subprocess-arguments"
    dataset = _dataset_with_no_scenarios(case_id)
    metrics = calculate_evaluation_metrics(_run(dataset=dataset))
    case_metrics = next(
        item
        for item in metrics.test_recommendation_metrics.per_case
        if item.case_id == case_id
    )

    assert case_metrics.true_positives == 0
    assert case_metrics.false_positives == 0
    assert case_metrics.false_negatives == 0
    assert case_metrics.precision is None
    assert case_metrics.recall is None
    assert case_metrics.f1 is None


def test_recommendation_micro_and_macro_averages_are_distinct() -> None:
    case = load_evaluation_dataset().cases[0]
    predictions = (*case.recommended_test_scenarios, "Test unrelated behavior.")
    result = _result_for(
        case.case_id,
        predictions,
        ((0, 0), (1, 1)),
    )

    metrics = calculate_evaluation_metrics(_run({case.case_id: result}))
    recommendations = metrics.test_recommendation_metrics

    assert recommendations.micro_averaged.precision == recommendations.precision
    assert recommendations.micro_averaged.precision == (
        recommendations.true_positives
        / (recommendations.true_positives + recommendations.false_positives)
    )
    assert recommendations.macro_averaged.precision > (
        recommendations.micro_averaged.precision
    )


def test_incomplete_recommendation_run_is_rejected_by_metrics() -> None:
    dataset = load_evaluation_dataset()
    run = _run(
        {dataset.cases[0].case_id: _perfect_result(dataset.cases[0])},
        is_complete=False,
    )
    partial_run = run.model_copy(update={"results": (run.results[0],)})

    with pytest.raises(ValueError, match="incomplete run"):
        calculate_evaluation_metrics(partial_run)


def test_recommendation_metric_output_is_reproducible() -> None:
    run = _run()

    first = calculate_evaluation_metrics(
        run
    ).test_recommendation_metrics.model_dump_json()
    second = calculate_evaluation_metrics(
        run
    ).test_recommendation_metrics.model_dump_json()

    assert first == second


def test_recommendation_changes_do_not_change_finding_metrics() -> None:
    case = load_evaluation_dataset().cases[0]
    original = _run()
    replacement = _result_for(case.case_id, ("Test unrelated behavior.",))
    changed = _run({case.case_id: replacement})
    original_metrics = calculate_evaluation_metrics(original)
    changed_metrics = calculate_evaluation_metrics(changed)

    assert original_metrics.true_positives == changed_metrics.true_positives
    assert original_metrics.false_positives == changed_metrics.false_positives
    assert original_metrics.false_negatives == changed_metrics.false_negatives
    assert (
        original_metrics.test_recommendation_metrics.false_positives
        != changed_metrics.test_recommendation_metrics.false_positives
    )
