import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from codeguard.ai.contracts import AIFindingCandidate
from codeguard.evaluation.contracts import (
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    FindingMatch,
    load_evaluation_dataset,
)
from codeguard.evaluation.contracts import (
    TestScenarioMatch as ScenarioMatch,
)
from codeguard.evaluation.regression import (
    IncompatibleEvaluationReportsError,
    MetricThreshold,
    RegressionPolicy,
    UndefinedMetricError,
    compare_evaluation_reports,
    render_regression_json,
)
from codeguard.evaluation.reports import (
    EvaluationReport,
    build_evaluation_report,
    render_json_report,
)
from codeguard.evaluation.runner import main


def _report(
    *,
    run_id: str = "baseline-run",
    system_version: str = "1.0.0",
    dataset: EvaluationDataset | None = None,
    finding_change: str | None = None,
    recommendation_change: str | None = None,
) -> EvaluationReport:
    evaluation_dataset = dataset or load_evaluation_dataset()
    results: list[EvaluationResult] = []
    for case_index, case in enumerate(evaluation_dataset.cases):
        expected_findings = tuple(finding.finding for finding in case.expected_findings)
        predicted_findings = expected_findings
        finding_matches = tuple(
            FindingMatch(
                predicted_finding_index=index,
                ground_truth_finding_id=finding.finding_id,
            )
            for index, finding in enumerate(case.expected_findings)
        )
        predicted_scenarios = case.recommended_test_scenarios
        scenario_matches = tuple(
            ScenarioMatch(
                predicted_scenario_index=index,
                expected_scenario_index=index,
            )
            for index in range(len(case.recommended_test_scenarios))
        )

        if case_index == 0 and finding_change == "extra_false_positive":
            predicted_findings = (*expected_findings, expected_findings[0])
        elif case_index == 0 and finding_change == "severity_mismatch":
            finding_payload = expected_findings[0].model_dump()
            finding_payload["severity"] = "low"
            changed_finding = AIFindingCandidate.model_validate(finding_payload)
            predicted_findings = (changed_finding, *expected_findings[1:])
        elif finding_change == "no_predictions":
            predicted_findings = ()
            finding_matches = ()

        if case_index == 0 and recommendation_change == "extra_false_positive":
            predicted_scenarios = (*predicted_scenarios, "Irrelevant test scenario.")
        elif case_index == 0 and recommendation_change == "missing_one":
            predicted_scenarios = predicted_scenarios[:-1]
            scenario_matches = tuple(
                ScenarioMatch(
                    predicted_scenario_index=index,
                    expected_scenario_index=index,
                )
                for index in range(len(predicted_scenarios))
            )

        results.append(
            EvaluationResult(
                case_id=case.case_id,
                predicted_findings=predicted_findings,
                finding_matches=finding_matches,
                predicted_test_scenarios=predicted_scenarios,
                test_scenario_matches=scenario_matches,
            )
        )

    run = EvaluationRun(
        run_id=run_id,
        dataset=evaluation_dataset,
        started_at=datetime(2026, 11, 27, 10, 0, tzinfo=UTC),
        system_name="synthetic-regression-fixture",
        system_version=system_version,
        results=tuple(results),
    )
    return build_evaluation_report(run)


def _policy(
    report: EvaluationReport,
    *thresholds: MetricThreshold,
    undefined_behavior: str = "inconclusive",
) -> RegressionPolicy:
    return RegressionPolicy(
        dataset_id=report.dataset_id,
        dataset_version=report.dataset_version,
        case_ids=tuple(case.case_id for case in report.metrics.per_case),
        metric_definitions_version=report.metric_definitions_version,
        thresholds=tuple(thresholds)
        or (MetricThreshold(metric="finding.f1", minimum_value=0.9),),
        undefined_metric_behavior=undefined_behavior,
    )


def _dataset_with_version(version: str) -> EvaluationDataset:
    payload: dict[str, Any] = json.loads(load_evaluation_dataset().model_dump_json())
    payload["dataset_version"] = version
    return EvaluationDataset.model_validate_json(json.dumps(payload))


def _dataset_with_id(dataset_id: str) -> EvaluationDataset:
    payload: dict[str, Any] = json.loads(load_evaluation_dataset().model_dump_json())
    payload["dataset_id"] = dataset_id
    return EvaluationDataset.model_validate_json(json.dumps(payload))


def _dataset_without_last_case() -> EvaluationDataset:
    payload: dict[str, Any] = json.loads(load_evaluation_dataset().model_dump_json())
    payload["cases"].pop()
    return EvaluationDataset.model_validate_json(json.dumps(payload))


def _write_comparison_inputs(
    tmp_path: Path,
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    policy: RegressionPolicy,
) -> tuple[Path, Path, Path]:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "policy.json"
    baseline_path.write_text(render_json_report(baseline), encoding="utf-8")
    candidate_path.write_text(render_json_report(candidate), encoding="utf-8")
    policy_path.write_text(policy.model_dump_json(indent=2), encoding="utf-8")
    return baseline_path, candidate_path, policy_path


def test_identical_reports_pass_and_allow_different_run_and_system_versions() -> None:
    baseline = _report()
    candidate = _report(run_id="candidate-run", system_version="2.0.0")
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.f1", minimum_value=0.9),
        MetricThreshold(
            metric="recommendation.f1",
            maximum_absolute_drop=0.05,
        ),
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "pass"
    assert result.baseline_run_id == "baseline-run"
    assert result.candidate_run_id == "candidate-run"
    assert result.baseline_system_version == "1.0.0"
    assert result.candidate_system_version == "2.0.0"
    assert all(comparison.status == "unchanged" for comparison in result.comparisons)


def test_improved_metrics_are_identified() -> None:
    baseline = _report(recommendation_change="missing_one")
    candidate = _report(run_id="candidate-run")
    policy = _policy(
        baseline,
        MetricThreshold(
            metric="recommendation.recall",
            maximum_absolute_drop=0.0,
        ),
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "pass"
    assert result.comparisons[0].status == "improved"
    assert result.comparisons[0].absolute_change is not None
    assert result.comparisons[0].absolute_change > 0


def test_decreased_metric_beyond_allowed_drop_is_a_regression() -> None:
    baseline = _report()
    candidate = _report(
        run_id="candidate-run",
        finding_change="extra_false_positive",
    )
    policy = _policy(
        baseline,
        MetricThreshold(
            metric="finding.precision",
            maximum_absolute_drop=0.05,
        ),
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "fail"
    assert result.comparisons[0].status == "regression"
    assert result.comparisons[0].absolute_change is not None
    assert result.comparisons[0].absolute_change < 0


def test_threshold_boundaries_are_inclusive() -> None:
    baseline = _report()
    candidate = _report(finding_change="extra_false_positive")
    baseline_precision = baseline.metrics.precision
    candidate_precision = candidate.metrics.precision
    assert baseline_precision is not None
    assert candidate_precision is not None
    exact_drop = baseline_precision - candidate_precision
    policy = _policy(
        baseline,
        MetricThreshold(
            metric="finding.precision",
            minimum_value=candidate_precision,
            maximum_absolute_drop=exact_drop,
        ),
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "pass"
    assert result.comparisons[0].status == "within_threshold"


def test_dataset_version_mismatch_is_rejected() -> None:
    baseline = _report()
    candidate = _report(
        run_id="other-dataset-version",
        dataset=_dataset_with_version("1.0.1"),
    )

    with pytest.raises(IncompatibleEvaluationReportsError, match="versions differ"):
        compare_evaluation_reports(baseline, candidate, _policy(baseline))


def test_dataset_id_mismatch_is_rejected() -> None:
    baseline = _report()
    candidate = _report(
        run_id="other-dataset-run",
        dataset=_dataset_with_id("other-golden-dataset"),
    )

    with pytest.raises(IncompatibleEvaluationReportsError, match="dataset IDs differ"):
        compare_evaluation_reports(baseline, candidate, _policy(baseline))


def test_missing_case_coverage_is_rejected() -> None:
    baseline = _report()
    candidate = _report(
        run_id="missing-case-run",
        dataset=_dataset_without_last_case(),
    )

    with pytest.raises(IncompatibleEvaluationReportsError, match="case coverage"):
        compare_evaluation_reports(baseline, candidate, _policy(baseline))


def test_metric_definition_mismatch_is_rejected() -> None:
    baseline = _report()
    candidate = _report(run_id="candidate-run")
    candidate_data = json.loads(candidate.model_dump_json())
    candidate_data["metric_definitions_version"] = "other-definition-version"
    changed_candidate = EvaluationReport.model_validate_json(json.dumps(candidate_data))

    with pytest.raises(
        IncompatibleEvaluationReportsError,
        match="metric definitions differ",
    ):
        compare_evaluation_reports(baseline, changed_candidate, _policy(baseline))


def test_reports_without_a_metric_version_use_the_compatible_legacy_version() -> None:
    report_data = json.loads(_report().model_dump_json())
    report_data.pop("metric_definitions_version")
    baseline = EvaluationReport.model_validate_json(json.dumps(report_data))
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.f1", minimum_value=0.9),
    )

    result = compare_evaluation_reports(baseline, _report(), policy)

    assert baseline.metric_definitions_version == "codeguard-eval-metrics-v1"
    assert result.overall_outcome == "pass"


@pytest.mark.parametrize(
    "undefined_side",
    ["baseline", "candidate"],
)
def test_undefined_metrics_are_inconclusive_by_policy(undefined_side: str) -> None:
    perfect = _report()
    no_predictions = _report(
        run_id="no-predictions-run",
        finding_change="no_predictions",
    )
    baseline = no_predictions if undefined_side == "baseline" else perfect
    candidate = no_predictions if undefined_side == "candidate" else perfect
    policy = _policy(
        perfect,
        MetricThreshold(metric="finding.precision", minimum_value=0.8),
        undefined_behavior="inconclusive",
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "inconclusive"
    assert result.comparisons[0].status == "inconclusive"
    assert result.comparisons[0].absolute_change is None


def test_undefined_metrics_can_be_rejected_by_policy() -> None:
    baseline = _report()
    candidate = _report(
        run_id="no-predictions-run",
        finding_change="no_predictions",
    )
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.precision", minimum_value=0.8),
        undefined_behavior="reject",
    )

    with pytest.raises(UndefinedMetricError, match="finding.precision"):
        compare_evaluation_reports(baseline, candidate, policy)


def test_finding_quality_regression_is_detected() -> None:
    baseline = _report()
    candidate = _report(
        run_id="severity-regression-run",
        finding_change="severity_mismatch",
    )
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.severity_accuracy", minimum_value=0.9),
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "fail"
    assert result.comparisons[0].status == "regression"


def test_recommendation_quality_regression_is_detected() -> None:
    baseline = _report()
    candidate = _report(
        run_id="recommendation-regression-run",
        recommendation_change="extra_false_positive",
    )
    policy = _policy(
        baseline,
        MetricThreshold(
            metric="recommendation.precision",
            minimum_value=0.98,
        ),
    )

    result = compare_evaluation_reports(baseline, candidate, policy)

    assert result.overall_outcome == "fail"
    assert result.comparisons[0].status == "regression"


def test_invalid_policies_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must define a minimum value"):
        MetricThreshold(metric="finding.precision")

    report = _report()
    with pytest.raises(ValidationError, match="thresholds must be unique"):
        _policy(
            report,
            MetricThreshold(metric="finding.f1", minimum_value=0.8),
            MetricThreshold(metric="finding.f1", maximum_absolute_drop=0.1),
        )

    with pytest.raises(ValidationError, match="undefined_metric_behavior"):
        RegressionPolicy.model_validate(
            {
                "dataset_id": report.dataset_id,
                "dataset_version": report.dataset_version,
                "case_ids": tuple(case.case_id for case in report.metrics.per_case),
                "metric_definitions_version": report.metric_definitions_version,
                "thresholds": [{"metric": "finding.f1", "minimum_value": 0.8}],
            }
        )


def test_regression_json_is_deterministic() -> None:
    baseline = _report()
    candidate = _report(
        run_id="candidate-run",
        finding_change="extra_false_positive",
    )
    policy = _policy(
        baseline,
        MetricThreshold(
            metric="finding.precision",
            maximum_absolute_drop=0.05,
        ),
    )
    result = compare_evaluation_reports(baseline, candidate, policy)

    first = render_regression_json(result)
    second = render_regression_json(
        compare_evaluation_reports(baseline, candidate, policy)
    )

    assert first == second
    assert json.loads(first)["overall_outcome"] == "fail"


def test_compare_cli_returns_pass_and_regression_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = _report()
    passing_candidate = _report(run_id="passing-candidate", system_version="2.0.0")
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.f1", minimum_value=0.95),
    )
    baseline_path, candidate_path, policy_path = _write_comparison_inputs(
        tmp_path,
        baseline,
        passing_candidate,
        policy,
    )
    arguments = [
        "compare",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--policy",
        str(policy_path),
    ]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["overall_outcome"] == "pass"

    failing_candidate = _report(
        run_id="failing-candidate",
        finding_change="extra_false_positive",
    )
    candidate_path.write_text(render_json_report(failing_candidate), encoding="utf-8")

    assert main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["overall_outcome"] == "fail"


def test_compare_cli_returns_distinct_incompatible_and_inconclusive_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = _report()
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.precision", minimum_value=0.8),
    )
    baseline_path, candidate_path, policy_path = _write_comparison_inputs(
        tmp_path,
        baseline,
        _report(run_id="candidate-run", dataset=_dataset_with_version("1.0.1")),
        policy,
    )
    arguments = [
        "compare",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--policy",
        str(policy_path),
    ]

    assert main(arguments) == 3
    assert "dataset version" in capsys.readouterr().err.lower()

    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.precision", minimum_value=0.8),
        undefined_behavior="inconclusive",
    )
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    candidate_path.write_text(
        render_json_report(
            _report(run_id="undefined-candidate", finding_change="no_predictions")
        ),
        encoding="utf-8",
    )
    assert main(arguments) == 4
    assert json.loads(capsys.readouterr().out)["overall_outcome"] == "inconclusive"


def test_compare_cli_returns_inconclusive_code_when_policy_rejects_undefined(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = _report()
    policy = _policy(
        baseline,
        MetricThreshold(metric="finding.precision", minimum_value=0.8),
        undefined_behavior="reject",
    )
    baseline_path, candidate_path, policy_path = _write_comparison_inputs(
        tmp_path,
        baseline,
        _report(run_id="undefined-candidate", finding_change="no_predictions"),
        policy,
    )
    arguments = [
        "compare",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--policy",
        str(policy_path),
    ]

    assert main(arguments) == 4
    assert "undefined" in capsys.readouterr().err.lower()
