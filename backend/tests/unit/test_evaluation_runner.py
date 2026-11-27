import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

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
from codeguard.evaluation.reports import (
    build_evaluation_report,
    render_json_report,
    render_markdown_report,
)
from codeguard.evaluation.runner import (
    IncompleteEvaluationRunError,
    evaluate_run_file,
    load_evaluation_run,
    main,
    serialize_evaluation_run,
)


def _synthetic_run(
    *,
    dataset: EvaluationDataset | None = None,
    is_complete: bool = True,
) -> EvaluationRun:
    evaluation_dataset = dataset or load_evaluation_dataset()
    results = tuple(
        EvaluationResult(
            case_id=case.case_id,
            predicted_findings=tuple(
                expected.finding for expected in case.expected_findings
            ),
            finding_matches=tuple(
                FindingMatch(
                    predicted_finding_index=index,
                    ground_truth_finding_id=expected.finding_id,
                )
                for index, expected in enumerate(case.expected_findings)
            ),
            predicted_test_scenarios=case.recommended_test_scenarios,
            test_scenario_matches=tuple(
                ScenarioMatch(
                    predicted_scenario_index=index,
                    expected_scenario_index=index,
                )
                for index in range(len(case.recommended_test_scenarios))
            ),
        )
        for case in evaluation_dataset.cases
    )
    return EvaluationRun(
        run_id="runner-synthetic-run",
        dataset=evaluation_dataset,
        started_at=datetime(2026, 11, 27, 9, 15, tzinfo=UTC),
        system_name="deterministic-fixture",
        system_version="2.4.0",
        is_complete=is_complete,
        results=results,
    )


def _write_run(path: Path, run: EvaluationRun) -> str:
    artifact = serialize_evaluation_run(run)
    path.write_text(artifact, encoding="utf-8")
    return artifact


def _historical_dataset() -> EvaluationDataset:
    payload: dict[str, Any] = json.loads(load_evaluation_dataset().model_dump_json())
    payload["dataset_version"] = "9.8.7"
    payload["scope_statement"] = "Historical fixture snapshot with a fixed scope."
    return EvaluationDataset.model_validate_json(json.dumps(payload))


def test_valid_run_artifact_preserves_its_embedded_dataset_snapshot(
    tmp_path: Path,
) -> None:
    historical_dataset = _historical_dataset()
    run = _synthetic_run(dataset=historical_dataset)
    input_path = tmp_path / "evaluation_run.json"
    artifact = _write_run(input_path, run)

    loaded_run = load_evaluation_run(input_path)
    report = evaluate_run_file(input_path)

    assert loaded_run == run
    assert "recommended_test_scenarios" in artifact
    assert loaded_run.dataset == historical_dataset
    assert report.dataset_id == historical_dataset.dataset_id
    assert report.dataset_version == "9.8.7"
    assert report.started_at == run.started_at


def test_runner_rejects_a_missing_input_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--input", str(tmp_path / "missing.json")])

    assert exit_code != 0
    assert "input file" in capsys.readouterr().err.lower()


def test_runner_rejects_invalid_json_without_echoing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text('{"private-marker":', encoding="utf-8")

    exit_code = main(["--input", str(input_path)])
    error_output = capsys.readouterr().err

    assert exit_code != 0
    assert "not valid json" in error_output.lower()
    assert "private-marker" not in error_output


def test_runner_rejects_invalid_evaluation_run_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload: dict[str, Any] = json.loads(_synthetic_run().model_dump_json())
    payload["system_name"] = 42
    input_path = tmp_path / "invalid-contract.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["--input", str(input_path)])
    error_output = capsys.readouterr().err

    assert exit_code != 0
    assert "EvaluationRun contract" in error_output
    assert "42" not in error_output


def test_runner_rejects_incomplete_run_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "incomplete.json"
    _write_run(input_path, _synthetic_run(is_complete=False))

    exit_code = main(["--input", str(input_path)])

    assert exit_code != 0
    assert "marked incomplete" in capsys.readouterr().err
    with pytest.raises(IncompleteEvaluationRunError):
        load_evaluation_run(input_path)


def test_structured_report_contains_correct_finding_and_recommendation_metrics(
    tmp_path: Path,
) -> None:
    run = _synthetic_run()
    input_path = tmp_path / "evaluation_run.json"
    _write_run(input_path, run)

    report = evaluate_run_file(input_path)
    expected_findings = sum(len(case.expected_findings) for case in run.dataset.cases)
    expected_scenarios = sum(
        len(case.recommended_test_scenarios) for case in run.dataset.cases
    )

    assert report.evaluation_run_id == run.run_id
    assert report.system_name == run.system_name
    assert report.system_version == run.system_version
    assert report.metrics.true_positives == expected_findings
    assert report.metrics.false_positives == 0
    assert report.metrics.false_negatives == 0
    assert report.metrics.test_recommendation_metrics.true_positives == (
        expected_scenarios
    )
    assert report.metrics.test_recommendation_metrics.false_positives == 0
    assert report.metrics.test_recommendation_metrics.false_negatives == 0


def test_json_report_serializes_the_same_metrics_model(
    tmp_path: Path,
) -> None:
    report = build_evaluation_report(_synthetic_run())

    serialized_report = render_json_report(report)
    payload = json.loads(serialized_report)

    assert payload["evaluation_run_id"] == report.evaluation_run_id
    assert payload["metrics"] == report.metrics.model_dump(mode="json")
    assert payload["metrics"]["test_recommendation_metrics"]["micro_averaged"]
    assert payload["metrics"]["test_recommendation_metrics"]["macro_averaged"]


def test_markdown_report_contains_metadata_metrics_cases_and_limitations() -> None:
    report = build_evaluation_report(_synthetic_run())

    markdown = render_markdown_report(report)

    assert "## Evaluation metadata" in markdown
    assert "## Executive summary" in markdown
    assert "## Finding quality metrics" in markdown
    assert "## Test recommendation quality" in markdown
    assert "## Per-case results" in markdown
    assert "## Dataset limitations" in markdown
    assert report.dataset_scope_statement in markdown
    assert report.case_limitations[0].case_id in markdown


def test_cli_writes_json_output_without_modifying_input(
    tmp_path: Path,
) -> None:
    run = _synthetic_run()
    input_path = tmp_path / "evaluation_run.json"
    original_input = _write_run(input_path, run)
    output_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["evaluation_run_id"] == (
        run.run_id
    )
    assert input_path.read_text(encoding="utf-8") == original_input


def test_cli_writes_markdown_to_standard_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "evaluation_run.json"
    _write_run(input_path, _synthetic_run())

    exit_code = main(["--input", str(input_path), "--format", "markdown"])

    assert exit_code == 0
    assert capsys.readouterr().out.startswith("# CodeGuard Evaluation Report")


def test_cli_rejects_output_that_would_overwrite_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _synthetic_run()
    input_path = tmp_path / "evaluation_run.json"
    original_input = _write_run(input_path, run)

    exit_code = main(["--input", str(input_path), "--output", str(input_path)])

    assert exit_code != 0
    assert "must not be the input file" in capsys.readouterr().err
    assert input_path.read_text(encoding="utf-8") == original_input


def test_cli_reports_an_invalid_output_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "evaluation_run.json"
    _write_run(input_path, _synthetic_run())
    output_path = tmp_path / "missing-directory" / "report.json"

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    assert exit_code != 0
    assert "Could not write report" in capsys.readouterr().err
    assert not output_path.exists()


def test_report_serialization_is_reproducible(tmp_path: Path) -> None:
    input_path = tmp_path / "evaluation_run.json"
    _write_run(input_path, _synthetic_run())

    first_report = evaluate_run_file(input_path)
    second_report = evaluate_run_file(input_path)

    assert render_json_report(first_report) == render_json_report(second_report)
    assert render_markdown_report(first_report) == render_markdown_report(second_report)


def test_json_and_markdown_label_undefined_metrics(
    tmp_path: Path,
) -> None:
    dataset_payload: dict[str, Any] = json.loads(
        load_evaluation_dataset().model_dump_json()
    )
    target_case = next(
        case
        for case in dataset_payload["cases"]
        if case["case_id"] == "safe-subprocess-arguments"
    )
    target_case["recommended_test_scenarios"] = []
    dataset = EvaluationDataset.model_validate_json(json.dumps(dataset_payload))
    report = build_evaluation_report(_synthetic_run(dataset=dataset))
    input_path = tmp_path / "undefined-metrics-run.json"
    _write_run(input_path, _synthetic_run(dataset=dataset))

    case_metrics = next(
        case
        for case in report.metrics.test_recommendation_metrics.per_case
        if case.case_id == target_case["case_id"]
    )
    json_payload = json.loads(render_json_report(report))
    markdown = render_markdown_report(report)
    serialized_case = next(
        case
        for case in json_payload["metrics"]["test_recommendation_metrics"]["per_case"]
        if case["case_id"] == target_case["case_id"]
    )

    assert case_metrics.precision is None
    assert serialized_case["precision"] is None
    assert "Undefined (zero denominator)" in markdown


def test_cli_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "--input" in capsys.readouterr().out
