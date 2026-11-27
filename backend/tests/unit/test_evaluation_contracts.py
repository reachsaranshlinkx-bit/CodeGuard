import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from codeguard.evaluation.contracts import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    FindingMatch,
    load_evaluation_dataset,
)


def dataset_payload() -> dict[str, Any]:
    dataset = load_evaluation_dataset()
    return json.loads(dataset.model_dump_json())


def validate_dataset_payload(payload: dict[str, Any]) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(json.dumps(payload))


def results_for_all_cases(
    dataset: EvaluationDataset,
    result: EvaluationResult,
) -> tuple[EvaluationResult, ...]:
    return tuple(
        result
        if case.case_id == result.case_id
        else EvaluationResult(case_id=case.case_id, predicted_findings=())
        for case in dataset.cases
    )


def test_packaged_golden_dataset_loads_with_expected_scope_and_categories() -> None:
    dataset = load_evaluation_dataset()
    category_counts = Counter(case.category for case in dataset.cases)

    assert dataset.schema_version == EVALUATION_SCHEMA_VERSION
    assert dataset.dataset_id == "python-pr-golden"
    assert dataset.dataset_version == "1.0.0"
    assert len(dataset.cases) == 10
    assert len({case.case_id for case in dataset.cases}) == 10
    assert category_counts["security"] == 4
    assert category_counts["reliability"] == 1
    assert category_counts["error_handling"] == 1
    assert category_counts["safe_pattern"] == 2
    assert category_counts["edge_case"] == 1
    assert category_counts["negative"] == 1
    assert all(
        not case.expected_findings
        for case in dataset.cases
        if case.case_type == "negative"
    )
    assert "does not establish" in dataset.scope_statement


def test_dataset_loader_reads_a_temporary_dataset_file(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset_payload()), encoding="utf-8")

    loaded = load_evaluation_dataset(path)

    assert loaded == load_evaluation_dataset()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("title"),
        lambda payload: payload.update(unexpected_field=True),
        lambda payload: payload.update(schema_version="2.0"),
        lambda payload: payload.update(dataset_version="not-a-version"),
    ],
)
def test_dataset_rejects_incomplete_or_incompatible_metadata(mutate) -> None:
    payload = dataset_payload()
    mutate(payload)

    with pytest.raises(ValidationError):
        validate_dataset_payload(payload)


def test_dataset_rejects_duplicate_case_ids() -> None:
    payload = dataset_payload()
    payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        validate_dataset_payload(payload)


def test_case_rejects_duplicate_ground_truth_ids() -> None:
    payload = dataset_payload()
    finding = deepcopy(payload["cases"][0]["expected_findings"][0])
    payload["cases"][0]["expected_findings"].append(finding)

    with pytest.raises(
        ValidationError, match="Ground-truth finding IDs must be unique"
    ):
        validate_dataset_payload(payload)


def test_ground_truth_must_reference_a_changed_source_file() -> None:
    payload = dataset_payload()
    payload["cases"][0]["expected_findings"][0]["finding"]["file_path"] = (
        "src/missing.py"
    )

    with pytest.raises(
        ValidationError, match="must reference an added or modified file"
    ):
        validate_dataset_payload(payload)


def test_ground_truth_source_line_must_be_added_by_the_pull_request() -> None:
    payload = dataset_payload()
    finding = payload["cases"][0]["expected_findings"][0]["finding"]
    finding["line_number"] = 1
    finding["evidence"] = "def parse_expression(value: str) -> int:"

    with pytest.raises(ValidationError, match="must reference an added source line"):
        validate_dataset_payload(payload)


def test_ground_truth_rejects_incorrect_evidence() -> None:
    payload = dataset_payload()
    payload["cases"][0]["expected_findings"][0]["finding"]["evidence"] = (
        "    return eval(other)"
    )

    with pytest.raises(
        ValidationError, match="invalid source evidence or line reference"
    ):
        validate_dataset_payload(payload)


def test_ground_truth_rejects_invalid_source_line() -> None:
    payload = dataset_payload()
    payload["cases"][0]["expected_findings"][0]["finding"]["line_number"] = 100

    with pytest.raises(
        ValidationError, match="invalid source evidence or line reference"
    ):
        validate_dataset_payload(payload)


def test_negative_case_cannot_contain_a_ground_truth_finding() -> None:
    payload = dataset_payload()
    negative_case = payload["cases"][6]
    negative_case["expected_findings"] = [
        {
            "finding_id": "unexpected-label",
            "finding": {
                "title": "Incorrectly labelled shell issue",
                "severity": "high",
                "confidence": "high",
                "file_path": "src/tool_runner.py",
                "line_number": 4,
                "problem": "An invalid label was added to this negative example.",
                "evidence": '    subprocess.run(["tool", name], check=True)',
                "why_it_matters": "The example intentionally has no supported issue.",
                "suggested_fix": "Remove the invalid ground-truth label.",
                "recommended_validation": "Check the expected labels for this case.",
            },
            "rationale": "This label is intentionally invalid.",
        }
    ]

    with pytest.raises(
        ValidationError, match="Negative cases cannot contain labelled findings"
    ):
        validate_dataset_payload(payload)


def test_evaluation_run_validates_case_and_ground_truth_references() -> None:
    dataset = load_evaluation_dataset()
    positive_case = dataset.cases[0]
    expected = positive_case.expected_findings[0]
    result = EvaluationResult(
        case_id=positive_case.case_id,
        predicted_findings=(expected.finding,),
        finding_matches=(
            FindingMatch(
                predicted_finding_index=0,
                ground_truth_finding_id=expected.finding_id,
            ),
        ),
    )
    run = EvaluationRun(
        run_id="manual-evaluation-001",
        dataset=dataset,
        started_at=datetime(2026, 11, 27, tzinfo=UTC),
        system_name="synthetic-test-system",
        system_version="0.1.0",
        results=results_for_all_cases(dataset, result),
    )

    assert run.results[0].finding_matches[0].ground_truth_finding_id == (
        "dynamic-execution"
    )

    invalid_result = EvaluationResult(
        case_id=positive_case.case_id,
        predicted_findings=(expected.finding,),
        finding_matches=(
            FindingMatch(
                predicted_finding_index=0,
                ground_truth_finding_id="missing-ground-truth",
            ),
        ),
    )
    with pytest.raises(ValidationError, match="unknown ground-truth finding"):
        EvaluationRun(
            run_id="manual-evaluation-002",
            dataset=dataset,
            started_at=datetime(2026, 11, 27, tzinfo=UTC),
            system_name="synthetic-test-system",
            system_version="0.1.0",
            results=results_for_all_cases(dataset, invalid_result),
        )


def test_evaluation_run_rejects_unknown_case_ids() -> None:
    dataset = load_evaluation_dataset()
    result = EvaluationResult(case_id="unknown-case", predicted_findings=())

    with pytest.raises(ValidationError, match="references unknown case"):
        EvaluationRun(
            run_id="manual-evaluation-003",
            dataset=dataset,
            started_at=datetime(2026, 11, 27, tzinfo=UTC),
            system_name="synthetic-test-system",
            system_version="0.1.0",
            results=(result,),
        )


def test_evaluation_run_rejects_out_of_range_prediction_match_indexes() -> None:
    dataset = load_evaluation_dataset()
    case = dataset.cases[0]
    prediction = case.expected_findings[0].finding

    result = EvaluationResult(
        case_id=case.case_id,
        predicted_findings=(prediction,),
        finding_matches=(
            FindingMatch(
                predicted_finding_index=1,
                ground_truth_finding_id=case.expected_findings[0].finding_id,
            ),
        ),
    )

    with pytest.raises(
        ValidationError,
        match="Finding match indexes must reference a predicted finding",
    ):
        EvaluationRun(
            run_id="manual-evaluation-negative-index",
            dataset=dataset,
            started_at=datetime(2026, 11, 27, tzinfo=UTC),
            system_name="synthetic-test-system",
            system_version="0.1.0",
            is_complete=False,
            results=(result,),
        )
