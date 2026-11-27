"""Structured JSON and human-readable Markdown evaluation reports."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codeguard.evaluation.contracts import (
    CaseIdentifier,
    EvaluationRun,
    NonBlankText,
)
from codeguard.evaluation.metrics import (
    CoreMetrics,
    EvaluationMetrics,
    calculate_evaluation_metrics,
)


class _ReportModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class EvaluationCaseLimitations(_ReportModel):
    """Dataset limitations attached to one case in the report."""

    case_id: CaseIdentifier
    limitations: tuple[NonBlankText, ...]


class EvaluationReport(_ReportModel):
    """Validated report metadata, metrics, and dataset limitations."""

    evaluation_run_id: CaseIdentifier
    dataset_id: CaseIdentifier
    dataset_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    system_name: NonBlankText
    system_version: NonBlankText
    started_at: datetime
    dataset_scope_statement: NonBlankText
    case_limitations: tuple[EvaluationCaseLimitations, ...]
    metrics: EvaluationMetrics

    @model_validator(mode="after")
    def validate_report_references(self) -> EvaluationReport:
        if self.evaluation_run_id != self.metrics.evaluation_run_id:
            raise ValueError("Report run ID must match its metrics run ID.")
        if self.dataset_id != self.metrics.dataset_id:
            raise ValueError("Report dataset ID must match its metrics dataset ID.")
        if self.dataset_version != self.metrics.dataset_version:
            raise ValueError(
                "Report dataset version must match its metrics dataset version."
            )
        metric_case_ids = tuple(case.case_id for case in self.metrics.per_case)
        limitation_case_ids = tuple(case.case_id for case in self.case_limitations)
        if metric_case_ids != limitation_case_ids:
            raise ValueError(
                "Report limitations must cover cases in the same order as metrics."
            )
        return self


def build_evaluation_report(run: EvaluationRun) -> EvaluationReport:
    """Calculate metrics and preserve metadata and limitations from the run."""

    metrics = calculate_evaluation_metrics(run)
    return EvaluationReport(
        evaluation_run_id=run.run_id,
        dataset_id=run.dataset.dataset_id,
        dataset_version=run.dataset.dataset_version,
        system_name=run.system_name,
        system_version=run.system_version,
        started_at=run.started_at,
        dataset_scope_statement=run.dataset.scope_statement,
        case_limitations=tuple(
            EvaluationCaseLimitations(
                case_id=case.case_id,
                limitations=case.analysis_limitations,
            )
            for case in run.dataset.cases
        ),
        metrics=metrics,
    )


def render_json_report(report: EvaluationReport) -> str:
    """Serialize a report with stable Pydantic field and dataset case order."""

    return f"{report.model_dump_json(indent=2)}\n"


def render_markdown_report(report: EvaluationReport) -> str:
    """Render a stable Markdown report from the same validated metrics model."""

    metrics = report.metrics
    recommendations = metrics.test_recommendation_metrics
    lines = [
        "# CodeGuard Evaluation Report",
        "",
        "## Evaluation metadata",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation run | {_markdown(report.evaluation_run_id)} |",
        f"| Dataset | {_markdown(report.dataset_id)} "
        f"v{_markdown(report.dataset_version)} |",
        f"| System | {_markdown(report.system_name)} |",
        f"| System version | {_markdown(report.system_version)} |",
        f"| Run timestamp | {_markdown(report.started_at.isoformat())} |",
        f"| Evaluated cases | {metrics.evaluated_case_count} |",
        "",
        "## Executive summary",
        "",
        "| Metric group | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _summary_row(
            "Findings",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.f1,
        ),
        _summary_row(
            "Test recommendations",
            recommendations.true_positives,
            recommendations.false_positives,
            recommendations.false_negatives,
            recommendations.precision,
            recommendations.recall,
            recommendations.f1,
        ),
        "",
        "## Finding quality metrics",
        "",
        _average_table(metrics.micro_averaged, metrics.macro_averaged),
        "",
        "| Quality measure | Accuracy / rate |",
        "| --- | ---: |",
        f"| Severity accuracy | {_format_metric(metrics.severity_accuracy)} |",
        f"| Exact evidence accuracy | {_format_metric(metrics.evidence_accuracy)} |",
        f"| Exact line-number accuracy | {_format_metric(metrics.line_accuracy)} |",
        "| Negative-case false-positive rate "
        f"| {_format_metric(metrics.negative_case_false_positive_rate)} |",
        "",
        "## Test recommendation quality",
        "",
        _average_table(
            recommendations.micro_averaged,
            recommendations.macro_averaged,
        ),
        "",
        "## Per-case results",
        "",
        "| Case | Findings TP/FP/FN | P / R / F1 | "
        "Recommendations TP/FP/FN | P / R / F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    recommendation_by_case = {case.case_id: case for case in recommendations.per_case}
    for finding_case in metrics.per_case:
        recommendation_case = recommendation_by_case[finding_case.case_id]
        finding_triplet = _metric_triplet(
            finding_case.precision,
            finding_case.recall,
            finding_case.f1,
        )
        recommendation_triplet = _metric_triplet(
            recommendation_case.precision,
            recommendation_case.recall,
            recommendation_case.f1,
        )
        lines.append(
            f"| {_markdown(finding_case.case_id)} "
            f"| {finding_case.true_positives}/{finding_case.false_positives}/"
            f"{finding_case.false_negatives} "
            f"| {finding_triplet} "
            f"| {recommendation_case.true_positives}/"
            f"{recommendation_case.false_positives}/"
            f"{recommendation_case.false_negatives} "
            f"| {recommendation_triplet} |"
        )

    lines.extend(
        [
            "",
            "## Dataset limitations",
            "",
            _markdown(report.dataset_scope_statement),
            "",
        ]
    )
    for case in report.case_limitations:
        limitations = "; ".join(_markdown(item) for item in case.limitations)
        lines.append(f"- **{_markdown(case.case_id)}:** {limitations}")
    lines.append("")
    return "\n".join(lines)


def _summary_row(
    label: str,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    precision: float | None,
    recall: float | None,
    f1: float | None,
) -> str:
    return (
        f"| {label} | {true_positives} | {false_positives} | {false_negatives} "
        f"| {_format_metric(precision)} | {_format_metric(recall)} "
        f"| {_format_metric(f1)} |"
    )


def _average_table(micro: CoreMetrics, macro: CoreMetrics) -> str:
    return "\n".join(
        [
            "| Average | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: |",
            f"| Micro | {_format_metric(micro.precision)} "
            f"| {_format_metric(micro.recall)} | {_format_metric(micro.f1)} |",
            f"| Macro | {_format_metric(macro.precision)} "
            f"| {_format_metric(macro.recall)} | {_format_metric(macro.f1)} |",
        ]
    )


def _metric_triplet(
    precision: float | None,
    recall: float | None,
    f1: float | None,
) -> str:
    return " / ".join(_format_metric(value) for value in (precision, recall, f1))


def _format_metric(value: float | None) -> str:
    if value is None:
        return "Undefined (zero denominator)"
    return f"{value:.1%}"


def _markdown(value: str) -> str:
    return value.replace("|", r"\|").replace("\r", " ").replace("\n", "<br>")
