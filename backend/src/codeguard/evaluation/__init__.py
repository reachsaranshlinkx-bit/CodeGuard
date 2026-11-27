"""Provider-neutral evaluation contracts and golden datasets."""

from codeguard.evaluation.contracts import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    ExpectedFinding,
    FindingMatch,
    TestScenarioMatch,
    load_evaluation_dataset,
)
from codeguard.evaluation.metrics import (
    CaseEvaluationMetrics,
    CoreMetrics,
    EvaluationMetrics,
    TestRecommendationCaseMetrics,
    TestRecommendationMetrics,
    calculate_evaluation_metrics,
)
from codeguard.evaluation.reports import (
    EvaluationCaseLimitations,
    EvaluationReport,
    build_evaluation_report,
    render_json_report,
    render_markdown_report,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "CaseEvaluationMetrics",
    "CoreMetrics",
    "EvaluationCase",
    "EvaluationCaseLimitations",
    "EvaluationDataset",
    "EvaluationMetrics",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationRun",
    "ExpectedFinding",
    "FindingMatch",
    "TestRecommendationCaseMetrics",
    "TestRecommendationMetrics",
    "TestScenarioMatch",
    "calculate_evaluation_metrics",
    "build_evaluation_report",
    "load_evaluation_dataset",
    "render_json_report",
    "render_markdown_report",
]
