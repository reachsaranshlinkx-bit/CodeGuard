"""Provider-neutral evaluation contracts and golden datasets."""

from codeguard.evaluation.contracts import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    ExpectedFinding,
    FindingMatch,
    load_evaluation_dataset,
)
from codeguard.evaluation.metrics import (
    CaseEvaluationMetrics,
    CoreMetrics,
    EvaluationMetrics,
    calculate_evaluation_metrics,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "CaseEvaluationMetrics",
    "CoreMetrics",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationMetrics",
    "EvaluationResult",
    "EvaluationRun",
    "ExpectedFinding",
    "FindingMatch",
    "calculate_evaluation_metrics",
    "load_evaluation_dataset",
]
