from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from codeguard.ai.contracts import AIFindingCandidate, AIReviewResult
from codeguard.ai.engine import AIReviewExecutionResult, AIReviewFindingAssessment
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.analyzers.python_risk import RiskFinding
from codeguard.api.dependencies import get_pull_request_analyzer
from codeguard.context.symbols import (
    RepositoryFileFailure,
    RepositorySymbolIndex,
)
from codeguard.core.config import Settings
from codeguard.github.exceptions import GitHubNotFoundError
from codeguard.main import create_app
from codeguard.services.pull_request_analyzer import (
    FileAnalysisResult,
    PullRequestAnalysisResult,
)

API_PAYLOAD = {
    "owner": "octocat",
    "repo": "CodeGuard",
    "pull_number": 42,
}

API_KEY = "test-codeguard-api-key-for-integration-tests"

API_HEADERS = {
    "X-CodeGuard-API-Key": API_KEY,
}


def make_analysis(
    repository_context: RepositorySymbolIndex | None = None,
) -> PullRequestAnalysisResult:
    finding = RiskFinding(
        rule_id="PY001",
        severity="high",
        message="eval() executes dynamically supplied code.",
        file_path="src/auth.py",
        lineno=2,
        cwe="CWE-95",
    )

    ast_result = PythonAnalysisResult(
        file_path="src/auth.py",
        functions=(),
        classes=(),
        imports=(),
        calls=(),
    )

    file_result = FileAnalysisResult(
        file_path="src/auth.py",
        status="modified",
        additions=1,
        deletions=0,
        changes=1,
        previous_filename=None,
        ast_result=ast_result,
        risk_findings=(finding,),
        added_line_numbers=frozenset({2}),
        changed_line_findings=(finding,),
        line_mapping_error=None,
    )

    return PullRequestAnalysisResult(
        owner="octocat",
        repo="CodeGuard",
        pull_number=42,
        title="Improve authentication",
        state="open",
        head_sha="abc123",
        base_sha="def456",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        files=(file_result,),
        repository_context=repository_context,
    )


def make_ai_execution_result(
    *,
    summary: str,
    assessments: tuple[AIReviewFindingAssessment, ...] = (),
) -> AIReviewExecutionResult:
    return AIReviewExecutionResult(
        provider_result=AIReviewResult(
            summary=summary,
            findings=tuple(item.candidate for item in assessments),
        ),
        assessments=assessments,
    )


class FakePullRequestAnalyzer:
    """Fake analyzer for API integration tests."""

    def __init__(
        self,
        error: Exception | None = None,
        analysis: PullRequestAnalysisResult | None = None,
    ) -> None:
        self.error = error
        self.analysis = analysis
        self.calls: list[tuple[str, str, int]] = []

    async def analyze(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> PullRequestAnalysisResult:
        self.calls.append((owner, repo, pull_number))

        if self.error is not None:
            raise self.error

        return self.analysis or make_analysis()


def create_test_client(
    analyzer: FakePullRequestAnalyzer,
) -> tuple[TestClient, object]:
    app = create_app(
        settings=Settings(
            _env_file=None,
            api_key=SecretStr(API_KEY),
        )
    )

    app.dependency_overrides[get_pull_request_analyzer] = lambda: analyzer

    return TestClient(app), app


def test_review_endpoint_returns_structured_report() -> None:
    analyzer = FakePullRequestAnalyzer()
    client, app = create_test_client(analyzer)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert response.status_code == 200

        data = response.json()

        assert data["owner"] == "octocat"
        assert data["repo"] == "CodeGuard"
        assert data["pull_number"] == 42

        assert data["total_findings"] == 1
        assert data["files_analyzed"] == 1

        assert data["severity_counts"]["high"] == 1

        finding = data["findings"][0]

        assert finding["rule_id"] == "PY001"
        assert finding["severity"] == "high"
        assert finding["line_number"] == 2
        assert finding["cwe"] == "CWE-95"

        assert analyzer.calls == [("octocat", "CodeGuard", 42)]
    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_returns_ai_summary_and_assessments_separately() -> None:
    candidate = AIFindingCandidate(
        title="Potential unsafe dynamic execution",
        severity="high",
        confidence="medium",
        file_path="src/auth.py",
        line_number=2,
        problem="The code executes dynamically supplied input.",
        evidence="eval(value)",
        why_it_matters="Untrusted input could execute arbitrary code.",
        suggested_fix="Use a safe parser for the expected input format.",
        recommended_validation="Add a test for untrusted input.",
    )
    assessment = AIReviewFindingAssessment(
        candidate=candidate,
        evidence_match=EvidenceMatchResult(
            status=EvidenceMatchStatus.MATCHED,
            reason=EvidenceMatchReason.EXACT_MATCH,
            file_path="src/auth.py",
            line_number=2,
        ),
    )
    analysis = replace(
        make_analysis(),
        ai_review_result=make_ai_execution_result(
            summary="One potential issue was assessed.",
            assessments=(assessment,),
        ),
    )
    analyzer = FakePullRequestAnalyzer(analysis=analysis)
    client, app = create_test_client(analyzer)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert response.status_code == 200
        data = response.json()

        assert data["ai_review_summary"] == "One potential issue was assessed."
        assert data["ai_assessments"] == [
            {
                "candidate": candidate.model_dump(mode="json"),
                "evidence_match": {
                    "status": "matched",
                    "reason": "exact_match",
                    "file_path": "src/auth.py",
                    "line_number": 2,
                    "location_reason": None,
                },
            }
        ]
        assert data["total_findings"] == 1
        assert data["severity_counts"]["high"] == 1
        assert len(data["findings"]) == 1
        assert data["findings"][0]["rule_id"] == "PY001"
    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_serializes_empty_ai_assessments() -> None:
    analysis = replace(
        make_analysis(),
        ai_review_result=make_ai_execution_result(
            summary="No additional potential issues were identified.",
        ),
    )
    analyzer = FakePullRequestAnalyzer(analysis=analysis)
    client, app = create_test_client(analyzer)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert response.status_code == 200
        data = response.json()

        assert data["ai_review_summary"] == (
            "No additional potential issues were identified."
        )
        assert data["ai_assessments"] == []
        assert data["total_findings"] == 1
        assert data["severity_counts"]["high"] == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "payload",
    [
        {"repo": "CodeGuard", "pull_number": 42},
        {"owner": "octocat", "repo": "CodeGuard", "pull_number": 0},
        {
            "owner": "octocat",
            "repo": "CodeGuard",
            "pull_number": 42,
            "token": "should-not-be-accepted",
        },
    ],
)
def test_review_endpoint_rejects_invalid_requests(
    payload: dict,
) -> None:
    analyzer = FakePullRequestAnalyzer()
    client, app = create_test_client(analyzer)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=payload,
                headers=API_HEADERS,
            )

        assert response.status_code == 422
        assert analyzer.calls == []
    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_maps_github_not_found_to_404() -> None:
    analyzer = FakePullRequestAnalyzer(
        error=GitHubNotFoundError(
            "Pull request not found",
            status_code=404,
        )
    )

    client, app = create_test_client(analyzer)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert response.status_code == 404
        assert response.json()["detail"] == (
            "GitHub repository or pull request not found."
        )
    finally:
        app.dependency_overrides.clear()


def test_review_endpoint_exposes_incomplete_repository_context() -> None:
    failure = RepositoryFileFailure(
        file_path="src/missing.py",
        stage="retrieval",
        error_type="GitHubNotFoundError",
        message="Internal retrieval error details.",
    )

    context = RepositorySymbolIndex(
        commit_sha="abc123",
        files=(),
        failures=(failure,),
    )

    analyzer = FakePullRequestAnalyzer(
        analysis=make_analysis(repository_context=context),
    )

    client, app = create_test_client(analyzer)

    try:
        with client:
            response = client.post(
                "/api/v1/reviews",
                json=API_PAYLOAD,
                headers=API_HEADERS,
            )

        assert response.status_code == 200

        data = response.json()
        context_data = data["repository_context"]

        assert context_data["commit_sha"] == "abc123"
        assert context_data["is_complete"] is False
        assert context_data["indexed_file_count"] == 0
        assert context_data["tree_truncated"] is False
        assert context_data["file_limit_reached"] is False

        assert context_data["failures"] == [
            {
                "file_path": "src/missing.py",
                "stage": "retrieval",
                "error_type": "GitHubNotFoundError",
            }
        ]

        assert "message" not in context_data["failures"][0]

    finally:
        app.dependency_overrides.clear()
