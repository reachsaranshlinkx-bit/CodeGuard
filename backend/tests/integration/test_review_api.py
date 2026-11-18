import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.analyzers.python_risk import RiskFinding
from codeguard.api.dependencies import get_pull_request_analyzer
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


def make_analysis() -> PullRequestAnalysisResult:
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
    )


class FakePullRequestAnalyzer:
    """Fake analyzer for API integration tests."""

    def __init__(
        self,
        error: Exception | None = None,
    ) -> None:
        self.error = error
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

        return make_analysis()


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
