import hashlib
import hmac
import json
import time
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
from pydantic import SecretStr

import codeguard.main as main
from codeguard.ai.contracts import AIFindingCandidate, AIReviewResult
from codeguard.ai.engine import (
    AIReviewExecutionResult,
    AIReviewFindingAssessment,
)
from codeguard.ai.evidence_matching import (
    EvidenceMatchReason,
    EvidenceMatchResult,
    EvidenceMatchStatus,
)
from codeguard.analyzers.python_ast import PythonAnalysisResult
from codeguard.core.config import Settings
from codeguard.review.report_builder import build_review_report
from codeguard.services.pull_request_analyzer import (
    FileAnalysisResult,
    PullRequestAnalysisResult,
)
from codeguard.services.webhook_job_worker import WebhookJobWorker

SECRET = "test-webhook-worker-secret"


class FakeGitHubClient:
    """Fake GitHub client for webhook worker integration tests."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.closed = False
        self.review_lookup_calls: list[tuple[str, str, int]] = []
        self.created_reviews: list[dict[str, object]] = []
        self.review_created_event = Event()

    async def aclose(self) -> None:
        self.closed = True

    async def get_all_pull_request_reviews(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> list[dict[str, object]]:
        self.review_lookup_calls.append((owner, repo, pull_number))
        return []

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, str | int]],
    ) -> dict[str, object]:
        self.created_reviews.append(
            {
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
                "commit_id": commit_id,
                "body": body,
                "comments": comments,
            }
        )
        self.review_created_event.set()
        return {"id": 123, "state": "COMMENTED"}


class FakeReviewProcessor:
    """Fake review processor that records processed pull requests."""

    def __init__(
        self,
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> None:
        self.arguments = None
        self.processed_event = Event()

    async def process(
        self,
        *,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> None:
        self.arguments = (owner, repo, pull_number)
        self.processed_event.set()


def test_webhook_is_received_and_processed_by_background_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors: list[FakeReviewProcessor] = []

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> FakeReviewProcessor:
        processor = FakeReviewProcessor(
            analyzer=analyzer,
            github_client=github_client,
            publishing_enabled=publishing_enabled,
        )
        processors.append(processor)
        return processor

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        fake_processor_factory,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "webhook_worker.sqlite3",
    )

    app = main.create_app(settings=settings)

    delivery_id = "webhook-worker-integration-001"

    payload = {
        "action": "opened",
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "octocat",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert response.json()["delivery_id"] == delivery_id

        assert len(processors) == 1

        processor = processors[0]

        assert processor.processed_event.wait(timeout=5)

        assert processor.arguments == (
            "octocat",
            "CodeGuard",
            42,
        )

        deadline = time.monotonic() + 3
        job = app.state.webhook_job_queue.get_job(delivery_id)

        while (
            job is not None
            and job.status != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = app.state.webhook_job_queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "completed"
        assert job.attempts == 1
        assert job.lease_expires_at is None

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_webhook_worker_retries_failed_processing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors = []
    succeeded_event = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    class FlakyReviewProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            self.calls = 0

        async def process(
            self,
            *,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> None:
            self.calls += 1

            if self.calls == 1:
                raise RuntimeError("Temporary GitHub API error")

            succeeded_event.set()

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> FlakyReviewProcessor:
        processor = FlakyReviewProcessor(
            analyzer=analyzer,
            github_client=github_client,
            publishing_enabled=publishing_enabled,
        )
        processors.append(processor)
        return processor

    original_worker_class = main.WebhookJobWorker

    def fake_worker_factory(
        *,
        queue,
        processor,
        max_attempts: int,
        retry_delay_seconds: int,
    ) -> WebhookJobWorker:
        return original_worker_class(
            queue=queue,
            processor=processor,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        fake_processor_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookJobWorker",
        fake_worker_factory,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_worker_retry_delay_seconds=0,
        webhook_database_path=tmp_path / "webhook_retry.sqlite3",
    )

    app = main.create_app(settings=settings)

    delivery_id = "webhook-retry-integration-001"

    payload = {
        "action": "opened",
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "octocat",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    queue = app.state.webhook_job_queue

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"

        assert succeeded_event.wait(timeout=5)

        assert len(processors) == 1
        assert processors[0].calls == 2

        deadline = time.monotonic() + 3
        job = queue.get_job(delivery_id)

        while (
            job is not None
            and job.status != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "completed"
        assert job.attempts == 2
        assert job.last_error is None
        assert job.lease_expires_at is None

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_webhook_worker_stops_after_maximum_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_clients: list[FakeGitHubClient] = []
    processors = []
    second_attempt_event = Event()

    def fake_client_factory(
        token: str | None = None,
    ) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    class AlwaysFailingProcessor:
        def __init__(
            self,
            *,
            analyzer,
            github_client,
            publishing_enabled: bool,
        ) -> None:
            self.calls = 0

        async def process(
            self,
            *,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> None:
            self.calls += 1

            if self.calls == 2:
                second_attempt_event.set()

            raise RuntimeError("Persistent processing failure")

    def fake_processor_factory(
        *,
        analyzer,
        github_client,
        publishing_enabled: bool,
    ) -> AlwaysFailingProcessor:
        processor = AlwaysFailingProcessor(
            analyzer=analyzer,
            github_client=github_client,
            publishing_enabled=publishing_enabled,
        )
        processors.append(processor)
        return processor

    original_worker_class = main.WebhookJobWorker

    def fake_worker_factory(
        *,
        queue,
        processor,
        max_attempts: int,
        retry_delay_seconds: int,
    ) -> WebhookJobWorker:
        return original_worker_class(
            queue=queue,
            processor=processor,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    monkeypatch.setattr(
        main,
        "GitHubClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookReviewProcessor",
        fake_processor_factory,
    )
    monkeypatch.setattr(
        main,
        "WebhookJobWorker",
        fake_worker_factory,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_worker_max_attempts=2,
        webhook_worker_retry_delay_seconds=0,
        webhook_database_path=tmp_path / "webhook_exhaustion.sqlite3",
    )

    app = main.create_app(settings=settings)

    delivery_id = "webhook-exhaustion-integration-001"

    payload = {
        "action": "opened",
        "repository": {
            "name": "CodeGuard",
            "owner": {
                "login": "octocat",
            },
        },
        "pull_request": {
            "number": 42,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    queue = app.state.webhook_job_queue

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"

        assert second_attempt_event.wait(timeout=5)

        deadline = time.monotonic() + 3
        job = queue.get_job(delivery_id)

        while (
            job is not None and job.status != "failed" and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "failed"
        assert job.attempts == 2
        assert job.lease_expires_at is None
        assert job.last_error == "Persistent processing failure"

        assert len(processors) == 1
        assert processors[0].calls == 2

    assert len(created_clients) == 1
    assert created_clients[0].closed is True


def test_signed_webhook_publishes_only_matched_ai_assessments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    owner = "octocat"
    repo = "CodeGuard"
    pull_number = 42
    head_sha = "b" * 40
    line_number = 12

    matched_candidate = AIFindingCandidate(
        title="Potential unsafe value handling",
        severity="medium",
        confidence="medium",
        file_path="src/auth.py",
        line_number=line_number,
        problem="The returned value may not be validated.",
        evidence="return value",
        why_it_matters="Unexpected values may reach downstream code.",
        suggested_fix="Validate the value before returning it.",
        recommended_validation="Add tests for malformed values.",
    )
    mismatched_candidate = AIFindingCandidate(
        title="Unverified candidate must not be published",
        severity="high",
        confidence="low",
        file_path="src/auth.py",
        line_number=line_number,
        problem="This assessment did not match the source.",
        evidence="return unknown",
        why_it_matters="Unverified evidence must not produce a comment.",
        suggested_fix="Recheck the source before suggesting a change.",
        recommended_validation="Confirm the source location.",
    )
    matched_assessment = AIReviewFindingAssessment(
        candidate=matched_candidate,
        evidence_match=EvidenceMatchResult(
            status=EvidenceMatchStatus.MATCHED,
            reason=EvidenceMatchReason.EXACT_MATCH,
            file_path=matched_candidate.file_path,
            line_number=matched_candidate.line_number,
        ),
    )
    mismatched_assessment = AIReviewFindingAssessment(
        candidate=mismatched_candidate,
        evidence_match=EvidenceMatchResult(
            status=EvidenceMatchStatus.MISMATCHED,
            reason=EvidenceMatchReason.CONTENT_MISMATCH,
            file_path=mismatched_candidate.file_path,
            line_number=mismatched_candidate.line_number,
        ),
    )
    assessments = (matched_assessment, mismatched_assessment)
    analysis = PullRequestAnalysisResult(
        owner=owner,
        repo=repo,
        pull_number=pull_number,
        title="Improve authentication",
        state="open",
        head_sha=head_sha,
        base_sha="a" * 40,
        html_url=f"https://github.com/{owner}/{repo}/pull/{pull_number}",
        files=(
            FileAnalysisResult(
                file_path="src/auth.py",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
                previous_filename=None,
                ast_result=PythonAnalysisResult(
                    file_path="src/auth.py",
                    functions=(),
                    classes=(),
                    imports=(),
                    calls=(),
                ),
                risk_findings=(),
                added_line_numbers=frozenset({line_number}),
                changed_line_findings=(),
                line_mapping_error=None,
            ),
        ),
        ai_review_result=AIReviewExecutionResult(
            provider_result=AIReviewResult(
                summary="Two candidates were assessed.",
                findings=(matched_candidate, mismatched_candidate),
            ),
            assessments=assessments,
        ),
    )
    expected_report = build_review_report(analysis)
    assert expected_report.findings == ()
    assert expected_report.ai_assessments == assessments

    class FakeAnalyzer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []
            self.analyzed_event = Event()

        async def analyze(
            self,
            owner: str,
            repo: str,
            pull_number: int,
        ) -> PullRequestAnalysisResult:
            self.calls.append((owner, repo, pull_number))
            self.analyzed_event.set()
            return analysis

    analyzer = FakeAnalyzer()
    created_clients: list[FakeGitHubClient] = []

    def fake_client_factory(token: str | None = None) -> FakeGitHubClient:
        client = FakeGitHubClient(token=token)
        created_clients.append(client)
        return client

    monkeypatch.setattr(main, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(
        main,
        "PullRequestAnalyzer",
        lambda **kwargs: analyzer,
    )

    settings = Settings(
        _env_file=None,
        github_webhook_secret=SecretStr(SECRET),
        enable_review_publishing=True,
        webhook_worker_enabled=True,
        webhook_worker_poll_interval_seconds=1,
        webhook_database_path=tmp_path / "webhook_ai_publish.sqlite3",
    )
    app = main.create_app(settings=settings)
    delivery_id = "webhook-ai-publish-integration-001"
    payload = {
        "action": "opened",
        "repository": {
            "name": repo,
            "owner": {"login": owner},
        },
        "pull_request": {"number": pull_number},
    }
    body = json.dumps(payload).encode("utf-8")
    digest = hmac.new(
        SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery_id,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert response.json()["delivery_id"] == delivery_id
        assert analyzer.analyzed_event.wait(timeout=5)

        github = created_clients[0]
        assert github.review_created_event.wait(timeout=5)

        deadline = time.monotonic() + 3
        job = app.state.webhook_job_queue.get_job(delivery_id)

        while (
            job is not None
            and job.status != "completed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = app.state.webhook_job_queue.get_job(delivery_id)

        assert job is not None
        assert job.status == "completed"
        assert job.attempts == 1

    assert analyzer.calls == [(owner, repo, pull_number)]
    assert len(created_clients) == 1

    github = created_clients[0]
    assert github.review_lookup_calls == [(owner, repo, pull_number)]
    assert len(github.created_reviews) == 1
    assert github.closed is True

    published = github.created_reviews[0]
    assert published["owner"] == owner
    assert published["repo"] == repo
    assert published["pull_number"] == pull_number
    assert published["commit_id"] == head_sha
    comments = published["comments"]
    assert isinstance(comments, list)
    assert len(comments) == 1
    assert comments[0]["path"] == "src/auth.py"
    assert comments[0]["line"] == line_number
    assert comments[0]["side"] == "RIGHT"
    assert matched_candidate.title in comments[0]["body"]
    assert matched_candidate.evidence in comments[0]["body"]
    assert mismatched_candidate.title not in comments[0]["body"]
