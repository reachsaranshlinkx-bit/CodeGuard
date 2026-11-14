import pytest
from pydantic import ValidationError

from codeguard.github.schemas import (
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubRepository,
)


def test_repository_schema_validates_response() -> None:
    payload = {
        "id": 123456,
        "name": "CodeGuard",
        "full_name": "owner/CodeGuard",
        "private": False,
        "html_url": "https://github.com/owner/CodeGuard",
        "default_branch": "main",
        "owner": {
            "login": "owner",
            "id": 100,
        },
        "unneeded_field": "ignored",
    }

    repository = GitHubRepository.model_validate(payload)

    assert repository.id == 123456
    assert repository.full_name == "owner/CodeGuard"
    assert repository.owner.login == "owner"
    assert repository.default_branch == "main"


def test_repository_schema_rejects_missing_required_fields() -> None:
    payload = {
        "id": 123456,
        "name": "CodeGuard",
    }

    with pytest.raises(ValidationError):
        GitHubRepository.model_validate(payload)


def test_pull_request_schema_validates_response() -> None:
    payload = {
        "url": "https://api.github.com/repos/owner/project/pulls/42",
        "html_url": "https://github.com/owner/project/pull/42",
        "number": 42,
        "state": "open",
        "title": "Fix authentication",
        "body": "Fix an authentication issue.",
        "draft": False,
        "user": {
            "login": "developer",
        },
        "base": {
            "ref": "main",
            "sha": "a" * 40,
        },
        "head": {
            "ref": "feature/authentication",
            "sha": "b" * 40,
        },
    }

    pull_request = GitHubPullRequest.model_validate(payload)

    assert pull_request.number == 42
    assert pull_request.title == "Fix authentication"
    assert pull_request.base.ref == "main"
    assert pull_request.head.ref == "feature/authentication"


def test_pull_request_file_supports_missing_patch() -> None:
    payload = {
        "filename": "src/auth.py",
        "status": "modified",
        "additions": 5,
        "deletions": 2,
        "changes": 7,
    }

    changed_file = GitHubPullRequestFile.model_validate(payload)

    assert changed_file.filename == "src/auth.py"
    assert changed_file.status == "modified"
    assert changed_file.patch is None


def test_pull_request_file_rejects_negative_additions() -> None:
    payload = {
        "filename": "src/auth.py",
        "status": "modified",
        "additions": -1,
        "deletions": 0,
        "changes": 1,
    }

    with pytest.raises(ValidationError):
        GitHubPullRequestFile.model_validate(payload)


def test_pull_request_file_rejects_invalid_status() -> None:
    payload = {
        "filename": "src/auth.py",
        "status": "unknown",
        "additions": 1,
        "deletions": 0,
        "changes": 1,
    }

    with pytest.raises(ValidationError):
        GitHubPullRequestFile.model_validate(payload)
