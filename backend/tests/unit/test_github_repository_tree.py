import pytest
from pydantic import ValidationError

from codeguard.github.schemas import GitHubRepositoryTree


def test_repository_tree_schema_validates_entries() -> None:
    payload = {
        "sha": "a" * 40,
        "truncated": False,
        "tree": [
            {
                "path": "src",
                "mode": "040000",
                "type": "tree",
                "sha": "b" * 40,
            },
            {
                "path": "src/example.py",
                "mode": "100644",
                "type": "blob",
                "sha": "c" * 40,
                "size": 128,
            },
        ],
    }

    result = GitHubRepositoryTree.model_validate(payload)

    assert result.sha == "a" * 40
    assert result.truncated is False
    assert len(result.tree) == 2
    assert result.tree[0].type == "tree"
    assert result.tree[1].path == "src/example.py"
    assert result.tree[1].size == 128


def test_repository_tree_schema_preserves_truncated_flag() -> None:
    payload = {
        "sha": "a" * 40,
        "truncated": True,
        "tree": [],
    }

    result = GitHubRepositoryTree.model_validate(payload)

    assert result.truncated is True


def test_repository_tree_schema_rejects_invalid_entries() -> None:
    payload = {
        "sha": "a" * 40,
        "truncated": False,
        "tree": [
            {
                "path": "src/example.py",
                "mode": "100644",
                "type": "invalid",
                "sha": "b" * 40,
            }
        ],
    }

    with pytest.raises(ValidationError):
        GitHubRepositoryTree.model_validate(payload)
