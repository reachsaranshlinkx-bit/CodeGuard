import pytest
from pydantic import ValidationError

from codeguard.github.schemas import GitHubFileContent

FILE_CONTENT_RESPONSE = {
    "type": "file",
    "encoding": "base64",
    "path": "src/example.py",
    "sha": "a" * 40,
    "size": 12,
    "content": "SGVsbG8gd29ybGQK",
}


def test_file_content_schema_validates_response() -> None:
    file_content = GitHubFileContent.model_validate(FILE_CONTENT_RESPONSE)

    assert file_content.type == "file"
    assert file_content.encoding == "base64"
    assert file_content.path == "src/example.py"
    assert file_content.sha == "a" * 40
    assert file_content.size == 12
    assert file_content.content == "SGVsbG8gd29ybGQK"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "directory"),
        ("encoding", "utf-8"),
        ("size", -1),
        ("content", None),
    ],
)
def test_file_content_schema_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    payload = FILE_CONTENT_RESPONSE.copy()
    payload[field] = value

    with pytest.raises(ValidationError):
        GitHubFileContent.model_validate(payload)


@pytest.mark.parametrize(
    "missing_field",
    [
        "type",
        "encoding",
        "path",
        "sha",
        "size",
        "content",
    ],
)
def test_file_content_schema_rejects_missing_fields(
    missing_field: str,
) -> None:
    payload = {
        key: value
        for key, value in FILE_CONTENT_RESPONSE.items()
        if key != missing_field
    }

    with pytest.raises(ValidationError):
        GitHubFileContent.model_validate(payload)


def test_file_content_schema_accepts_empty_file() -> None:
    payload = {
        **FILE_CONTENT_RESPONSE,
        "size": 0,
        "content": "",
    }

    file_content = GitHubFileContent.model_validate(payload)

    assert file_content.size == 0
    assert file_content.content == ""
