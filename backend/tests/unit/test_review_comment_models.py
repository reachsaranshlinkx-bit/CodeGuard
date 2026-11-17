from dataclasses import FrozenInstanceError

import pytest

from codeguard.review.comment_models import InlineReviewComment


def test_inline_review_comment_accepts_valid_data() -> None:
    comment = InlineReviewComment(
        path="src/auth.py",
        line=12,
        body="Avoid evaluating untrusted input.",
    )

    assert comment.path == "src/auth.py"
    assert comment.line == 12
    assert comment.body == "Avoid evaluating untrusted input."
    assert comment.side == "RIGHT"


def test_inline_review_comment_defaults_to_right_side() -> None:
    comment = InlineReviewComment(
        path="src/auth.py",
        line=5,
        body="Security issue detected.",
    )

    assert comment.side == "RIGHT"


def test_inline_review_comment_serializes_to_github_payload() -> None:
    comment = InlineReviewComment(
        path="src/auth.py",
        line=12,
        body="Avoid evaluating untrusted input.",
    )

    assert comment.to_github_payload() == {
        "path": "src/auth.py",
        "line": 12,
        "side": "RIGHT",
        "body": "Avoid evaluating untrusted input.",
    }


def test_inline_review_comment_is_immutable() -> None:
    comment = InlineReviewComment(
        path="src/auth.py",
        line=12,
        body="Security issue detected.",
    )

    with pytest.raises(FrozenInstanceError):
        comment.line = 15  # type: ignore[misc]


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "/src/auth.py",
        "../auth.py",
        "src/../auth.py",
        "src\\auth.py",
        "src//auth.py",
        "src/auth.py/",
    ],
)
def test_inline_review_comment_rejects_invalid_paths(
    path: str,
) -> None:
    with pytest.raises(ValueError):
        InlineReviewComment(
            path=path,
            line=12,
            body="Security issue detected.",
        )


@pytest.mark.parametrize(
    "line",
    [0, -1, True, 1.5],
)
def test_inline_review_comment_rejects_invalid_line_numbers(
    line: int,
) -> None:
    with pytest.raises(ValueError):
        InlineReviewComment(
            path="src/auth.py",
            line=line,
            body="Security issue detected.",
        )


@pytest.mark.parametrize("body", ["", "   "])
def test_inline_review_comment_rejects_empty_body(
    body: str,
) -> None:
    with pytest.raises(ValueError):
        InlineReviewComment(
            path="src/auth.py",
            line=12,
            body=body,
        )


def test_inline_review_comment_rejects_unsupported_side() -> None:
    with pytest.raises(ValueError):
        InlineReviewComment(
            path="src/auth.py",
            line=12,
            body="Security issue detected.",
            side="LEFT",  # type: ignore[arg-type]
        )
