import pytest

from codeguard.review.comment_models import InlineReviewComment
from codeguard.review.finding_models import ReviewReport
from codeguard.review.review_fingerprint import (
    create_review_fingerprint,
    create_review_marker,
)


def make_report(
    *,
    owner: str = "octocat",
    repo: str = "CodeGuard",
    pull_number: int = 42,
    head_sha: str = "a" * 40,
) -> ReviewReport:
    return ReviewReport(
        owner=owner,
        repo=repo,
        pull_number=pull_number,
        title="Improve authentication",
        html_url="https://github.com/octocat/CodeGuard/pull/42",
        head_sha=head_sha,
        findings=(),
        files_analyzed=1,
    )


def make_comments() -> tuple[InlineReviewComment, ...]:
    return (
        InlineReviewComment(
            path="src/auth.py",
            line=12,
            body="Avoid evaluating untrusted input.",
        ),
        InlineReviewComment(
            path="src/database.py",
            line=25,
            body="Use parameterized queries.",
        ),
    )


def test_fingerprint_is_deterministic() -> None:
    report = make_report()
    comments = make_comments()

    first = create_review_fingerprint(report, comments)
    second = create_review_fingerprint(report, comments)

    assert first == second
    assert len(first) == 64


def test_fingerprint_is_independent_of_comment_order() -> None:
    report = make_report()
    comments = make_comments()

    reversed_comments = tuple(reversed(comments))

    assert create_review_fingerprint(
        report,
        comments,
    ) == create_review_fingerprint(
        report,
        reversed_comments,
    )


def test_fingerprint_changes_when_head_sha_changes() -> None:
    comments = make_comments()

    first = create_review_fingerprint(
        make_report(head_sha="a" * 40),
        comments,
    )

    second = create_review_fingerprint(
        make_report(head_sha="b" * 40),
        comments,
    )

    assert first != second


def test_fingerprint_changes_when_repository_changes() -> None:
    comments = make_comments()

    first = create_review_fingerprint(
        make_report(repo="CodeGuard"),
        comments,
    )

    second = create_review_fingerprint(
        make_report(repo="AnotherRepo"),
        comments,
    )

    assert first != second


def test_fingerprint_changes_when_pull_number_changes() -> None:
    comments = make_comments()

    first = create_review_fingerprint(
        make_report(pull_number=42),
        comments,
    )

    second = create_review_fingerprint(
        make_report(pull_number=43),
        comments,
    )

    assert first != second


def test_fingerprint_changes_when_comment_content_changes() -> None:
    report = make_report()

    original = make_comments()

    modified = (
        InlineReviewComment(
            path="src/auth.py",
            line=12,
            body="Use a safe alternative to eval.",
        ),
        original[1],
    )

    assert create_review_fingerprint(
        report,
        original,
    ) != create_review_fingerprint(
        report,
        modified,
    )


def test_review_marker_has_expected_format() -> None:
    fingerprint = "a" * 64

    assert create_review_marker(fingerprint) == (
        f"<!-- codeguard-review:v1:{fingerprint} -->"
    )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "invalid",
        "g" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_review_marker_rejects_invalid_fingerprints(
    fingerprint: str,
) -> None:
    with pytest.raises(ValueError):
        create_review_marker(fingerprint)
