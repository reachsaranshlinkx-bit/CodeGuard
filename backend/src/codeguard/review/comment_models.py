from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class InlineReviewComment:
    """An immutable inline security comment for a GitHub PR review."""

    path: str
    line: int
    body: str
    side: Literal["RIGHT"] = "RIGHT"

    def __post_init__(self) -> None:
        """Validate the inline review comment."""

        if (
            not isinstance(self.path, str)
            or not self.path
            or self.path != self.path.strip()
        ):
            raise ValueError("Comment path must not be empty.")

        if (
            self.path.startswith("/")
            or self.path.endswith("/")
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
            or any(ord(char) < 32 for char in self.path)
        ):
            raise ValueError("Comment path must be a valid relative path.")

        if (
            isinstance(self.line, bool)
            or not isinstance(self.line, int)
            or self.line < 1
        ):
            raise ValueError("Comment line must be a positive integer.")

        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("Comment body must not be empty.")

        if self.side != "RIGHT":
            raise ValueError("Only RIGHT-side comments are supported.")

    def to_github_payload(self) -> dict[str, str | int]:
        """Convert the comment into GitHub's inline review payload."""

        return {
            "path": self.path,
            "line": self.line,
            "side": self.side,
            "body": self.body,
        }
