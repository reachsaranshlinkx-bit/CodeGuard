from __future__ import annotations

import re

_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?: .*)?$"
)


def parse_added_line_numbers(
    patch: str | None,
) -> frozenset[int]:
    """Return new-file line numbers added by a unified diff patch.

    Removed lines are excluded. Context lines advance the new-file
    line counter. Malformed or incomplete hunks raise ValueError.
    """

    if patch is None or patch == "":
        return frozenset()

    if not isinstance(patch, str):
        raise TypeError("Patch must be a string or None.")

    added_lines: set[int] = set()

    old_remaining = 0
    new_remaining = 0
    new_line = 0
    inside_hunk = False

    for line in patch.splitlines():
        if line.startswith("@@"):
            if inside_hunk and (old_remaining != 0 or new_remaining != 0):
                raise ValueError("Inconsistent hunk line counts.")

            match = _HUNK_HEADER.fullmatch(line)

            if match is None:
                raise ValueError("Malformed hunk header.")

            old_remaining = int(match.group("old_count") or 1)
            new_remaining = int(match.group("new_count") or 1)
            new_line = int(match.group("new_start"))

            inside_hunk = old_remaining > 0 or new_remaining > 0
            continue

        if not inside_hunk:
            continue

        # Git metadata does not represent an added or removed line.
        if line == "\\ No newline at end of file":
            continue

        if not line:
            raise ValueError("Malformed hunk line.")

        prefix = line[0]

        if prefix == "+":
            if new_remaining <= 0:
                raise ValueError("Inconsistent hunk line counts.")

            added_lines.add(new_line)
            new_line += 1
            new_remaining -= 1

        elif prefix == "-":
            if old_remaining <= 0:
                raise ValueError("Inconsistent hunk line counts.")

            old_remaining -= 1

        elif prefix == " ":
            if old_remaining <= 0 or new_remaining <= 0:
                raise ValueError("Inconsistent hunk line counts.")

            old_remaining -= 1
            new_remaining -= 1
            new_line += 1

        else:
            raise ValueError("Malformed hunk line.")

        if old_remaining == 0 and new_remaining == 0:
            inside_hunk = False

    if inside_hunk and (old_remaining != 0 or new_remaining != 0):
        raise ValueError("Inconsistent hunk line counts.")

    return frozenset(added_lines)
