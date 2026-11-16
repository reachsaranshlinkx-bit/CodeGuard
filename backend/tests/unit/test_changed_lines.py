import pytest

from codeguard.review.changed_lines import parse_added_line_numbers


def test_identifies_added_line_numbers() -> None:
    patch = """\
diff --git a/example.py b/example.py
index abc123..def456 100644
--- a/example.py
+++ b/example.py
@@ -1,2 +1,3 @@
 def greet():
-    return "hello"
+    return input("name")
+    print("done")
"""

    assert parse_added_line_numbers(patch) == frozenset({2, 3})


def test_context_lines_advance_new_line_numbers() -> None:
    patch = """\
@@ -10,3 +10,4 @@
 def run():
-    old()
+    first()
+    second()
     return 1
"""

    assert parse_added_line_numbers(patch) == frozenset({11, 12})


def test_removed_lines_do_not_count_as_added() -> None:
    patch = """\
@@ -1,4 +1,3 @@
 def process():
-    old_first()
-    old_second()
+    new_process()
     return True
"""

    assert parse_added_line_numbers(patch) == frozenset({2})


def test_handles_multiple_diff_hunks() -> None:
    patch = """\
@@ -1,2 +1,3 @@
 def first():
+    added()
     return None
@@ -20,3 +21,3 @@
 def second():
-    old()
+    new()
     return True
"""

    assert parse_added_line_numbers(patch) == frozenset({2, 22})


def test_handles_new_file_hunk() -> None:
    patch = """\
@@ -0,0 +1,2 @@
+def hello():
+    pass
"""

    assert parse_added_line_numbers(patch) == frozenset({1, 2})


def test_handles_hunk_headers_with_omitted_counts() -> None:
    patch = """\
@@ -1 +1,2 @@
+added_line()
 context_line()
"""

    assert parse_added_line_numbers(patch) == frozenset({1})


def test_returns_empty_set_when_patch_has_only_deletions() -> None:
    patch = """\
@@ -3,2 +3,0 @@
-old_first()
-old_second()
"""

    assert parse_added_line_numbers(patch) == frozenset()


@pytest.mark.parametrize("patch", [None, ""])
def test_returns_empty_set_when_patch_is_unavailable(
    patch: str | None,
) -> None:
    assert parse_added_line_numbers(patch) == frozenset()


def test_returns_empty_set_without_diff_hunks() -> None:
    patch = """\
diff --git a/image.png b/image.png
Binary files differ
"""

    assert parse_added_line_numbers(patch) == frozenset()


def test_rejects_malformed_hunk_header() -> None:
    patch = """\
@@ -invalid,2 +1,3 @@
+new_line()
"""

    with pytest.raises(ValueError, match="hunk"):
        parse_added_line_numbers(patch)


def test_rejects_inconsistent_hunk_line_counts() -> None:
    patch = """\
@@ -1,2 +1,3 @@
 def example():
+    return True
"""

    with pytest.raises(ValueError, match="hunk"):
        parse_added_line_numbers(patch)


def test_ignores_no_newline_marker() -> None:
    patch = """\
@@ -1,1 +1,1 @@
-old_line()
+new_line()
\\ No newline at end of file
"""

    assert parse_added_line_numbers(patch) == frozenset({1})
