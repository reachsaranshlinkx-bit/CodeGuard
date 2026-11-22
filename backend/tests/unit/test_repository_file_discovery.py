import pytest

from codeguard.context.file_discovery import (
    RepositoryFileDiscovery,
)
from codeguard.github.schemas import GitHubRepositoryTree


def make_tree(
    entries: list[dict[str, object]],
    *,
    truncated: bool = False,
) -> GitHubRepositoryTree:
    return GitHubRepositoryTree.model_validate(
        {
            "sha": "a" * 40,
            "truncated": truncated,
            "tree": entries,
        }
    )


def make_blob(
    path: str,
    *,
    size: int | None = 100,
    sha: str = "b" * 40,
    mode: str = "100644",
) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path,
        "mode": mode,
        "type": "blob",
        "sha": sha,
    }

    if size is not None:
        entry["size"] = size

    return entry


def test_discovery_excludes_symlinks_and_includes_executable_files() -> None:
    tree = make_tree(
        [
            make_blob("src/regular.py"),
            make_blob("src/executable.py", mode="100755"),
            make_blob("src/linked.py", mode="120000"),
        ]
    )

    result = RepositoryFileDiscovery().discover(tree)

    assert [file.path for file in result.files] == [
        "src/executable.py",
        "src/regular.py",
    ]


def test_discovery_selects_python_files_only() -> None:
    tree = make_tree(
        [
            make_blob("src/main.py"),
            make_blob("src/config.json"),
            make_blob("README.md"),
            {
                "path": "src",
                "mode": "040000",
                "type": "tree",
                "sha": "c" * 40,
            },
            {
                "path": "vendor",
                "mode": "160000",
                "type": "commit",
                "sha": "d" * 40,
            },
        ]
    )

    result = RepositoryFileDiscovery().discover(tree)

    assert [file.path for file in result.files] == ["src/main.py"]
    assert result.is_complete is True


def test_discovery_excludes_generated_directories() -> None:
    tree = make_tree(
        [
            make_blob("src/main.py"),
            make_blob(".venv/lib/package.py"),
            make_blob("__pycache__/module.py"),
            make_blob("node_modules/package/index.py"),
            make_blob("tests/test_main.py"),
        ]
    )

    result = RepositoryFileDiscovery().discover(tree)

    assert [file.path for file in result.files] == [
        "src/main.py",
        "tests/test_main.py",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/absolute.py",
        "src/../outside.py",
        "src//example.py",
        "src\\example.py",
    ],
)
def test_discovery_rejects_unsafe_paths(path: str) -> None:
    tree = make_tree([make_blob(path)])

    result = RepositoryFileDiscovery().discover(tree)

    assert result.files == ()
    assert result.skipped_invalid_path_count == 1


def test_discovery_skips_oversized_files() -> None:
    tree = make_tree(
        [
            make_blob("src/small.py", size=100),
            make_blob("src/large.py", size=2_000),
        ]
    )

    result = RepositoryFileDiscovery(
        max_file_size_bytes=1_000,
    ).discover(tree)

    assert [file.path for file in result.files] == ["src/small.py"]
    assert result.skipped_oversized_file_count == 1


def test_discovery_skips_files_with_unknown_size() -> None:
    tree = make_tree(
        [
            make_blob("src/example.py", size=None),
        ]
    )

    result = RepositoryFileDiscovery().discover(tree)

    assert result.files == ()
    assert result.skipped_unknown_size_count == 1


def test_discovery_respects_maximum_file_count() -> None:
    tree = make_tree(
        [
            make_blob("src/c.py"),
            make_blob("src/a.py"),
            make_blob("src/b.py"),
        ]
    )

    result = RepositoryFileDiscovery(
        max_files=2,
    ).discover(tree)

    assert [file.path for file in result.files] == [
        "src/a.py",
        "src/b.py",
    ]
    assert result.file_limit_reached is True
    assert result.is_complete is False


def test_discovery_preserves_truncated_tree_status() -> None:
    tree = make_tree(
        [make_blob("src/main.py")],
        truncated=True,
    )

    result = RepositoryFileDiscovery().discover(tree)

    assert len(result.files) == 1
    assert result.tree_truncated is True
    assert result.is_complete is False


@pytest.mark.parametrize(
    ("max_files", "max_file_size_bytes"),
    [
        (0, 100),
        (-1, 100),
        (True, 100),
        (100, 0),
        (100, -1),
        (100, True),
    ],
)
def test_discovery_rejects_invalid_limits(
    max_files: int,
    max_file_size_bytes: int,
) -> None:
    with pytest.raises(ValueError):
        RepositoryFileDiscovery(
            max_files=max_files,
            max_file_size_bytes=max_file_size_bytes,
        )
