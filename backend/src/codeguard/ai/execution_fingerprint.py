import hashlib
import json
import re

from pydantic import ValidationError

from codeguard.ai.contracts import AIReviewInput
from codeguard.context.file_discovery import DiscoveredPythonFile
from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
    _validate_repository_path,
)
from codeguard.context.symbols import RepositoryFileFailure

AI_EXECUTION_INPUT_FORMAT_VERSION = "ai-execution-input-v1"
_GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")


def _validate_git_sha(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _GIT_SHA1_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a canonical 40-character lowercase Git SHA-1."
        )


def _validate_snapshot(snapshot: RepositorySourceSnapshot) -> bool:
    _validate_git_sha(snapshot.commit_sha, "source_snapshot.commit_sha")

    if type(snapshot.tree_truncated) is not bool:
        raise ValueError("Snapshot tree_truncated must be a boolean.")

    if type(snapshot.file_limit_reached) is not bool:
        raise ValueError("Snapshot file_limit_reached must be a boolean.")

    counts = (
        snapshot.skipped_invalid_path_count,
        snapshot.skipped_oversized_file_count,
        snapshot.skipped_unknown_size_count,
    )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts
    ):
        raise ValueError("Snapshot skipped-file counts must be non-negative integers.")

    if not isinstance(snapshot.files, tuple) or any(
        not isinstance(source_file, RepositorySourceFile)
        or not isinstance(source_file.discovered_file, DiscoveredPythonFile)
        or not isinstance(source_file.source, str)
        for source_file in snapshot.files
    ):
        raise ValueError("Snapshot files must contain valid repository source files.")

    if not isinstance(snapshot.failures, tuple) or any(
        not isinstance(failure, RepositoryFileFailure) or failure.stage != "retrieval"
        for failure in snapshot.failures
    ):
        raise ValueError("Snapshot failures must contain valid retrieval failures.")

    if not isinstance(snapshot.omitted_source_paths, tuple):
        raise ValueError("Snapshot omitted source paths must be a tuple.")

    file_paths = tuple(source_file.path for source_file in snapshot.files)
    failure_paths = tuple(failure.file_path for failure in snapshot.failures)
    omitted_paths = snapshot.omitted_source_paths

    for path in (*file_paths, *failure_paths, *omitted_paths):
        if not isinstance(path, str):
            raise ValueError("Snapshot repository paths must be strings.")

        _validate_repository_path(path)

    if len(file_paths) != len(set(file_paths)):
        raise ValueError("Snapshot contains duplicate retained source paths.")

    if len(failure_paths) != len(set(failure_paths)):
        raise ValueError("Snapshot contains duplicate retrieval failure paths.")

    if len(omitted_paths) != len(set(omitted_paths)):
        raise ValueError("Snapshot contains duplicate omitted source paths.")

    if (
        set(file_paths) & set(failure_paths)
        or set(file_paths) & set(omitted_paths)
        or set(failure_paths) & set(omitted_paths)
    ):
        raise ValueError("Snapshot source paths must have distinct outcomes.")

    is_complete = snapshot.is_complete
    if type(is_complete) is not bool:
        raise ValueError("Snapshot is_complete must be a boolean.")

    return is_complete


def build_ai_execution_input_fingerprint(
    base_sha: str,
    review_input: AIReviewInput,
    source_snapshot: RepositorySourceSnapshot,
) -> str:
    """Build a versioned digest of the exact AI review and evidence inputs."""

    _validate_git_sha(base_sha, "base_sha")

    if not isinstance(review_input, AIReviewInput):
        raise TypeError("review_input must be an AIReviewInput instance.")

    try:
        original_head_sha = review_input.head_sha
        _validate_git_sha(original_head_sha, "review_input.head_sha")
        validated_review_input = AIReviewInput.model_validate(
            {
                "owner": review_input.owner,
                "repo": review_input.repo,
                "pull_number": review_input.pull_number,
                "head_sha": review_input.head_sha,
                "context": review_input.context,
            },
            strict=True,
        )
    except (AttributeError, ValidationError):
        raise ValueError("review_input must contain valid AI review inputs.") from None

    if validated_review_input.head_sha != original_head_sha:
        raise ValueError("review_input.head_sha must not be normalized.")

    if not isinstance(source_snapshot, RepositorySourceSnapshot):
        raise TypeError("source_snapshot must be a RepositorySourceSnapshot instance.")

    is_complete = _validate_snapshot(source_snapshot)

    if source_snapshot.commit_sha != validated_review_input.head_sha:
        raise ValueError("source_snapshot.commit_sha must match review_input.head_sha.")

    payload = {
        "format": AI_EXECUTION_INPUT_FORMAT_VERSION,
        "base_sha": base_sha,
        "review_input": {
            "owner": validated_review_input.owner,
            "repo": validated_review_input.repo,
            "pull_number": validated_review_input.pull_number,
            "head_sha": validated_review_input.head_sha,
            "context": validated_review_input.context,
        },
        "source_snapshot": {
            "commit_sha": source_snapshot.commit_sha,
            "files": [
                {"path": source_file.path, "source": source_file.source}
                for source_file in sorted(
                    source_snapshot.files,
                    key=lambda item: item.path,
                )
            ],
            "failures": [
                {"file_path": failure.file_path}
                for failure in sorted(
                    source_snapshot.failures,
                    key=lambda item: item.file_path,
                )
            ],
            "omitted_source_paths": sorted(source_snapshot.omitted_source_paths),
            "is_complete": is_complete,
            "tree_truncated": source_snapshot.tree_truncated,
            "file_limit_reached": source_snapshot.file_limit_reached,
            "skipped_invalid_path_count": (source_snapshot.skipped_invalid_path_count),
            "skipped_oversized_file_count": (
                source_snapshot.skipped_oversized_file_count
            ),
            "skipped_unknown_size_count": (source_snapshot.skipped_unknown_size_count),
        },
    }

    try:
        canonical_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        canonical_bytes = canonical_json.encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("Fingerprint inputs cannot be serialized safely.") from None

    digest = hashlib.sha256(canonical_bytes).hexdigest()

    return f"{AI_EXECUTION_INPUT_FORMAT_VERSION}:{digest}"
