from pathlib import PurePosixPath

from codeguard.context.source_snapshot import (
    RepositorySourceFile,
    RepositorySourceSnapshot,
)


class AIReviewContextBuilder:
    """Build bounded, deterministic source context for an AI review."""

    _MAX_CONTEXT_CHARACTERS = 50_000

    def build(
        self,
        *,
        changed_file_paths: tuple[str, ...],
        related_file_paths: tuple[str, ...],
        source_snapshot: RepositorySourceSnapshot,
    ) -> str | None:
        """Select requested source files and render them within the context limit."""

        changed_paths = self._python_paths(changed_file_paths)
        changed_path_set = set(changed_paths)
        related_paths = tuple(
            path
            for path in self._python_paths(related_file_paths)
            if path not in changed_path_set
        )

        if not changed_paths and not related_paths:
            return None

        requested_paths = (*changed_paths, *related_paths)
        source_files = {
            source_file.path: source_file for source_file in source_snapshot.files
        }
        unavailable_paths = tuple(
            path for path in requested_paths if path not in source_files
        )
        candidates = tuple(path for path in requested_paths if path in source_files)

        if not candidates:
            return None

        selected_paths: set[str] = set()

        # Revisit paths that did not fit before later files were selected: their
        # omission metadata may have made the earlier candidate too large.
        while True:
            added_path = False

            for path in candidates:
                if path in selected_paths:
                    continue

                trial_paths = selected_paths | {path}
                omitted_paths = tuple(
                    candidate
                    for candidate in candidates
                    if candidate not in trial_paths
                )
                context = self._render(
                    source_snapshot=source_snapshot,
                    source_files=source_files,
                    selected_paths=trial_paths,
                    changed_paths=changed_paths,
                    related_paths=related_paths,
                    unavailable_paths=unavailable_paths,
                    context_omitted_paths=omitted_paths,
                )

                if len(context) <= self._MAX_CONTEXT_CHARACTERS:
                    selected_paths.add(path)
                    added_path = True

            if not added_path:
                break

        if not selected_paths:
            return None

        omitted_paths = tuple(path for path in candidates if path not in selected_paths)
        context = self._render(
            source_snapshot=source_snapshot,
            source_files=source_files,
            selected_paths=selected_paths,
            changed_paths=changed_paths,
            related_paths=related_paths,
            unavailable_paths=unavailable_paths,
            context_omitted_paths=omitted_paths,
        )

        if len(context) > self._MAX_CONTEXT_CHARACTERS:
            raise RuntimeError(
                "AI review context exceeded its configured character limit."
            )

        return context

    @staticmethod
    def _python_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
        """Return unique Python paths in deterministic order."""

        return tuple(
            sorted(
                {path for path in paths if PurePosixPath(path).suffix.lower() == ".py"}
            )
        )

    def _render(
        self,
        *,
        source_snapshot: RepositorySourceSnapshot,
        source_files: dict[str, RepositorySourceFile],
        selected_paths: set[str],
        changed_paths: tuple[str, ...],
        related_paths: tuple[str, ...],
        unavailable_paths: tuple[str, ...],
        context_omitted_paths: tuple[str, ...],
    ) -> str:
        """Render metadata and selected files without altering source text."""

        snapshot_status = "complete" if source_snapshot.is_complete else "incomplete"
        lines = [
            "AI review repository context",
            f"Repository snapshot coverage: {snapshot_status}",
            f"Tree truncated: {source_snapshot.tree_truncated}",
            f"File limit reached: {source_snapshot.file_limit_reached}",
            f"Retrieval failures: {len(source_snapshot.failures)}",
            "Source files omitted from snapshot: "
            f"{len(source_snapshot.omitted_source_paths)}",
            f"Invalid paths skipped: {source_snapshot.skipped_invalid_path_count}",
            f"Oversized files skipped: {source_snapshot.skipped_oversized_file_count}",
            "Files with unknown size skipped: "
            f"{source_snapshot.skipped_unknown_size_count}",
        ]

        if unavailable_paths:
            failure_paths = {
                failure.file_path: failure for failure in source_snapshot.failures
            }
            snapshot_omitted_paths = set(source_snapshot.omitted_source_paths)
            lines.append("Requested files unavailable from the retained snapshot:")

            for path in unavailable_paths:
                if path in failure_paths:
                    reason = "retrieval failed"
                elif path in snapshot_omitted_paths:
                    reason = "source omitted from snapshot"
                else:
                    reason = "not present in retained snapshot"
                lines.append(f"- {path} ({reason})")

        if context_omitted_paths:
            lines.append("Files omitted to fit the context limit:")
            lines.extend(f"- {path}" for path in context_omitted_paths)

        for heading, paths in (
            ("Changed files", changed_paths),
            ("Related caller files", related_paths),
        ):
            included_paths = tuple(path for path in paths if path in selected_paths)

            if not included_paths:
                continue

            lines.append(f"{heading}:")
            for path in included_paths:
                lines.append(f"File: {path}")
                lines.append(source_files[path].source)

        return "\n".join(lines)
