"""Provider-neutral evaluation artifact loading and command-line execution.

An input artifact is the JSON serialization of ``EvaluationRun`` and includes
the exact embedded ``EvaluationDataset`` snapshot used for that run.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from codeguard.evaluation.contracts import EvaluationRun
from codeguard.evaluation.reports import (
    EvaluationReport,
    build_evaluation_report,
    render_json_report,
    render_markdown_report,
)


class InvalidEvaluationJSONError(ValueError):
    """Raised when an artifact cannot be parsed as JSON."""


class IncompleteEvaluationRunError(ValueError):
    """Raised when an artifact is marked incomplete."""


class OutputInputCollisionError(ValueError):
    """Raised when an output destination resolves to the input artifact."""


def serialize_evaluation_run(run: EvaluationRun) -> str:
    """Serialize a run with its exact dataset snapshot and completeness flag."""

    return f"{run.model_dump_json(indent=2)}\n"


def load_evaluation_run(input_path: Path) -> EvaluationRun:
    """Read and validate a serialized run without consulting the packaged data."""

    input_text = input_path.read_text(encoding="utf-8")
    try:
        run = EvaluationRun.model_validate_json(input_text)
    except ValidationError as error:
        if any(detail["type"] == "json_invalid" for detail in error.errors()):
            raise InvalidEvaluationJSONError from None
        raise
    if not run.is_complete:
        raise IncompleteEvaluationRunError
    return run


def evaluate_run_file(input_path: Path) -> EvaluationReport:
    """Load a complete run artifact and calculate its structured report."""

    return build_evaluation_report(load_evaluation_run(input_path))


def write_report_atomically(
    output_path: Path,
    input_path: Path,
    contents: str,
) -> None:
    """Write a report atomically, refusing to replace the input artifact."""

    if output_path.resolve() == input_path.resolve():
        raise OutputInputCollisionError
    if not output_path.parent.is_dir():
        raise FileNotFoundError
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeguard-evaluation",
        description=(
            "Generate a deterministic report from a validated EvaluationRun JSON "
            "artifact."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="EvaluationRun JSON artifact containing its dataset snapshot.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Report format (default: json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report destination; writes to stdout when omitted.",
    )
    return parser


def _report_error(message: str) -> int:
    sys.stderr.write(f"codeguard-evaluation: {message}\n")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a non-zero status for input or output failures."""

    arguments = _argument_parser().parse_args(argv)
    try:
        report = evaluate_run_file(arguments.input)
    except FileNotFoundError:
        return _report_error("Evaluation input file was not found.")
    except InvalidEvaluationJSONError:
        return _report_error("Evaluation input is not valid JSON.")
    except UnicodeDecodeError:
        return _report_error("Evaluation input must be UTF-8 encoded.")
    except IncompleteEvaluationRunError:
        return _report_error(
            "Evaluation run is marked incomplete; include results for every case."
        )
    except ValidationError:
        return _report_error(
            "Evaluation input does not satisfy the EvaluationRun contract."
        )
    except OSError:
        return _report_error("Could not read the evaluation input file.")

    rendered_report = (
        render_json_report(report)
        if arguments.format == "json"
        else render_markdown_report(report)
    )
    if arguments.output is None:
        try:
            sys.stdout.write(rendered_report)
            sys.stdout.flush()
        except OSError:
            return _report_error("Could not write the report to standard output.")
        return 0

    try:
        write_report_atomically(
            arguments.output,
            arguments.input,
            rendered_report,
        )
    except OutputInputCollisionError:
        return _report_error("Output destination must not be the input file.")
    except OSError:
        return _report_error(
            "Could not write report; check that the destination is writable."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
