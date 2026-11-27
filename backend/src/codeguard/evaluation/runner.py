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
from codeguard.evaluation.regression import (
    IncompatibleEvaluationReportsError,
    RegressionPolicy,
    UndefinedMetricError,
    compare_evaluation_reports,
    render_regression_json,
)
from codeguard.evaluation.reports import (
    EvaluationReport,
    build_evaluation_report,
    render_json_report,
    render_markdown_report,
)

_INVALID_INPUT_EXIT_CODE = 2
_INCOMPATIBLE_REPORTS_EXIT_CODE = 3
_INCONCLUSIVE_EXIT_CODE = 4
_REPORT_WRITE_EXIT_CODE = 5


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
        _raise_if_invalid_json(error)
        raise
    if not run.is_complete:
        raise IncompleteEvaluationRunError
    return run


def _raise_if_invalid_json(error: ValidationError) -> None:
    if any(detail["type"] == "json_invalid" for detail in error.errors()):
        raise InvalidEvaluationJSONError from None


def _load_evaluation_report(input_path: Path) -> EvaluationReport:
    input_text = input_path.read_text(encoding="utf-8")
    try:
        return EvaluationReport.model_validate_json(input_text)
    except ValidationError as error:
        _raise_if_invalid_json(error)
        raise


def _load_regression_policy(input_path: Path) -> RegressionPolicy:
    input_text = input_path.read_text(encoding="utf-8")
    try:
        return RegressionPolicy.model_validate_json(input_text)
    except ValidationError as error:
        _raise_if_invalid_json(error)
        raise


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
        epilog=(
            "Compare reports with: codeguard-evaluation compare --baseline "
            "baseline.json --candidate candidate.json --policy policy.json"
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


def _report_error(message: str, exit_code: int = 1) -> int:
    sys.stderr.write(f"codeguard-evaluation: {message}\n")
    return exit_code


def _run_report_cli(arguments: Sequence[str]) -> int:
    parsed_arguments = _argument_parser().parse_args(arguments)
    try:
        report = evaluate_run_file(parsed_arguments.input)
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
        if parsed_arguments.format == "json"
        else render_markdown_report(report)
    )
    if parsed_arguments.output is None:
        try:
            sys.stdout.write(rendered_report)
            sys.stdout.flush()
        except OSError:
            return _report_error("Could not write the report to standard output.")
        return 0

    try:
        write_report_atomically(
            parsed_arguments.output,
            parsed_arguments.input,
            rendered_report,
        )
    except OutputInputCollisionError:
        return _report_error("Output destination must not be the input file.")
    except OSError:
        return _report_error(
            "Could not write report; check that the destination is writable."
        )
    return 0


def _comparison_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeguard-evaluation compare",
        description="Compare candidate metrics with an approved baseline report.",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Regression JSON destination; writes to stdout when omitted.",
    )
    return parser


def _run_comparison_cli(arguments: Sequence[str]) -> int:
    parsed_arguments = _comparison_argument_parser().parse_args(arguments)
    try:
        baseline = _load_evaluation_report(parsed_arguments.baseline)
        candidate = _load_evaluation_report(parsed_arguments.candidate)
        policy = _load_regression_policy(parsed_arguments.policy)
        result = compare_evaluation_reports(baseline, candidate, policy)
    except FileNotFoundError:
        return _report_error(
            "Baseline, candidate, or policy input file was not found.",
            _INVALID_INPUT_EXIT_CODE,
        )
    except InvalidEvaluationJSONError:
        return _report_error(
            "Baseline, candidate, or policy input is not valid JSON.",
            _INVALID_INPUT_EXIT_CODE,
        )
    except UnicodeDecodeError:
        return _report_error(
            "Baseline, candidate, and policy files must be UTF-8 encoded.",
            _INVALID_INPUT_EXIT_CODE,
        )
    except IncompatibleEvaluationReportsError as error:
        return _report_error(str(error), _INCOMPATIBLE_REPORTS_EXIT_CODE)
    except UndefinedMetricError as error:
        return _report_error(str(error), _INCONCLUSIVE_EXIT_CODE)
    except ValidationError:
        return _report_error(
            "Baseline, candidate, or policy does not satisfy its contract.",
            _INVALID_INPUT_EXIT_CODE,
        )
    except OSError:
        return _report_error(
            "Could not read the baseline, candidate, or policy file.",
            _INVALID_INPUT_EXIT_CODE,
        )

    rendered_report = render_regression_json(result)
    if parsed_arguments.output is None:
        try:
            sys.stdout.write(rendered_report)
            sys.stdout.flush()
        except OSError:
            return _report_error(
                "Could not write the comparison to standard output.",
                _REPORT_WRITE_EXIT_CODE,
            )
    else:
        try:
            output_path = parsed_arguments.output
            source_paths = (
                parsed_arguments.baseline,
                parsed_arguments.candidate,
                parsed_arguments.policy,
            )
            if any(
                output_path.resolve() == source.resolve() for source in source_paths
            ):
                raise OutputInputCollisionError
            write_report_atomically(
                output_path,
                parsed_arguments.baseline,
                rendered_report,
            )
        except OutputInputCollisionError:
            return _report_error(
                "Output destination must not replace a comparison input.",
                _REPORT_WRITE_EXIT_CODE,
            )
        except OSError:
            return _report_error(
                "Could not write comparison report; check the destination.",
                _REPORT_WRITE_EXIT_CODE,
            )

    if result.overall_outcome == "fail":
        return 1
    if result.overall_outcome == "inconclusive":
        return _INCONCLUSIVE_EXIT_CODE
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run report generation or the opt-in ``compare`` subcommand."""

    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "compare":
        return _run_comparison_cli(arguments[1:])
    return _run_report_cli(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
