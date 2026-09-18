"""
Studio validate-toc command — validate Table of Contents in Markdown files.

Checks that TOC exists, anchors point to real headings, all headings are
covered, and the TOC is not stale.  Thin CLI wrapper around
``studio.utils.toc.validate_toc``.

@cpt-flow:cpt-studio-flow-traceability-validation-validate:p1
@cpt-dod:cpt-studio-dod-traceability-validation-structure:p1
"""

# @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-imports
import argparse
from pathlib import Path
from typing import List

from ..utils import error_codes as EC
from ..utils.severity import run_verdict
from ..utils.toc import DEFAULT_MAX_SECTION_LINES, add_toc_max_level_argument, validate_toc
from ..utils.ui import ui
# @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-imports

# @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-validate-one
def _validate_one_file(filepath: Path, args: argparse.Namespace) -> dict:
    """Validate a single file, returning its result dict. Never raises --
    a missing file or a read failure (permission denied, binary/non-UTF-8
    content, a TOCTOU race) is reported as an ERROR result instead, so one
    bad file in a batch can't abort validation of the rest.
    """
    if not filepath.is_file():
        return {
            "file": str(filepath),
            "status": "ERROR",
            "message": "File not found",
            "code": EC.FILE_LOAD_ERROR,
        }

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "file": str(filepath),
            "status": "ERROR",
            "message": f"Could not read file: {exc}",
            "code": EC.FILE_READ_ERROR,
        }

    report = validate_toc(
        content,
        artifact_path=filepath,
        max_heading_level=args.max_level,
        max_section_lines=args.max_section_lines,
    )
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    # Per-file status is decided by the same rule as the run's. Leaving it out
    # of `--fail-on-warnings` would report every file as WARN under an overall
    # FAIL, so a reader scanning the list could not find the file that failed.
    file_verdict = run_verdict(
        len(errors),
        len(warnings),
        fail_on_warnings=bool(getattr(args, "fail_on_warnings", False)),
        warn_status="WARN",
    )
    file_result: dict = {
        "file": str(filepath),
        "status": file_verdict.status,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    if file_verdict.failed_on:
        file_result["failed_on"] = file_verdict.failed_on
    if args.verbose or errors:
        file_result["errors"] = errors
    if args.verbose or warnings:
        file_result["warnings"] = warnings
    return file_result
# @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-validate-one


def cmd_validate_toc(argv: List[str]) -> int:
    """Validate Table of Contents in markdown files."""
    # @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-parse-args
    p = argparse.ArgumentParser(
        prog="cfs validate-toc",
        description="Validate Table of Contents in Markdown files",
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Markdown file path(s) to validate",
    )
    add_toc_max_level_argument(p)
    p.add_argument(
        "--max-section-lines",
        type=int,
        default=DEFAULT_MAX_SECTION_LINES,
        help=f"Warn when a section exceeds this many lines (default: {DEFAULT_MAX_SECTION_LINES})",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Include full error details in output",
    )
    p.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Fail the run when there are warnings but no errors (exit 2)",
    )
    args = p.parse_args(argv)
    # @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-parse-args

    # @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-resolve-files
    files_to_validate = [Path(f).resolve() for f in args.files]
    # @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-resolve-files

    # @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-foreach-file
    results = [_validate_one_file(filepath, args) for filepath in files_to_validate]
    total_errors = 0
    total_warnings = 0
    for file_result in results:
        if file_result["status"] == "ERROR":
            total_errors += 1
        else:
            total_errors += file_result["error_count"]
            total_warnings += file_result["warning_count"]
    # @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-foreach-file

    # @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-return
    # The same verdict helper the other two commands use. `validate-toc` keeps
    # its third status: it is the one command whose WARN nothing keys off, and
    # `--fail-on-warnings` reaches it through the same parameter rather than
    # through a second copy of the rule.
    verdict = run_verdict(
        total_errors,
        total_warnings,
        fail_on_warnings=bool(getattr(args, "fail_on_warnings", False)),
        warn_status="WARN",
    )

    output: dict = {
        "status": verdict.status,
        "files_validated": len(results),
        "error_count": total_errors,
        "warning_count": total_warnings,
        "results": results,
    }
    if verdict.failed_on:
        output["failed_on"] = verdict.failed_on

    ui.result(output, human_fn=_human_validate_toc)

    return verdict.exit_code
    # @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-return

# @cpt-begin:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-format
def _human_validate_toc_file(r: dict) -> None:
    """Render one file's line in the human report."""
    path = r.get("file", "?")
    status = r.get("status", "?")
    warns = r.get("warning_count", 0)
    if status == "PASS":
        ui.file_action(path, "unchanged")
        return
    if status == "ERROR":
        ui.error(f"{path}: {r.get('message', 'unknown error')}")
        return
    if status not in ("FAIL", "WARN"):
        ui.substep(f"{path}: {status}")
        return
    if status == "FAIL" and not r.get("failed_on"):
        ui.warn(f"{path}: {r.get('error_count', 0)} error(s), {warns} warning(s)")
    else:
        ui.warn(f"{path}: {warns} warning(s)")
    for e in r.get("errors", []):
        ui.substep(f"  ✗ {e}")
    for w in r.get("warnings", []):
        ui.substep(f"  ⚠ {w}")


def _human_validate_toc(data: dict) -> None:
    ui.header("Validate TOC")
    for r in data.get("results", []):
        _human_validate_toc_file(r)
    overall = data.get("status", "")
    n = data.get("files_validated", 0)
    if overall == "PASS":
        ui.success(f"{n} file(s) validated, all TOCs correct.")
    elif overall == "FAIL" and data.get("failed_on") == "warnings":
        # Without this the reader of a `--fail-on-warnings` run is told
        # "0 error(s) found" by a run that just failed, and is given no reason.
        ui.error(
            f"{n} file(s) validated, {data.get('warning_count', 0)} warning(s) "
            "— failing because --fail-on-warnings is set."
        )
    elif overall == "FAIL":
        ui.error(f"{n} file(s) validated, {data.get('error_count', 0)} error(s) found.")
    else:
        ui.warn(f"{n} file(s) validated ({overall}).")
    ui.blank()
# @cpt-end:cpt-studio-algo-traceability-validation-validate-toc:p1:inst-toc-format
