"""
Studio TOC Command — Generate Table of Contents for Markdown files.

Thin CLI wrapper around the unified ``studio.utils.toc`` module.

@cpt-flow:cpt-studio-flow-developer-experience-toc:p1
"""

# @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-imports
import argparse
from pathlib import Path
from typing import TYPE_CHECKING, List

from studio.utils.toc import (
    add_toc_max_level_argument,
    process_file as _process_file,
    validate_toc as _validate_toc,
)
from ..utils.ui import ui

if TYPE_CHECKING:  # pragma: no cover - for the annotation only
    # Imported lazily inside `cmd_toc` at runtime: `cfs toc` should not pull
    # the whole validate command tree in just to resolve a heading depth.
    from .validate_toc import TocResolution
# @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-imports


def _process_toc_file(
    filepath_str: str,
    *,
    max_level: int,
    dry_run: bool,
    indent_size: int,
) -> dict:
    # @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-process
    filepath = Path(filepath_str).resolve()
    return _process_file(
        filepath,
        max_level=max_level,
        dry_run=dry_run,
        indent_size=indent_size,
    )
    # @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-process

# @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-validate
def _post_generation_validation(filepath: Path, resolution: "TocResolution") -> dict:
    """Check what was just written — unless nothing validates this kind's TOC.

    A kind declaring `toc = false` is skipped by `cfs validate` and reported
    as not applicable by `cfs validate-toc`. Writing the table is still the
    right answer to an explicit request — that switch says a table is not
    required, and has no way to say one is forbidden — but this command should
    not be the only one in the toolchain that judges it.
    """
    if not resolution.checked:
        return {
            "status": "SKIPPED",
            "reason": (
                f"{resolution.kind} declares toc = false; "
                "no command validates a table of contents for this kind"
            ),
        }
    report = _validate_toc(
        filepath.read_text(encoding="utf-8"),
        artifact_path=filepath,
        max_heading_level=resolution.max_level,
    )
    errs = report.get("errors", [])
    warns = report.get("warnings", [])
    if not errs and not warns:
        return {"status": "PASS"}
    return {
        "status": "FAIL" if errs else "WARN",
        "errors": len(errs),
        "warnings": len(warns),
        "details": errs + warns,
    }
# @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-validate


def cmd_toc(argv: List[str]) -> int:
    """Generate/update Table of Contents in markdown files."""
    # @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-parse-args
    p = argparse.ArgumentParser(
        prog="cfs toc",
        description="Generate or update Table of Contents in Markdown files",
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Markdown file path(s) to process",
    )
    add_toc_max_level_argument(p, default=None)
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent spaces per nesting level (default: 2)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    p.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip post-generation validation",
    )
    args = p.parse_args(argv)
    # @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-parse-args

    # @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-kind-depth
    # Generate to the depth the checks will judge the result at. Regenerating
    # at this command's own default in a project where a kind configures a
    # shallower one produces a TOC listing headings that `validate-toc` then
    # reports as anchors to nothing — the documented way to fix a stale TOC
    # would hand back a file that fails validation.
    from .validate_toc import resolve_toc, resolve_toc_targets

    toc_targets = resolve_toc_targets()
    # @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-kind-depth

    results = []
    # @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-foreach-file
    validation_errors = 0
    for filepath_str in args.files:
        filepath = Path(filepath_str).resolve()
        resolution = resolve_toc(toc_targets, filepath, args.max_level)
        result = _process_toc_file(
            filepath_str,
            max_level=resolution.max_level,
            dry_run=args.dry_run,
            indent_size=args.indent,
        )

        # Auto-validate after generation (unless skipped or dry-run)
        if (not args.skip_validate
                and not args.dry_run
                and filepath.is_file()
                and result.get("status") not in ("ERROR", "SKIP")):
            validation = _post_generation_validation(filepath, resolution)
            result["validation"] = validation
            validation_errors += int(validation.get("errors") or 0)

        results.append(result)
    # @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-foreach-file

    # @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-return
    output = {
        "status": "OK",
        "files_processed": len(results),
        "results": results,
    }

    if validation_errors:
        output["status"] = "VALIDATION_FAIL"
    elif any(r["status"] == "ERROR" for r in results):
        output["status"] = "PARTIAL" if len(results) > 1 else "ERROR"

    ui.result(output, human_fn=_human_toc)

    if validation_errors:
        return 2
    # @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-return
    return 1 if output["status"] == "ERROR" else 0

# @cpt-begin:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-format
def _human_toc(data: dict) -> None:
    ui.header("Table of Contents")
    for r in data.get("results", []):
        path = r.get("file", "?")
        status = r.get("status", "?")
        if status == "UPDATED":
            ui.file_action(path, "updated")
        elif status == "CREATED":
            ui.file_action(path, "created")
        elif status == "UNCHANGED":
            ui.file_action(path, "unchanged")
        elif status == "ERROR":
            ui.warn(f"{path}: {r.get('message', 'error')}")
        else:
            ui.substep(f"{path}: {status}")
        val = r.get("validation", {})
        if val.get("status") == "SKIPPED":
            ui.substep(f"  (not validated: {val.get('reason', 'not applicable')})")
        if val.get("status") == "FAIL":
            for detail in val.get("details", []):
                ui.warn(f"  {detail}")
    n = data.get("files_processed", 0)
    overall = data.get("status", "")
    if overall in ("OK", "PASS"):
        ui.success(f"{n} file(s) processed.")
    elif overall == "VALIDATION_FAIL":
        ui.error(f"{n} file(s) processed, validation errors found.")
    else:
        ui.warn(f"{n} file(s) processed ({overall}).")
    ui.blank()
# @cpt-end:cpt-studio-flow-developer-experience-toc:p1:inst-toc-gen-format
