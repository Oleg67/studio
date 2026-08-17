"""Enforcement reachability: an empty scan must not read as compliance.

Each test names the repository state it builds and the verdict that state
produces. The rule this corpus argues for is that "cannot assess" is not
"PASS", and that the exit code should follow whether the caller asked for a
guarantee:

* a guarantee was requested (thresholds passed, or a checked ``to_code="true"``
  ID exists) and it cannot be assessed -> non-zero exit;
* only a report was requested -> exit 0, but the output says plainly that
  nothing was measured.

Most tests here pin behaviour that is already correct, so they are ordinary
regression cover. The three marked ``xfail(strict=True)`` state the verdict this
corpus proposes for cases the commands currently answer differently; the marker
records that today's behaviour differs, and its reason names the code that
decides. Because the marker is strict, an unexpected pass fails the suite -- so
if a case is settled the other way, or the behaviour changes, the corpus says so
instead of going quietly green.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "studio" / "scripts"))

from studio.commands.spec_coverage import cmd_spec_coverage
from studio.utils.artifacts_meta import ArtifactsMeta, CodebaseEntry, Kit, SystemNode
from studio.utils.codebase import CodeFile, cross_validate_code

TO_CODE_IDS = {
    "cpt-flags-flow-evaluate",
    "cpt-flags-algo-core-bucket",
    "cpt-flags-dod-core-verification",
}


def _context(project_root: Path, codebase: list[CodebaseEntry] | None = None) -> MagicMock:
    """Build a context whose single system registers ``codebase``."""
    meta = ArtifactsMeta(
        version=1,
        project_root=".",
        kits={"test": Kit("test", "CFS", "kits/test")},
        systems=[
            SystemNode(
                name="sys1",
                slug="sys1",
                kit="test",
                artifacts=[],
                codebase=codebase or [],
                children=[],
            ),
        ],
    )
    ctx = MagicMock()
    ctx.meta = meta
    ctx.project_root = project_root
    return ctx


def _run_spec_coverage(ctx: MagicMock, argv: list[str] | None = None) -> tuple[int, dict]:
    """Run spec-coverage against ``ctx`` and return (exit code, parsed report)."""
    with patch("studio.utils.context.get_context", return_value=ctx):
        with patch("sys.stdout", new_callable=StringIO) as out:
            code = cmd_spec_coverage(argv or [])
    return code, json.loads(out.getvalue())


# ---------------------------------------------------------------------------
# The enforcement already exists -- it is only unreachable.
#
# These pass today. They are the evidence that closing the gap is a guard
# change and not new validation logic, so they must keep passing afterwards.
# ---------------------------------------------------------------------------

class TestEnforcementExistsOnEmptyInput:
    def test_every_unmet_to_code_id_is_reported_when_no_code_files_exist(self):
        result = cross_validate_code([], set(TO_CODE_IDS), set(TO_CODE_IDS), traceability="FULL")

        coverage_errors = [e for e in result["errors"] if e["type"] == "coverage"]
        assert len(coverage_errors) == len(TO_CODE_IDS)
        assert {e["id"] for e in coverage_errors} == TO_CODE_IDS

    def test_nothing_claimed_means_nothing_owed(self):
        """No checked to_code IDs and no code is a legitimately clean state.

        This is the leniency the design deliberately keeps: a spec-first
        repository with a PRD and an ADR but no implementation yet must not be
        failed. It is also the assertion most at risk from an over-broad fix.
        """
        result = cross_validate_code([], set(), set(), traceability="FULL")

        assert result["errors"] == []
        assert result["warnings"] == []

    def test_docs_only_traceability_is_unaffected(self, tmp_path: Path):
        result = cross_validate_code([], set(), set(), traceability="DOCS-ONLY")

        assert result["errors"] == []


class TestKnownGoodStaysGreen:
    """Regression pins. A fix that breaks one of these is too broad."""

    def test_a_marked_implementation_reports_no_coverage_error(self, tmp_path: Path):
        source = tmp_path / "evaluator.py"
        source.write_text(
            "# @cpt-algo:cpt-flags-algo-core-bucket:p1\ndef bucket():\n    return 0\n",
            encoding="utf-8",
        )
        code_file, _ = CodeFile.from_path(source)

        result = cross_validate_code(
            [code_file],
            {"cpt-flags-algo-core-bucket"},
            {"cpt-flags-algo-core-bucket"},
            traceability="FULL",
        )

        assert [e for e in result["errors"] if e["type"] == "coverage"] == []

    def test_an_unmarked_file_still_fails(self, tmp_path: Path):
        """Already-correct behaviour: one scanned file makes enforcement fire."""
        source = tmp_path / "store.py"
        source.write_text("def put():\n    return None\n", encoding="utf-8")
        code_file, _ = CodeFile.from_path(source)

        result = cross_validate_code(
            [code_file],
            {"cpt-flags-algo-core-bucket"},
            {"cpt-flags-algo-core-bucket"},
            traceability="FULL",
        )

        coverage_errors = [e for e in result["errors"] if e["type"] == "coverage"]
        assert len(coverage_errors) == 1
        assert coverage_errors[0]["id"] == "cpt-flags-algo-core-bucket"

    def test_populated_codebase_reports_normally(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "src"
            src.mkdir()
            (src / "main.py").write_text(
                "# @cpt-algo:cpt-flags-algo-core-bucket:p1\nx = 1\n", encoding="utf-8"
            )
            ctx = _context(root, [CodebaseEntry(path="src", extensions=[".py"])])

            code, report = _run_spec_coverage(ctx)

        assert code == 0
        assert report["summary"]["total_files"] > 0

    def test_an_unassessable_report_without_thresholds_still_exits_zero(self):
        """No guarantee was requested, so the exit code must not move.

        Only the *status* should change (see the xfail below). Keeping this exit
        code is what lets the fix land without failing spec-first projects.
        """
        with TemporaryDirectory() as directory:
            ctx = _context(Path(directory), codebase=[])

            code, _ = _run_spec_coverage(ctx)

        assert code == 0


class TestVerdictIsDeterministic:
    def test_repeated_runs_agree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "src"
            src.mkdir()
            (src / "main.py").write_text(
                "# @cpt-algo:cpt-flags-algo-core-bucket:p1\nx = 1\n", encoding="utf-8"
            )
            verdicts = set()
            for _ in range(5):
                ctx = _context(root, [CodebaseEntry(path="src", extensions=[".py"])])
                code, report = _run_spec_coverage(ctx)
                verdicts.add((code, report["status"], report["summary"]["total_files"]))

        assert len(verdicts) == 1


# ---------------------------------------------------------------------------
# Known-bad must go red. These fail today; that is the defect.
# ---------------------------------------------------------------------------

class TestUnassessableIsNotCompliant:
    @pytest.mark.xfail(
        strict=True,
        reason="_empty_coverage_result() hard-codes status PASS when nothing was scanned",
    )
    def test_an_empty_scan_is_not_reported_as_pass(self):
        with TemporaryDirectory() as directory:
            ctx = _context(Path(directory), codebase=[])

            _, report = _run_spec_coverage(ctx)

        assert report["status"] != "PASS"

    @pytest.mark.xfail(
        strict=True,
        reason="thresholds are not evaluated at all when the scan is empty, so the run exits 0",
    )
    def test_requested_thresholds_cannot_be_satisfied_by_an_empty_scan(self):
        """0.0% against a required 90 is the maximum possible miss, not a pass."""
        with TemporaryDirectory() as directory:
            ctx = _context(Path(directory), codebase=[])

            code, _ = _run_spec_coverage(ctx, ["--min-coverage", "90"])

        assert code != 0

    @pytest.mark.xfail(
        strict=True,
        reason="both states share _empty_coverage_result(), so the reports are identical",
    )
    def test_unregistered_and_resolved_to_nothing_are_distinguishable(self):
        """Two different setup mistakes must not produce the same report.

        No registered entry is a missing-configuration problem; an entry that
        resolves to zero files is a wrong-path problem. A report that cannot
        tell them apart cannot state its own denominator.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, unregistered = _run_spec_coverage(_context(root, codebase=[]))
            _, resolved_to_nothing = _run_spec_coverage(
                _context(root, [CodebaseEntry(path="does/not/exist", extensions=[".py"])])
            )

        assert unregistered != resolved_to_nothing
