"""
Tests for commands/where_defined.py and commands/where_used.py.

Covers: cmd_where_defined, cmd_where_used, _human_where_defined, _human_where_used.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "studio" / "scripts"))

from studio.commands.get_content import cmd_get_content
from studio.commands.list_ids import cmd_list_ids
from studio.commands.where_defined import cmd_where_defined, _human_where_defined
from studio.commands.where_used import cmd_where_used, _human_where_used
from studio.utils.context import StudioContext as CypilotContext, set_context
from studio.utils.ui import set_json_mode
from studio.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_project(root: Path) -> Path:
    """Bootstrap a minimal Constructor Studio project. Returns adapter dir."""
    (root / ".git").mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text(
        '<!-- @cf:root-agents -->\n```toml\ncf-studio-path = "adapter"\n```\n',
        encoding="utf-8",
    )
    adapter = root / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "config").mkdir(exist_ok=True)
    (adapter / "config" / "AGENTS.md").write_text("# Adapter\n", encoding="utf-8")

    from studio.utils import toml_utils
    toml_utils.dump({
        "version": "1.0",
        "project_root": "..",
        "kits": {"cypilot": {"format": "CFS", "path": "kits/sdlc"}},
        "systems": [{
            "name": "Test",
            "kits": "cypilot",
            "artifacts": [{"path": "architecture/PRD.md", "kind": "PRD"}],
        }],
    }, adapter / "config" / "artifacts.toml")

    # Create the artifact with a defined + referenced ID
    art_dir = root / "architecture"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "PRD.md").write_text(
        "- [x] `p1` - **ID**: `cpt-test-item-1`\n"
        "<!-- @cpt-ref: cpt-test-item-1 -->\n",
        encoding="utf-8",
    )

    # Kit structure
    kit_dir = root / "kits" / "sdlc" / "artifacts" / "PRD"
    kit_dir.mkdir(parents=True, exist_ok=True)
    (kit_dir / "template.md").write_text(
        "---\ncypilot-template:\n  version:\n    major: 1\n    minor: 0\n  kind: PRD\n---\n"
        "- [ ] `p1` - **ID**: `cpt-{system}-item-{slug}`\n",
        encoding="utf-8",
    )
    from _test_helpers import write_constraints_toml
    write_constraints_toml(root / "kits" / "sdlc", {
        "PRD": {"identifiers": {"item": {"template": "cpt-{system}-item-{slug}"}}},
    })
    return adapter


def _with_context(root: Path):
    """Load CypilotContext from project root and set as global."""
    ctx = CypilotContext.load(root)
    set_context(ctx)
    return ctx


class _ContextTestBase(unittest.TestCase):
    """Base that isolates global context around each test."""

    def setUp(self):
        set_context(None)

    def tearDown(self):
        set_context(None)


# =========================================================================
# cmd_where_defined
# =========================================================================

class TestCmdWhereDefined(_ContextTestBase):

    def test_empty_id_returns_error(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd_where_defined(["--id", ""])
        self.assertEqual(rc, 1)
        out = json.loads(stdout.getvalue())
        self.assertEqual(out["status"], "ERROR")

    def test_no_id_returns_error(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd_where_defined([])
        self.assertEqual(rc, 1)

    def test_both_positional_and_flag_warns(self):
        """When both positional and --id are given, positional wins."""
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with self.assertLogs("studio.utils.context", level="WARNING") as logs:
                with redirect_stdout(stdout):
                    cmd_where_defined(["cpt-test-item-1", "--id", "cpt-other"])
            self.assertTrue(any("using positional" in message for message in logs.output))

    def test_artifact_not_found(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd_where_defined(["--id", "test", "--artifact", "/nonexistent/file.md"])
        self.assertEqual(rc, 1)
        out = json.loads(stdout.getvalue())
        self.assertEqual(out["status"], "ERROR")

    def test_artifact_no_context(self):
        with TemporaryDirectory() as td:
            art = Path(td) / "art.md"
            art.write_text("test\n", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_defined(["--id", "test", "--artifact", str(art)])
            self.assertEqual(rc, 1)

    def test_artifact_not_in_registry(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            unregistered = root / "random.md"
            unregistered.write_text("content\n", encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_defined(["--id", "test", "--artifact", str(unregistered)])
                self.assertEqual(rc, 1)
                out = json.loads(stdout.getvalue())
                self.assertIn("not in Constructor Studio registry", out.get("message", ""))
            finally:
                os.chdir(cwd)

    def test_artifact_outside_project(self):
        """Artifact exists but is outside project root."""
        with TemporaryDirectory() as td1, TemporaryDirectory() as td2:
            root = Path(td1)
            _setup_project(root)
            outside = Path(td2) / "outside.md"
            outside.write_text("content\n", encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_defined(["--id", "test", "--artifact", str(outside)])
                self.assertEqual(rc, 1)
            finally:
                os.chdir(cwd)

    def test_no_context_returns_error(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_defined(["--id", "test"])
                self.assertEqual(rc, 1)
                out = json.loads(stdout.getvalue())
                self.assertEqual(out["status"], "ERROR")
            finally:
                os.chdir(cwd)

    def test_found_single_definition(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_defined(["cpt-test-item-1"])
            self.assertEqual(rc, 0)
            out = json.loads(stdout.getvalue())
            self.assertEqual(out["status"], "FOUND")
            self.assertEqual(out["count"], 1)

    def test_not_found_returns_2(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_defined(["cpt-nonexistent-id"])
            self.assertEqual(rc, 2)
            out = json.loads(stdout.getvalue())
            self.assertEqual(out["status"], "NOT_FOUND")

    def test_ambiguous_multiple_definitions(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            adapter = _setup_project(root)
            from studio.utils import toml_utils
            toml_utils.dump({
                "version": "1.0",
                "project_root": "..",
                "kits": {"cypilot": {"format": "CFS", "path": "kits/sdlc"}},
                "systems": [{
                    "name": "Test",
                    "kits": "cypilot",
                    "artifacts": [
                        {"path": "architecture/PRD.md", "kind": "PRD"},
                        {"path": "architecture/DESIGN.md", "kind": "DESIGN"},
                    ],
                }],
            }, adapter / "config" / "artifacts.toml")
            (root / "architecture" / "DESIGN.md").write_text(
                "- [x] `p1` - **ID**: `cpt-test-item-1`\n",
                encoding="utf-8",
            )
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_defined(["cpt-test-item-1"])
            self.assertEqual(rc, 2)
            out = json.loads(stdout.getvalue())
            self.assertEqual(out["status"], "AMBIGUOUS")
            self.assertEqual(out["count"], 2)

    def test_with_artifact_flag_found(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            art_path = root / "architecture" / "PRD.md"
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_defined(["--id", "cpt-test-item-1", "--artifact", str(art_path)])
                self.assertEqual(rc, 0)
                out = json.loads(stdout.getvalue())
                self.assertEqual(out["status"], "FOUND")
            finally:
                os.chdir(cwd)

    def test_no_artifacts_to_scan(self):
        """All registered artifacts are missing from disk."""
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            (root / "architecture" / "PRD.md").unlink()
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_defined(["cpt-test-item-1"])
            self.assertEqual(rc, 0)
            out = json.loads(stdout.getvalue())
            self.assertEqual(out["status"], "NO_ARTIFACTS")
            self.assertEqual(out["artifacts_scanned"], 0)


# =========================================================================
# cmd_where_used
# =========================================================================

class TestCmdWhereUsed(_ContextTestBase):

    def test_empty_id_returns_error(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd_where_used(["--id", ""])
        self.assertEqual(rc, 1)
        out = json.loads(stdout.getvalue())
        self.assertEqual(out["status"], "ERROR")

    def test_no_id_returns_error(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd_where_used([])
        self.assertEqual(rc, 1)

    def test_both_positional_and_flag_warns(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with self.assertLogs("studio.utils.context", level="WARNING") as logs:
                with redirect_stdout(stdout):
                    cmd_where_used(["cpt-test-item-1", "--id", "cpt-other"])
            self.assertTrue(any("using positional" in message for message in logs.output))

    def test_artifact_not_found(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd_where_used(["--id", "test", "--artifact", "/nonexistent/file.md"])
        self.assertEqual(rc, 1)

    def test_artifact_no_context(self):
        with TemporaryDirectory() as td:
            art = Path(td) / "art.md"
            art.write_text("test\n", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_used(["--id", "test", "--artifact", str(art)])
            self.assertEqual(rc, 1)

    def test_artifact_not_in_registry(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            unregistered = root / "random.md"
            unregistered.write_text("content\n", encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_used(["--id", "test", "--artifact", str(unregistered)])
                self.assertEqual(rc, 1)
            finally:
                os.chdir(cwd)

    def test_no_context_returns_error(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_used(["--id", "test"])
                self.assertEqual(rc, 1)
            finally:
                os.chdir(cwd)

    def test_found_references(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_used(["cpt-test-item-1"])
            self.assertEqual(rc, 0)
            out = json.loads(stdout.getvalue())
            self.assertGreaterEqual(out["count"], 0)

    def test_include_definitions(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_used(["cpt-test-item-1", "--include-definitions"])
            self.assertEqual(rc, 0)
            out = json.loads(stdout.getvalue())
            self.assertGreaterEqual(out["count"], 1)

    def test_no_references_found(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_used(["cpt-nonexistent-id"])
            self.assertEqual(rc, 0)
            out = json.loads(stdout.getvalue())
            self.assertEqual(out["count"], 0)

    def test_with_artifact_flag(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            art_path = root / "architecture" / "PRD.md"
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_used(["--id", "cpt-test-item-1", "--artifact", str(art_path), "--include-definitions"])
                self.assertEqual(rc, 0)
            finally:
                os.chdir(cwd)

    def test_no_artifacts_to_scan(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            (root / "architecture" / "PRD.md").unlink()
            _with_context(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = cmd_where_used(["cpt-test-item-1"])
            self.assertEqual(rc, 0)
            out = json.loads(stdout.getvalue())
            self.assertEqual(out["artifacts_scanned"], 0)

    def test_artifact_outside_project(self):
        with TemporaryDirectory() as td1, TemporaryDirectory() as td2:
            root = Path(td1)
            _setup_project(root)
            outside = Path(td2) / "outside.md"
            outside.write_text("content\n", encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(str(root))
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    rc = cmd_where_used(["--id", "test", "--artifact", str(outside)])
                self.assertEqual(rc, 1)
            finally:
                os.chdir(cwd)


# =========================================================================
# Human formatters (need human mode)
# =========================================================================

class _HumanModeBase(unittest.TestCase):
    def setUp(self):
        set_json_mode(False)

    def tearDown(self):
        set_json_mode(True)


class TestHumanWhereDefined(_HumanModeBase):

    def test_found_with_checked(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_defined({
                "status": "FOUND", "id": "cpt-x", "artifacts_scanned": 1, "count": 1,
                "definitions": [{"artifact": "/tmp/A.md", "artifact_type": "PRD", "line": 10, "checked": True}],
            })
        out = buf.getvalue()
        self.assertIn("cpt-x", out)
        self.assertIn("10", out)

    def test_not_found(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_defined({
                "status": "NOT_FOUND", "id": "cpt-missing", "artifacts_scanned": 2,
                "count": 0, "definitions": [],
            })
        out = buf.getvalue()
        self.assertIn("not found", out.lower())

    def test_ambiguous(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_defined({
                "status": "AMBIGUOUS", "id": "cpt-dup", "artifacts_scanned": 2, "count": 2,
                "definitions": [
                    {"artifact": "/tmp/A.md", "artifact_type": "PRD", "line": 1, "checked": False},
                    {"artifact": "/tmp/B.md", "artifact_type": "DESIGN", "line": 5, "checked": False},
                ],
            })
        out = buf.getvalue()
        self.assertIn("Ambiguous", out)

    def test_no_line(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_defined({
                "status": "FOUND", "id": "cpt-x", "artifacts_scanned": 1, "count": 1,
                "definitions": [{"artifact": "/tmp/A.md", "artifact_type": "PRD", "line": "", "checked": False}],
            })
        # Should not crash


class TestHumanWhereUsed(_HumanModeBase):

    def test_found_with_checked(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_used({
                "id": "cpt-x", "artifacts_scanned": 1, "count": 1,
                "references": [{"artifact": "/tmp/A.md", "artifact_type": "PRD", "line": 10, "type": "reference", "checked": True}],
            })
        out = buf.getvalue()
        self.assertIn("cpt-x", out)

    def test_no_refs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_used({
                "id": "cpt-missing", "artifacts_scanned": 2, "count": 0, "references": [],
            })
        out = buf.getvalue()
        self.assertIn("No references", out)

    def test_no_line(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _human_where_used({
                "id": "cpt-x", "artifacts_scanned": 1, "count": 1,
                "references": [{"artifact": "/tmp/A.md", "artifact_type": "PRD", "line": "", "type": "ref", "checked": False}],
            })
        # Should not crash


# =========================================================================
# The documented contracts — architecture/specs/cli.md, query commands (#292, #348)
# =========================================================================

class TestDocumentedContracts(_ContextTestBase):
    """Each test pins one statement the CLI spec makes about a query command's JSON
    shape or exit code, against the real command. The spec was rewritten from these
    commands' actual output; these keep it that way."""

    def setUp(self):
        super().setUp()
        self._json_was = __import__("studio.utils.ui", fromlist=["is_json_mode"]).is_json_mode()
        set_json_mode(True)

    def tearDown(self):
        set_json_mode(self._json_was)
        super().tearDown()

    @staticmethod
    def _run(cmd, argv):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = cmd(argv)
        return rc, json.loads(stdout.getvalue())

    # ---- where-defined ---------------------------------------------------
    def test_where_defined_found_shape(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            rc, out = self._run(cmd_where_defined, ["cpt-test-item-1"])
        self.assertEqual(rc, 0)
        self.assertEqual(set(out), {"status", "id", "artifacts_scanned", "count", "definitions"})
        self.assertEqual(out["status"], "FOUND")
        self.assertEqual(out["count"], len(out["definitions"]), 1)
        record = out["definitions"][0]
        self.assertEqual(set(record), {"artifact", "artifact_type", "line", "kind", "checked"})
        self.assertIsNone(record["kind"])  # documented: this command does not infer it
        self.assertTrue(Path(record["artifact"]).is_absolute())

    def test_where_defined_not_found_keeps_the_shape_and_exits_2(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            rc, out = self._run(cmd_where_defined, ["cpt-test-item-nowhere"])
        self.assertEqual(rc, 2)
        self.assertEqual(set(out), {"status", "id", "artifacts_scanned", "count", "definitions"})
        self.assertEqual((out["status"], out["count"], out["definitions"]), ("NOT_FOUND", 0, []))

    def test_where_defined_ambiguous_lists_all_and_exits_2(self):
        with TemporaryDirectory() as td:
            first, second = Path(td) / "a.md", Path(td) / "b.md"
            for path in (first, second):
                path.write_text("# Doc\n\n**ID**: `cpt-test-item-1`\n", encoding="utf-8")
            with patch(
                "studio.commands.where_defined.resolve_target_and_artifacts",
                return_value=("cpt-test-item-1", object(), [(first, "PRD"), (second, "PRD")], {}, None),
            ):
                rc, out = self._run(cmd_where_defined, ["cpt-test-item-1"])
        self.assertEqual(rc, 2)
        self.assertEqual(out["status"], "AMBIGUOUS")
        self.assertEqual(out["count"], len(out["definitions"]), 2)

    # ---- list-ids ----------------------------------------------------------
    def test_list_ids_no_match_is_an_empty_answer_with_exit_0(self):
        """A real scan with nothing matching — not the hand-built dict the human
        renderer's tests use."""
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            rc, out = self._run(cmd_list_ids, ["--pattern", "zzz-matches-nothing"])
        self.assertEqual(rc, 0)
        self.assertEqual(out, {"count": 0, "artifacts_scanned": 1, "ids": []})

    # ---- get-content -------------------------------------------------------
    @staticmethod
    def _marked_code(td: Path) -> Path:
        """Two blocks for one ID, so "the first block" is distinguishable from "a block"."""
        code = td / "impl.py"
        code.write_text(
            "# @cpt-algo:cpt-test-algo-a:p1\n"
            "def outer():\n"
            "    # @cpt-begin:cpt-test-algo-a:p1:inst-one\n"
            "    one = 1\n"
            "    # @cpt-end:cpt-test-algo-a:p1:inst-one\n"
            "    # @cpt-begin:cpt-test-algo-a:p1:inst-two\n"
            "    two = 2\n"
            "    # @cpt-end:cpt-test-algo-a:p1:inst-two\n"
            "    return one + two\n",
            encoding="utf-8",
        )
        return code

    def test_get_content_code_without_inst_returns_the_first_block_with_null_inst(self):
        with TemporaryDirectory() as td:
            code = self._marked_code(Path(td))
            rc, first = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--code", str(code)])
            rc_two, second = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--code", str(code), "--inst", "two"])
        self.assertEqual((rc, rc_two), (0, 0))
        self.assertEqual(set(first), {"status", "id", "inst", "text"})
        self.assertIsNone(first["inst"])
        self.assertEqual(first["text"].strip(), "one = 1")   # the first block, not the whole scope
        self.assertEqual(second["inst"], "two")               # the bare id selects the block
        self.assertEqual(second["text"].strip(), "two = 2")

    def test_get_content_unknown_inst_falls_back_to_the_first_block_but_echoes_the_request(self):
        """Documented as a quirk: check `text`, not `inst`, to know what came back."""
        with TemporaryDirectory() as td:
            code = self._marked_code(Path(td))
            rc, out = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--code", str(code), "--inst", "absent"])
        self.assertEqual(rc, 0)
        self.assertEqual(out["inst"], "absent")
        self.assertEqual(out["text"].strip(), "one = 1")

    def test_get_content_prefixed_inst_never_matches_and_falls_back(self):
        """The marker parser stores `inst-two` as `two`, so the spelling the option's own
        help text suggests is one the lookup can never match: it falls back to the first
        block while `inst` echoes the request. Documented as the trap it is; when the
        command learns to strip the prefix, this test and the spec change together."""
        with TemporaryDirectory() as td:
            code = self._marked_code(Path(td))
            rc, out = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--code", str(code), "--inst", "inst-two"])
        self.assertEqual(rc, 0)
        self.assertEqual(out["inst"], "inst-two")
        self.assertEqual(out["text"].strip(), "one = 1")  # not "two = 2"

    def test_get_content_code_wins_when_both_paths_are_given(self):
        """The id is defined in the registered artifact and absent from the code file;
        the answer comes from the code file."""
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            code = self._marked_code(root)
            rc, out = self._run(cmd_get_content, [
                "--id", "cpt-test-item-1",
                "--artifact", str(root / "architecture" / "PRD.md"),
                "--code", str(code),
            ])
        self.assertEqual(rc, 2)
        self.assertEqual(out, {"status": "NOT_FOUND", "id": "cpt-test-item-1", "inst": None})

    def test_get_content_inst_is_ignored_with_artifact(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            rc, out = self._run(cmd_get_content, [
                "--id", "cpt-test-item-1",
                "--artifact", str(root / "architecture" / "PRD.md"),
                "--inst", "inst-anything",
            ])
        self.assertNotEqual(out["status"], "ERROR")
        self.assertNotIn("inst", out)  # the artifact branch never saw the flag

    def test_get_content_neither_path_is_an_error(self):
        rc, out = self._run(cmd_get_content, ["--id", "cpt-test-item-1"])
        self.assertEqual(rc, 1)
        self.assertEqual(out["status"], "ERROR")

    def test_get_content_unparsable_code_file_exits_1(self):
        with TemporaryDirectory() as td:
            code = Path(td) / "broken.py"
            code.write_text("# @cpt-begin:cpt-test-algo-a:p1:inst-one\nx = 1\n", encoding="utf-8")
            rc, out = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--code", str(code)])
        self.assertEqual(rc, 1)
        self.assertEqual(out["status"], "ERROR")
        self.assertIn("marker-begin-no-end", out["message"])

    def test_get_content_code_needs_no_project_but_artifact_does(self):
        """Documented asymmetry: `--code` reads the file directly, `--artifact` resolves
        a registered artifact. Run from a directory that is not a Studio project."""
        with TemporaryDirectory() as td:
            outside = Path(td)
            code = self._marked_code(outside)
            doc = outside / "doc.md"
            doc.write_text("# A\n\n**ID**: `cpt-test-algo-a`\n\nbody\n", encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(outside)
                rc_code, from_code = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--code", str(code)])
                rc_art, from_artifact = self._run(cmd_get_content, ["--id", "cpt-test-algo-a", "--artifact", str(doc)])
            finally:
                os.chdir(cwd)
        self.assertEqual((rc_code, from_code["status"]), (0, "FOUND"))
        self.assertEqual((rc_art, from_artifact["status"]), (1, "ERROR"))
        self.assertIn("not initialized", from_artifact["message"])

    # ---- list-ids, --include-code ---------------------------------------------
    def _list_ids_with_code_scan(self, argv, scan_result):
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            argv = [a.replace("<PRD>", str(root / "architecture" / "PRD.md")) for a in argv]
            with patch("studio.commands.list_ids.scan_registered_codebase_references",
                       return_value=scan_result) as scan:
                rc, out = self._run(cmd_list_ids, argv)
        return rc, out, scan

    def test_list_ids_source_is_an_error_outside_a_workspace_unless_artifact_is_given(self):
        """Documented: `--source` needs workspace mode — and `--artifact` wins, so with
        both the source is never consulted and the would-be error does not happen."""
        with TemporaryDirectory() as td:
            root = Path(td)
            _setup_project(root)
            _with_context(root)
            rc, alone = self._run(cmd_list_ids, ["--source", "anything"])
            rc_both, both = self._run(cmd_list_ids, [
                "--artifact", str(root / "architecture" / "PRD.md"), "--source", "anything"])
        self.assertEqual((rc, alone["status"]), (1, "ERROR"))
        self.assertIn("workspace", alone["message"])
        self.assertEqual(rc_both, 0)
        self.assertEqual((both["artifacts_scanned"], both["count"]), (1, 1))

    def test_list_ids_code_files_skipped_appears_only_when_nonzero(self):
        rc, clean, _ = self._list_ids_with_code_scan(["--include-code"], ([], 3, 0))
        rc2, skipped, _ = self._list_ids_with_code_scan(["--include-code"], ([], 3, 2))
        self.assertEqual((rc, rc2), (0, 0))
        self.assertEqual(clean["code_files_scanned"], 3)
        self.assertNotIn("code_files_skipped", clean)
        self.assertEqual(skipped["code_files_skipped"], 2)

    def test_list_ids_include_code_is_a_no_op_with_artifact_but_still_reports_zero(self):
        """Pins a wart, not a wish: the scan is skipped, yet `code_files_scanned` is
        emitted as 0 — unlike `where-used`, which omits both counters. Documented as
        such; if list-ids is aligned with where-used, this test and the spec change
        together."""
        rc, out, scan = self._list_ids_with_code_scan(["--artifact", "<PRD>", "--include-code"], ([], 3, 2))
        self.assertEqual(rc, 0)
        scan.assert_not_called()
        self.assertEqual(out["code_files_scanned"], 0)   # not 3: the scan never ran
        self.assertNotIn("code_files_skipped", out)


if __name__ == "__main__":
    unittest.main()
