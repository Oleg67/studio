"""Tests for change-summary requirement linkage — changed files to the IDs they carry.

The contract is the same as the window's: never raise, never go silent, and always
report the denominator. So most of these force a failure or an awkward file shape
rather than confirming the happy path.

Git setup helpers are imported from the window suite rather than copied — `tests/` is
on `sys.path` via conftest, and a second copy of the same four git calls is duplication
a reviewer would rightly flag.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from studio.utils import change_summary as cs
from test_change_summary_core import _commit, _git, _make_repo, _point_ref


# --------------------------------------------------------------------------- helpers

MARKER = "cpt-studio-algo-developer-experience-change-summary"


def _repo_with_base(tmp_path: Path) -> Path:
    """A repo whose branch point is one commit behind HEAD."""
    repo = _make_repo(tmp_path / "r")
    _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))
    return repo


def _code(body: str = "") -> str:
    return (
        f'"""Module.\n\n@cpt-algo:{MARKER}:p1\n"""\n\n'
        f"# @cpt-begin:{MARKER}:p1:inst-x\n"
        f"def f():\n    return 1\n{body}"
        f"# @cpt-end:{MARKER}:p1:inst-x\n"
    )


def _artifact() -> str:
    return f"# Feature\n\n- [x] `p1` - **ID**: `{MARKER}`\n\n1. [x] - `p1` - step - `inst-x`\n"


def _report(repo: Path) -> cs.LinkReport:
    return cs.link_changed_files(repo, cs.resolve_window(repo))


# ----------------------------------------------------------------- name-status parsing

class TestNameStatusParsing:

    @pytest.mark.parametrize("line,expected", [
        ("M\tsrc/a.py", ("M", "src/a.py")),
        ("A\tnew.py", ("A", "new.py")),
        ("D\tgone.py", ("D", "gone.py")),
        ("R100\told.py\tnew.py", ("R", "new.py")),
        ("C075\tsrc.py\tcopy.py", ("C", "copy.py")),
        ("T\tmode.py", ("T", "mode.py")),
    ])
    def test_each_status_shape_parses(self, line, expected):
        assert cs._parse_name_status(line) == expected

    def test_a_rename_yields_the_new_path_not_the_old(self):
        """Taking parts[1] would send every rename to the unreadable branch."""
        assert cs._parse_name_status("R100\told.py\tnew.py") == ("R", "new.py")

    @pytest.mark.parametrize("line", ["", "M", "\tno-status", "   "])
    def test_malformed_lines_are_dropped_not_raised(self, line):
        assert cs._parse_name_status(line) is None


# ------------------------------------------------------------------ per-file traceability

class TestWhatAFileDoesWithIds:

    def test_code_reports_references_and_no_definitions(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text(_code(), encoding="utf-8")

        references, defines, reason = cs._file_traceability(path)

        assert references == [MARKER]
        assert defines == []
        assert reason == cs.REASON_OK

    def test_an_artifact_reports_definitions_and_no_references(self, tmp_path):
        path = tmp_path / "f.md"
        path.write_text(_artifact(), encoding="utf-8")

        references, defines, reason = cs._file_traceability(path)

        assert defines == [MARKER], "a changed spec declares requirements"
        assert references == [], "and does not reference them as code"

    def test_a_file_with_no_markers_reports_neither_and_no_failure(self, tmp_path):
        path = tmp_path / "plain.py"
        path.write_text("x = 1\n", encoding="utf-8")

        references, defines, reason = cs._file_traceability(path)

        assert (references, defines, reason) == ([], [], cs.REASON_OK)

    def test_a_binary_file_is_unreadable_not_marker_free(self, tmp_path):
        """"Carries no markers" and "could not be read" are different claims."""
        path = tmp_path / "blob.py"
        path.write_bytes(b"\xff\xfe\x00\x01binary\x00")

        references, defines, reason = cs._file_traceability(path)

        assert reason == cs.REASON_FILE_UNREADABLE
        assert (references, defines) == ([], [])

    def test_ids_are_sorted_and_deduplicated(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text(_code() + _code().replace("inst-x", "inst-y"), encoding="utf-8")

        references, _defines, _reason = cs._file_traceability(path)

        assert references == sorted(set(references))


# ------------------------------------------------------------------------- scope

class TestProjectScope:

    def test_a_file_inside_the_project_is_in_scope(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

        assert cs._in_project_scope(repo / "a.py", repo) is True

    def test_a_file_outside_the_project_is_not(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        outside = tmp_path / "elsewhere.py"
        outside.write_text("x = 1\n", encoding="utf-8")

        assert cs._in_project_scope(outside, repo) is False

    def test_a_missing_file_is_not_in_scope(self, tmp_path):
        repo = _make_repo(tmp_path / "r")

        assert cs._in_project_scope(repo / "never.py", repo) is False


# ------------------------------------------------------------------- the whole report

class TestTheReport:

    def test_a_changed_code_file_links_to_its_requirement(self, tmp_path):
        repo = _repo_with_base(tmp_path)
        (repo / "m.py").write_text(_code(), encoding="utf-8")
        _git(repo, "add", "m.py")
        _git(repo, "commit", "-q", "-m", "add module")

        report = _report(repo)

        assert report.available is True
        assert report.changed == 1
        assert report.linked == 1
        assert [f.references for f in report.files] == [[MARKER]]

    def test_a_changed_artifact_counts_as_declaring_not_unlinked(self, tmp_path):
        repo = _repo_with_base(tmp_path)
        (repo / "f.md").write_text(_artifact(), encoding="utf-8")
        _git(repo, "add", "f.md")
        _git(repo, "commit", "-q", "-m", "add spec")

        report = _report(repo)

        assert report.declaring == 1
        assert report.linked == 0, "a spec is a requirement source, not code serving one"

    def test_an_untracked_file_is_reported_not_omitted(self, tmp_path):
        """git diff cannot see it, so omitting it would hide a brand-new module."""
        repo = _repo_with_base(tmp_path)
        (repo / "fresh.py").write_text(_code(), encoding="utf-8")

        report = _report(repo)

        statuses = {f.status for f in report.files}
        assert "?" in statuses
        assert report.linked == 1

    def test_a_deleted_file_is_reported_as_gone_not_excluded(self, tmp_path):
        repo = _repo_with_base(tmp_path)
        (repo / "a.txt").unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "delete")

        report = _report(repo)

        gone = [f for f in report.files if f.reason == cs.REASON_FILE_GONE]
        assert len(gone) == 1
        assert report.excluded == 0, "a deletion is not a policy exclusion"

    def test_a_renamed_file_keeps_its_requirement_link(self, tmp_path):
        repo = _repo_with_base(tmp_path)
        (repo / "m.py").write_text(_code(), encoding="utf-8")
        _git(repo, "add", "m.py")
        _git(repo, "commit", "-q", "-m", "add")
        _git(repo, "mv", "m.py", "renamed.py")
        _git(repo, "commit", "-q", "-m", "rename")

        report = _report(repo)

        linked = [f for f in report.files if f.references]
        assert linked, "a rename must not lose the link"
        assert all(f.path.endswith("renamed.py") for f in linked)

    def test_a_binary_change_is_counted_unreadable(self, tmp_path):
        repo = _repo_with_base(tmp_path)
        (repo / "blob.py").write_bytes(b"\xff\xfe\x00binary")
        _git(repo, "add", "blob.py")
        _git(repo, "commit", "-q", "-m", "add blob")

        report = _report(repo)

        assert report.unreadable == 1
        assert report.linked == 0

    def test_no_changes_is_available_with_a_zero_denominator(self, tmp_path):
        repo = _repo_with_base(tmp_path)

        report = _report(repo)

        assert report.available is True
        assert (report.changed, report.linked) == (0, 0)
        assert report.reason == cs.REASON_OK


class TestTheReportSaysWhyItCouldNot:

    def test_an_unavailable_window_propagates_exactly_one_reason(self, tmp_path):
        window = cs.resolve_window(tmp_path)   # not a repo

        report = cs.link_changed_files(tmp_path, window)

        assert report.available is False
        assert report.reason == window.reason

    def test_a_since_only_window_has_no_base_commit_to_diff(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)

        report = cs.link_changed_files(tmp_path, window)

        assert report.available is False
        assert report.reason == cs.REASON_NO_BASE_COMMIT

    def test_a_failing_diff_is_reported(self, tmp_path, monkeypatch):
        repo = _repo_with_base(tmp_path)
        window = cs.resolve_window(repo)
        monkeypatch.setattr(cs, "_git_lines", lambda *_a, **_k: None)

        report = cs.link_changed_files(repo, window)

        assert report.available is False
        assert report.reason == cs.REASON_DIFF_UNAVAILABLE


class TestInvariants:

    def test_git_lines_degrades_rather_than_raising(self, tmp_path, monkeypatch):
        def _boom(*_a, **_k):
            raise OSError("no exec")
        monkeypatch.setattr(cs.subprocess, "run", _boom)

        assert cs._git_lines(tmp_path, ["diff"]) is None

    def test_git_lines_returns_nothing_on_a_non_zero_exit(self, tmp_path):
        repo = _make_repo(tmp_path / "r")

        assert cs._git_lines(repo, ["rev-parse", "--verify", "refs/heads/no-such"]) is None

    def test_a_scope_check_that_errors_refuses_rather_than_admits(self, tmp_path, monkeypatch):
        def _boom(*_a, **_k):
            raise OSError("stat failed")
        monkeypatch.setattr(cs.codebase, "resolve_entry_code_files", _boom)

        assert cs._in_project_scope(tmp_path / "a.py", tmp_path) is False

    def test_an_out_of_scope_change_is_counted_excluded_not_dropped(self, tmp_path):
        """A tracked symlink is refused by the shared policy but still appears in the
        diff, so it must land in `excluded` — a published counter that would otherwise
        never be exercised."""
        if os.name == "nt":
            pytest.skip("symlink creation needs privileges on Windows")
        repo = _repo_with_base(tmp_path)
        outside = tmp_path / "outside.py"
        outside.write_text(_code(), encoding="utf-8")
        (repo / "link.py").symlink_to(outside)
        _git(repo, "add", "link.py")
        _git(repo, "commit", "-q", "-m", "add symlink")

        report = _report(repo)

        assert report.excluded == 1, "refused by policy, but still counted"
        assert report.changed == 1
        assert all(not f.path.endswith("link.py") for f in report.files)

    def test_every_new_reason_is_path_free(self):
        home = os.path.expanduser("~")
        for name in ("REASON_NO_BASE_COMMIT", "REASON_DIFF_UNAVAILABLE",
                     "REASON_FILE_GONE", "REASON_FILE_UNREADABLE"):
            value = getattr(cs, name)
            assert value and os.sep not in value and home not in value

    def test_the_counters_partition_what_was_seen(self, tmp_path):
        """linked/declaring overlap by design, but neither may exceed what was seen."""
        repo = _repo_with_base(tmp_path)
        (repo / "m.py").write_text(_code(), encoding="utf-8")
        (repo / "f.md").write_text(_artifact(), encoding="utf-8")
        (repo / "blob.py").write_bytes(b"\xff\xfebinary")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "mixed")

        report = _report(repo)

        assert report.changed == len(report.files) + report.excluded
        assert report.linked <= report.changed
        assert report.declaring <= report.changed
        assert report.unreadable <= report.changed

    def test_the_same_state_yields_identical_reports(self, tmp_path):
        repo = _repo_with_base(tmp_path)
        (repo / "m.py").write_text(_code(), encoding="utf-8")
        _git(repo, "add", "m.py")
        _git(repo, "commit", "-q", "-m", "add")

        shapes = set()
        for _ in range(5):
            report = _report(repo)
            shapes.add((
                report.changed, report.linked, report.declaring,
                tuple((f.path, f.status, tuple(f.references)) for f in report.files),
            ))

        assert len(shapes) == 1


class TestGoldenShape:

    def test_a_mixed_change_set_renders_a_stable_report(self, tmp_path):
        """Regression fixture: a later change that degrades the output is caught."""
        repo = _repo_with_base(tmp_path)
        (repo / "m.py").write_text(_code(), encoding="utf-8")
        (repo / "f.md").write_text(_artifact(), encoding="utf-8")
        (repo / "plain.py").write_text("x = 1\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "mixed")

        report = _report(repo)
        actual = sorted(
            (f.path, f.status, len(f.references), len(f.defines), f.reason)
            for f in report.files
        )

        assert actual == [
            ("f.md", "A", 0, 1, ""),
            ("m.py", "A", 1, 0, ""),
            ("plain.py", "A", 0, 0, ""),
        ]
        assert (report.changed, report.linked, report.declaring) == (3, 1, 1)
        assert (report.excluded, report.unreadable) == (0, 0)
