"""Tests for the change-summary core — window resolution and event selection.

The module's whole contract is *never raise, never go silent*: every failure path
returns a value carrying a reason. So these tests are weighted toward forcing each
failure rather than confirming the happy path, and several assert on the *reason*
rather than merely on emptiness — an empty result with the wrong explanation is the
defect this feature exists to remove.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest

from studio.utils import change_summary as cs
from studio.utils import decision_log


# --------------------------------------------------------------------------- helpers

def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _make_repo(repo: Path) -> Path:
    """A minimal repo with one commit and deterministic identity."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _commit(repo: Path, name: str, body: str = "x\n") -> str:
    (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", f"add {name}")
    return _git(repo, "rev-parse", "HEAD")


def _point_ref(repo: Path, ref: str, sha: str) -> None:
    """Create a remote-tracking ref without needing a real remote."""
    _git(repo, "update-ref", ref, sha)


def _write_log(path: Path, rows: list, extra: str = "") -> Path:
    body = "".join(json.dumps(r) + "\n" for r in rows) + extra
    path.write_text(body, encoding="utf-8")
    return path


def _event(ts: str, run_id: str = "r1", event: str = "validation") -> dict:
    return {"schema": 1, "ts": ts, "run_id": run_id, "decision_id": "d",
            "event": event, "command": "validate", "payload": {}}


# --------------------------------------------------------------------------- window

class TestTheWindowComesFromGit:

    def test_a_repo_with_a_base_ref_yields_a_window(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        base = _git(repo, "rev-parse", "HEAD")
        _point_ref(repo, "refs/remotes/upstream/main", base)
        _commit(repo, "b.txt")

        window = cs.resolve_window(repo)

        assert window.available is True
        assert window.reason == cs.REASON_OK
        assert window.base_ref == "upstream/main"
        assert window.base_sha == base
        assert window.since  # the base commit's own time

    def test_an_explicit_since_short_circuits_git_entirely(self):
        # A path that is not a repo, and does not exist: proof no git call is needed.
        window = cs.resolve_window(Path("/nonexistent-abc"), since="2026-01-01T00:00:00+00:00")

        assert window.available is True
        assert window.since == "2026-01-01T00:00:00+00:00"
        assert window.base_ref == ""

    def test_an_explicit_base_ref_is_honoured(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        base = _git(repo, "rev-parse", "HEAD")
        _git(repo, "branch", "release")
        _commit(repo, "b.txt")

        window = cs.resolve_window(repo, base="release")

        assert window.base_ref == "release"
        assert window.base_sha == base

    def test_an_unknown_explicit_base_ref_is_refused_not_swapped(self, tmp_path):
        """The mutation that matters: a default ref exists, so a fallback would 'work'."""
        repo = _make_repo(tmp_path / "r")
        _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))

        window = cs.resolve_window(repo, base="no-such-ref")

        assert window.available is False
        assert window.reason == cs.REASON_BASE_REF_UNKNOWN
        # Must NOT have silently measured against the discoverable default.
        assert window.base_ref == ""

    def test_the_canonical_remote_is_preferred_over_a_lagging_fork(self, tmp_path):
        """In a fork workflow origin is the contributor's fork and lags upstream.

        Preferring origin/HEAD would widen the window to include long-shipped work.
        """
        repo = _make_repo(tmp_path / "r")
        old = _git(repo, "rev-parse", "HEAD")
        newer = _commit(repo, "b.txt")
        _commit(repo, "c.txt")
        _point_ref(repo, "refs/remotes/origin/HEAD", old)       # the stale fork
        _point_ref(repo, "refs/remotes/upstream/main", newer)    # the canonical remote

        window = cs.resolve_window(repo)

        assert window.base_ref == "upstream/main"
        assert window.base_sha == newer


class TestTheWindowSaysWhyItCouldNotBeBuilt:

    def test_a_non_repo_is_reported_not_crashed(self, tmp_path):
        window = cs.resolve_window(tmp_path)

        assert window.available is False
        assert window.reason == cs.REASON_NOT_A_REPO

    def test_git_unavailable_is_distinguished_from_not_a_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "_git_line", lambda *_a, **_k: None)

        window = cs.resolve_window(tmp_path)

        assert window.reason == cs.REASON_GIT_UNAVAILABLE

    def test_no_discoverable_base_ref_is_reported(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        # Rename the only branch to something none of the candidates match.
        _git(repo, "branch", "-m", "wip-nothing-standard")

        window = cs.resolve_window(repo)

        assert window.available is False
        assert window.reason == cs.REASON_NO_BASE_REF

    def test_unrelated_histories_have_no_merge_base(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "r")
        _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))
        monkeypatch.setattr(cs, "_merge_base", lambda *_a, **_k: None)

        window = cs.resolve_window(repo)

        assert window.available is False
        assert window.reason == cs.REASON_NO_MERGE_BASE
        assert window.base_ref == "upstream/main"   # what we did learn is still reported

    def test_a_base_commit_without_a_readable_time_is_reported(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "r")
        _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))
        monkeypatch.setattr(cs, "_commit_time", lambda *_a, **_k: None)

        window = cs.resolve_window(repo)

        assert window.reason == cs.REASON_NO_BASE_TIME
        assert window.base_sha  # still reported

    def test_the_reason_list_has_no_enumerated_ref_names(self):
        """The first version of this string listed the refs it tried and went stale."""
        assert "origin" not in cs.REASON_NO_BASE_REF
        assert "main" not in cs.REASON_NO_BASE_REF


# --------------------------------------------------------------------------- events

class TestEventSelection:

    def test_events_before_the_boundary_are_excluded(self, tmp_path):
        window = cs.ChangeWindow(since="2026-06-01T00:00:00+00:00", available=True)
        log = _write_log(tmp_path / "d.jsonl", [
            _event("2026-05-31T23:59:59+00:00", "old"),
            _event("2026-06-02T00:00:00+00:00", "new"),
        ])

        selection = cs.select_events(window, path=log)

        assert selection.available is True
        assert [e["run_id"] for e in selection.events] == ["new"]
        assert selection.scanned == 2

    def test_an_event_exactly_at_the_boundary_is_included(self, tmp_path):
        boundary = "2026-06-01T00:00:00+00:00"
        window = cs.ChangeWindow(since=boundary, available=True)
        log = _write_log(tmp_path / "d.jsonl", [_event(boundary, "edge")])

        selection = cs.select_events(window, path=log)

        assert len(selection.events) == 1, "the branch point itself belongs to the window"

    def test_a_z_suffix_and_an_offset_compare_correctly(self, tmp_path):
        window = cs.ChangeWindow(since="2026-06-01T00:00:00Z", available=True)
        log = _write_log(tmp_path / "d.jsonl", [
            _event("2026-06-01T01:00:00+02:00", "before"),   # 23:00 previous day UTC
            _event("2026-06-01T03:00:00+02:00", "after"),    # 01:00 UTC
        ])

        selection = cs.select_events(window, path=log)

        assert [e["run_id"] for e in selection.events] == ["after"]

    def test_unparseable_lines_are_counted_not_hidden(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)
        log = _write_log(
            tmp_path / "d.jsonl",
            [_event("2026-06-01T00:00:00+00:00")],
            extra="{ this is not json\n\n[]\n",
        )

        selection = cs.select_events(window, path=log)

        assert selection.skipped_lines >= 1, "corruption must be reported, not swallowed"

    def test_undated_events_are_excluded_and_counted(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)
        rows = [
            _event("2026-06-01T00:00:00+00:00", "dated"),
            {"schema": 1, "run_id": "no-ts", "event": "validation", "payload": {}},
            _event("not-a-timestamp", "bad-ts"),
        ]
        log = _write_log(tmp_path / "d.jsonl", rows)

        selection = cs.select_events(window, path=log)

        assert [e["run_id"] for e in selection.events] == ["dated"]
        assert selection.undated == 2, "cannot-tell is counted, never guessed either way"

    def test_a_naive_timestamp_is_refused_rather_than_assumed_utc(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)
        log = _write_log(tmp_path / "d.jsonl", [_event("2026-06-01T00:00:00", "naive")])

        selection = cs.select_events(window, path=log)

        assert selection.events == []
        assert selection.undated == 1

    def test_runs_preserve_first_seen_order(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)
        log = _write_log(tmp_path / "d.jsonl", [
            _event("2026-06-01T00:00:01+00:00", "second"),
            _event("2026-06-01T00:00:02+00:00", "first"),
            _event("2026-06-01T00:00:03+00:00", "second"),
        ])

        selection = cs.select_events(window, path=log)

        assert selection.runs == ["second", "first"], "first-seen order, not sorted"

    def test_the_scanned_count_is_reported_even_when_nothing_is_selected(self, tmp_path):
        window = cs.ChangeWindow(since="2026-12-01T00:00:00+00:00", available=True)
        log = _write_log(tmp_path / "d.jsonl", [
            _event("2026-06-01T00:00:00+00:00"), _event("2026-06-02T00:00:00+00:00"),
        ])

        selection = cs.select_events(window, path=log)

        assert selection.events == []
        assert selection.scanned == 2, "a verdict without its denominator is the defect"


class TestEventSelectionSaysWhyItCouldNotRead:

    def test_an_unavailable_window_propagates_exactly_one_reason(self, tmp_path):
        window = cs.resolve_window(tmp_path)   # not a repo

        selection = cs.select_events(window)

        assert selection.available is False
        assert selection.reason == window.reason, "one cause reported, not two"

    def test_a_disabled_log_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFS_DECISION_LOG", "0")
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)

        selection = cs.select_events(window, path=tmp_path / "d.jsonl")

        assert selection.reason == cs.REASON_LOG_DISABLED

    def test_an_absent_log_is_reported(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)

        selection = cs.select_events(window, path=tmp_path / "never-written.jsonl")

        assert selection.reason == cs.REASON_LOG_ABSENT

    def test_outside_a_studio_project_is_reported(self, monkeypatch):
        monkeypatch.delenv("CFS_DECISION_LOG", raising=False)
        monkeypatch.setattr(decision_log, "default_log_path", lambda: None)
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)

        selection = cs.select_events(window)

        assert selection.reason == cs.REASON_NOT_A_PROJECT

    def test_a_window_whose_since_will_not_parse_is_reported(self, tmp_path):
        window = cs.ChangeWindow(since="nonsense", available=True)
        log = _write_log(tmp_path / "d.jsonl", [_event("2026-06-01T00:00:00+00:00")])

        selection = cs.select_events(window, path=log)

        assert selection.available is False
        assert selection.reason == cs.REASON_NO_BASE_TIME


# --------------------------------------------------------------------- fail-safe

class TestNothingRaises:

    def test_a_git_timeout_degrades(self, tmp_path, monkeypatch):
        def _boom(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="git", timeout=1)
        monkeypatch.setattr(cs.subprocess, "run", _boom)

        assert cs._git_line(tmp_path, ["status"]) is None

    def test_an_oserror_from_git_degrades(self, tmp_path, monkeypatch):
        def _boom(*_a, **_k):
            raise OSError("no exec")
        monkeypatch.setattr(cs.subprocess, "run", _boom)

        window = cs.resolve_window(tmp_path)

        assert window.available is False
        assert window.reason == cs.REASON_GIT_UNAVAILABLE

    def test_an_unreadable_log_is_reported_not_raised(self, tmp_path, monkeypatch):
        log = _write_log(tmp_path / "d.jsonl", [_event("2026-06-01T00:00:00+00:00")])
        monkeypatch.setattr(Path, "is_file", lambda _self: (_ for _ in ()).throw(OSError("nope")))
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)

        selection = cs.select_events(window, path=log)

        assert selection.reason == cs.REASON_LOG_ABSENT

    def test_the_line_count_degrades_to_zero_on_a_read_error(self, tmp_path, monkeypatch):
        log = _write_log(tmp_path / "d.jsonl", [_event("2026-06-01T00:00:00+00:00")])
        monkeypatch.setattr(Path, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("nope")))

        assert cs._count_log_lines(log) == 0

    def test_hostile_event_shapes_do_not_raise(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)
        log = tmp_path / "d.jsonl"
        log.write_text(
            json.dumps({"ts": 12345, "run_id": None}) + "\n"
            + json.dumps({"ts": ["list"], "run_id": {"d": 1}}) + "\n"
            + json.dumps("a bare string") + "\n",
            encoding="utf-8",
        )

        selection = cs.select_events(window, path=log)

        assert selection.available is True
        assert selection.events == []


# ---------------------------------------------------------------------- invariants

class TestInvariants:

    @pytest.mark.parametrize("reason_name", [
        n for n in dir(cs) if n.startswith("REASON_") and n != "REASON_OK"
    ])
    def test_every_reason_is_human_readable_prose(self, reason_name):
        value = getattr(cs, reason_name)
        assert isinstance(value, str) and value
        assert value == value.lower() or value[0].islower(), "reason reads as prose, not a code"

    def test_an_unavailable_result_always_carries_a_reason(self, tmp_path):
        for result in (cs.resolve_window(tmp_path), cs.select_events(cs.ChangeWindow())):
            assert result.available is False
            assert result.reason, "unavailable without a reason is the silent failure"

    def test_an_available_window_carries_no_reason(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))

        assert cs.resolve_window(repo).reason == cs.REASON_OK


class TestPrivacy:

    def test_no_reason_string_leaks_a_path_home_or_username(self):
        import os
        home = os.path.expanduser("~")
        user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
        for name in dir(cs):
            if not name.startswith("REASON_"):
                continue
            value = getattr(cs, name)
            assert os.sep not in value, f"{name} contains a path separator"
            assert home not in value
            if user:
                assert user not in value

    def test_no_network_is_used(self, tmp_path, monkeypatch):
        """Prove it rather than assert it: make sockets impossible and still work."""
        def _no_sockets(*_a, **_k):
            raise AssertionError("change_summary must not open a socket")
        monkeypatch.setattr(socket, "socket", _no_sockets)

        repo = _make_repo(tmp_path / "r")
        _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))
        window = cs.resolve_window(repo)
        log = _write_log(tmp_path / "d.jsonl", [_event("2026-06-01T00:00:00+00:00")])

        assert window.available is True
        assert cs.select_events(window, path=log).available is True


class TestDeterminism:

    def test_the_same_state_yields_identical_results(self, tmp_path):
        repo = _make_repo(tmp_path / "r")
        _point_ref(repo, "refs/remotes/upstream/main", _git(repo, "rev-parse", "HEAD"))
        _commit(repo, "b.txt")
        log = _write_log(tmp_path / "d.jsonl", [
            _event("2026-06-01T00:00:00+00:00", "a"), _event("2026-06-02T00:00:00+00:00", "b"),
        ])

        windows, selections = [], []
        for _ in range(5):
            window = cs.resolve_window(repo)
            windows.append(window)
            selections.append(cs.select_events(window, path=log))

        assert len({(w.base_ref, w.base_sha, w.since) for w in windows}) == 1
        assert len({(tuple(s.runs), s.scanned, s.undated) for s in selections}) == 1


class TestGrouping:

    def test_grouping_preserves_run_order(self, tmp_path):
        window = cs.ChangeWindow(since="2026-01-01T00:00:00+00:00", available=True)
        log = _write_log(tmp_path / "d.jsonl", [
            _event("2026-06-01T00:00:01+00:00", "z"),
            _event("2026-06-01T00:00:02+00:00", "a"),
            _event("2026-06-01T00:00:03+00:00", "z"),
        ])
        selection = cs.select_events(window, path=log)

        grouped = cs.group_by_run(selection)

        assert list(grouped) == ["z", "a"], "run order follows the log, not the alphabet"
        assert [len(v) for v in grouped.values()] == [2, 1]

    def test_grouping_an_empty_selection_is_empty_not_an_error(self):
        assert cs.group_by_run(cs.EventSelection()) == {}
