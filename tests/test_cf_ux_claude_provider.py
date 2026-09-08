"""Tests for the cf-ux Claude provider — that a run which never loaded the skill
is not scored as one that did.

The suite these back cannot run here: `claude` is a CLI this repository does not
vendor, and a real invocation costs money and needs credentials. So the provider
is driven with a faked `subprocess.run` and a faked sandbox, which is enough to
pin the three things that went wrong — the missing permission flag, a guard that
searched for a string the CLI never emits, and a skill failure recorded as
metadata where the grader would never see it.

What these tests deliberately do *not* claim: that `--permission-mode
bypassPermissions` makes the skill execute. That is a fact about the CLI, and it
belongs to whoever can run one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_PROVIDERS = Path(__file__).resolve().parents[1] / "tests" / "prompts" / "cf-ux" / "providers"
if str(_PROVIDERS) not in sys.path:
    sys.path.insert(0, str(_PROVIDERS))

claude_provider = pytest.importorskip("claude_provider")


# --------------------------------------------------------------------------- helpers

def _skill_call(call_id: str = "t1", name: str = "Skill", skill: str = "cf") -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": call_id, "name": name, "input": {"command": skill}},
    ]}}


def _skill_result(call_id: str = "t1", *, is_error: bool = False) -> dict:
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": call_id, "is_error": is_error, "content": "..."},
    ]}}


def _result(text: str = "the answer", cost: float = 0.01) -> dict:
    return {"type": "result", "subtype": "success", "result": text,
            "session_id": "s1", "num_turns": 3, "total_cost_usd": cost}


def _stream(*events: dict) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


@pytest.fixture
def run_provider(tmp_path, monkeypatch):
    """Drive `call_api` against a canned transcript; hand back the call's argv."""
    seen: dict = {}

    @contextmanager
    def _fake_sandbox():
        yield tmp_path

    def _make(stdout: str, returncode: int = 0, stderr: str = ""):
        def _fake_run(cmd, **_kwargs):
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

        monkeypatch.setattr(claude_provider, "sandbox", _fake_sandbox)
        monkeypatch.setattr(claude_provider.subprocess, "run", _fake_run)
        return claude_provider.call_api("write a PRD"), seen

    return _make


# ------------------------------------------------------- the permission flag

class TestTheSkillIsAllowedToExecute:

    def test_the_invocation_carries_a_permission_mode(self, run_provider):
        """The defect itself: with no permission flag, skill execution is denied in
        print mode because there is nobody to ask, so the agent answers directly
        and the run scores the fallback path."""
        _out, seen = run_provider(_stream(_skill_call(), _skill_result(), _result()))

        cmd = seen["cmd"]
        assert "--permission-mode" in cmd, "print mode has nobody to grant permission"
        assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"

    def test_the_transcript_is_requested_not_only_the_answer(self, run_provider):
        """Tool-use events are the only place the run says whether the skill was
        reached, and `-p` emits them only in the streaming format."""
        _out, seen = run_provider(_stream(_skill_call(), _skill_result(), _result()))

        cmd = seen["cmd"]
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in cmd, "without it `-p` returns the result alone"


# ------------------------------------------------- a run that loaded the skill

class TestARunThatLoadedTheSkillIsScored:

    def test_the_answer_and_the_totals_come_back(self, run_provider):
        out, _seen = run_provider(
            _stream(_skill_call(), _skill_result(), _result("gate rendered", cost=0.02)),
        )

        assert "error" not in out
        assert out["output"] == "gate rendered"
        assert out["cost"] == 0.02
        assert out["metadata"]["skill_state"] == "ran"
        assert out["metadata"]["num_turns"] == 3

    def test_a_skill_that_ran_alongside_a_failed_unrelated_tool_still_counts(self, run_provider):
        """Only the `Skill` call's own result decides this. An unrelated tool
        erroring mid-workflow is ordinary, and must not read as the skill failing."""
        other = {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "other", "is_error": True, "content": "nope"},
        ]}}
        out, _seen = run_provider(
            _stream(_skill_call(), _skill_result(), other, _result()),
        )

        assert "error" not in out
        assert out["metadata"]["skill_state"] == "ran"


# --------------------------------------------- a run that did not, is not scored

class TestARunThatNeverLoadedTheSkillIsNotScored:
    """The heart of the report: the fallback answer is plausible and well-formed,
    so left to the grader it scores as a pass and the suite reports confidently on
    an agent that never loaded Studio."""

    def test_no_skill_call_at_all_is_an_error_not_an_output(self, run_provider):
        out, _seen = run_provider(_stream(_result("here is a PRD I wrote myself")))

        assert "output" not in out, "a fallback answer must never reach the grader"
        assert "did not run (absent)" in out["error"]
        assert out["metadata"]["skill_state"] == "absent"
        assert out["metadata"]["unscored_output"].startswith("here is a PRD")

    def test_a_skill_call_that_errored_is_an_error(self, run_provider):
        out, _seen = run_provider(
            _stream(_skill_call(), _skill_result(is_error=True), _result("drafted directly")),
        )

        assert "output" not in out
        assert "did not run (failed)" in out["error"]

    def test_the_real_failure_signature_is_recognised(self, run_provider):
        """The observed failure is `<error>Execute skill: cf</error>`. The previous
        guard looked for "skills failed to load", which the CLI never emits, so the
        one check meant to catch this could not fire."""
        transcript = _stream(_result("<error>Execute skill: cf</error> so I answered"))

        out, _seen = run_provider(transcript)

        assert "output" not in out
        assert "Execute skill:" in out["error"]

    def test_a_different_skill_running_is_not_the_cf_skill_running(self, run_provider):
        out, _seen = run_provider(
            _stream(_skill_call(skill="superpowers"), _skill_result(), _result()),
        )

        assert "output" not in out
        assert "none of them named 'cf'" in out["error"]
        assert out["metadata"]["skills_invoked"], "and it says which did run"


# ------------------------------------------------------------------ fail-closed

class TestAnUnreadableRunIsNeverScored:

    def test_a_missing_result_event_is_an_error(self, run_provider):
        """Without the terminal event there is no answer to grade and no transcript
        to trust. Returning the raw text would hand the rubric something to score
        with no idea what produced it."""
        out, _seen = run_provider(_stream(_skill_call(), _skill_result()))

        assert "output" not in out
        assert "no result event" in out["error"]
        assert out["metadata"]["stdout_tail"]

    def test_a_non_json_stream_is_an_error(self, run_provider):
        out, _seen = run_provider("not json at all\nnor this\n")

        assert "output" not in out
        assert "no result event" in out["error"]

    def test_one_malformed_line_does_not_discard_the_rest(self, run_provider):
        """A transcript is a stream; one bad entry says nothing about the others.
        The run is judged by what it contains, not by whether every line parsed."""
        transcript = (
            json.dumps(_skill_call()) + "\n"
            + "{ this line is broken\n"
            + json.dumps(_skill_result()) + "\n"
            + json.dumps(_result("survived")) + "\n"
        )

        out, _seen = run_provider(transcript)

        assert out["output"] == "survived"
        assert out["metadata"]["skill_state"] == "ran"

    def test_a_non_zero_exit_still_reports_the_exit(self, run_provider):
        out, _seen = run_provider("", returncode=2, stderr="unknown flag")

        assert "claude exited 2" in out["error"]
