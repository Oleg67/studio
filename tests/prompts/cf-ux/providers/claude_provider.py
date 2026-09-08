"""promptfoo native python provider — Claude Code CLI in a cf-studio sandbox."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from _sandbox import SandboxError, sandbox

CLAUDE_BIN = "claude"
CALL_TIMEOUT_S = 850  # under promptfoo worker timeout (900s)

# Cheap-by-default model + low reasoning. Override via env if a scenario
# legitimately needs a stronger model.
#
# Note on 1M context: the 1M-token window is a beta enabled via
# `--betas context-1m-2025-08-07` and only on Opus/Sonnet. Haiku 4.5 has the
# standard 200K window — by selecting Haiku we implicitly opt out of 1M, and
# we never pass --betas here.
DEFAULT_MODEL = os.environ.get("CF_UX_CLAUDE_MODEL", "claude-haiku-4-5")
DEFAULT_EFFORT = os.environ.get("CF_UX_CLAUDE_EFFORT", "low")

# Skill *execution* asks for permission, and `-p` has nobody to ask, so the
# request is denied, the `cf` skill never runs, and the agent answers directly
# instead — returning something plausible that the shared rubric then scores as
# though Studio had behaved well. The codex provider has always passed the
# equivalent pair (`--sandbox workspace-write`, `approval_policy="never"`); this
# is the same decision for this CLI, and the asymmetry was the bug.
#
# Scoped to a throwaway tree: `_sandbox.sandbox()` builds a fresh directory
# under the system temp dir and wipes it in `finally`, on `atexit`, and on
# SIGTERM/SIGINT/SIGHUP. The one exception is `CF_UX_SHARED_SANDBOX`, which
# points a run at a directory the caller chose — noted in the README, because
# there the agent writes where it is told to.
PERMISSION_MODE = "bypassPermissions"

#: The tool Claude Code reports when it executes a skill.
_SKILL_TOOL = "Skill"
#: The skill this suite exists to measure. Matched loosely against the tool
#: input, since only `cf-*` skills are installed in the sandbox and the input's
#: key for the skill name is not part of any stable contract.
_SKILL_NAME = "cf"
#: What a *failed* execution looks like in the transcript: `<error>Execute
#: skill: cf</error>`. The previous guard searched for "skills failed to load",
#: which the CLI never emits, so it could not fire.
_SKILL_ERROR_MARK = "Execute skill:"


def _stream_events(raw: str) -> list[dict[str, Any]]:
    """Parse `--output-format stream-json`: one JSON object per line.

    A line that will not parse is skipped rather than fatal — the stream is a
    transcript, and one malformed entry says nothing about the rest. Whether the
    run is usable at all is decided by :func:`_result_event`, which fails closed.
    """
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _result_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The terminal `result` event, which carries the answer and the totals."""
    for event in reversed(events):
        if event.get("type") == "result":
            return event
    return None


def _content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _skill_state(events: list[dict[str, Any]], raw: str) -> tuple[str, list[str], str]:
    """Did the `cf` skill actually run? Returns ``(state, names, detail)``.

    ``state`` is ``"ran"``, ``"failed"`` or ``"absent"``. This is a *positive*
    check against tool-use events in the transcript, not a substring search over
    the prose: the fallback answer is well-formed and says nothing about whether
    the skill was reached, which is exactly why the old heuristic scored it.
    """
    calls: dict[Any, Any] = {}
    errored: set[Any] = set()
    for event in events:
        for block in _content_blocks(event):
            kind = block.get("type")
            if kind == "tool_use" and block.get("name") == _SKILL_TOOL:
                calls[block.get("id")] = block.get("input") or {}
            elif kind == "tool_result" and block.get("is_error"):
                errored.add(block.get("tool_use_id"))

    names = sorted({json.dumps(payload, default=str, sort_keys=True) for payload in calls.values()})
    if _SKILL_ERROR_MARK in raw:
        return "failed", names, f"the transcript carries {_SKILL_ERROR_MARK!r}"
    if not calls:
        return "absent", names, f"no {_SKILL_TOOL} tool call in the transcript"
    if all(call_id in errored for call_id in calls):
        return "failed", names, f"every {_SKILL_TOOL} call came back as an error"
    if not any(_SKILL_NAME in payload for payload in names):
        return "failed", names, f"a {_SKILL_TOOL} ran but none of them named {_SKILL_NAME!r}"
    return "ran", names, ""


def call_api(prompt: str, options: dict | None = None, context: dict | None = None) -> dict:
    started = time.monotonic()
    try:
        with sandbox() as cwd:
            return _invoke(prompt, cwd, started)
    except SandboxError as exc:
        return {"error": f"sandbox setup failed: {exc}"}
    except subprocess.TimeoutExpired:
        return {"error": f"claude timed out after {CALL_TIMEOUT_S}s"}
    except Exception as exc:  # noqa: BLE001 — surface unexpected errors to promptfoo
        return {"error": f"unexpected: {type(exc).__name__}: {exc}"}


def _invoke(prompt: str, cwd: Path, started: float) -> dict:
    # Explicit skill invocation: Claude Code uses `/cf <prompt>`.
    invoked = f"/cf {prompt}"
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", DEFAULT_MODEL,
        "--effort", DEFAULT_EFFORT,
        # Without this the skill is denied and the run scores the fallback path.
        "--permission-mode", PERMISSION_MODE,
        # The transcript, not just the answer: tool-use events are the only place
        # the run says whether the skill was reached. `--verbose` is what makes
        # `-p` emit the intermediate events rather than the result alone.
        "--output-format", "stream-json",
        "--verbose",
        "--max-budget-usd", "0.30",
        invoked,
    ]
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=CALL_TIMEOUT_S, check=False,
        stdin=subprocess.DEVNULL,
    )
    duration = time.monotonic() - started

    if proc.returncode != 0:
        return {
            "error": f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}",
            "metadata": {"duration_s": round(duration, 2)},
        }

    events = _stream_events(proc.stdout)
    payload = _result_event(events)
    base = {"duration_s": round(duration, 2), "sandbox": str(cwd)}
    if payload is None:
        # Fail closed. Without the terminal event there is no answer to grade and
        # no transcript to trust, so returning the raw text would hand the rubric
        # something to score with no idea what produced it.
        return {
            "error": "claude emitted no result event; stream-json may not have been honoured",
            "metadata": {**base, "stdout_tail": proc.stdout.strip()[-500:]},
        }

    output_text = payload.get("result") or ""
    state, skills, detail = _skill_state(events, proc.stdout)
    metadata = {
        **base,
        "session_id": payload.get("session_id"),
        "num_turns": payload.get("num_turns"),
        "total_cost_usd": payload.get("total_cost_usd"),
        "skill_state": state,
        "skills_invoked": skills,
    }
    cost = payload.get("total_cost_usd")
    if state != "ran":
        # A hard error, not a metadata flag. The fallback answer is plausible and
        # well-formed, so left to the grader it scores as a pass and the suite
        # reports on an agent that never loaded Studio. A run that did not engage
        # the skill is not a measurement of the skill, and must not be graded as
        # one — this is the part of the fix that keeps the defect from returning
        # the next time an invocation detail changes.
        result: dict[str, Any] = {
            "error": f"cf skill did not run ({state}): {detail}",
            "metadata": {**metadata, "unscored_output": output_text[:500]},
        }
    else:
        result = {"output": output_text, "metadata": metadata}
    if isinstance(cost, (int, float)):
        result["cost"] = float(cost)
    return result
