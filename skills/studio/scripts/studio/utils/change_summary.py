"""Change-summary core — resolve the window a digest covers, and select the
decision-log events recorded inside it.

The digest answers "what changed on this branch, and why". This module owns the two
halves that have no output format: **which span of work counts as "the run"**, and
**which recorded decisions fall inside it**. Rendering belongs to the command
wrapper; linking changed files to requirements is separate again.

Three deliberate choices:

* **The window comes from git, not from a decision-log ``run_id``.** A ``run_id`` is
  one CLI invocation, but a reviewer's "run" is a branch's worth of work. The window
  is the span since the merge-base with the default branch, so ``run_id`` becomes a
  grouping key *inside* that span rather than the span itself.
* **Nothing here raises, and nothing here is silent.** Every path returns a value
  carrying an explicit ``reason`` when a dimension is unavailable. A digest that
  quietly shows less is the defect this effort exists to remove, so "cannot tell" is
  always reported rather than rounded down to "nothing to say".
* **Reason strings carry no filesystem paths**, so no ``$HOME`` or username can
  reach a rendered digest through them.

Git access is a narrow read-only query helper, not a general runner. The two existing
private ``_run_git`` helpers in this package have incompatible contracts — one returns
``(code, stdout, stderr)``, the other returns a string and raises — so a third generic
copy would duplicate both. ``_git_line`` answers only "one line of stdout, or nothing".

@cpt-algo:cpt-studio-algo-developer-experience-change-summary:p1
"""

# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-datamodel
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import decision_log

logger = logging.getLogger(__name__)

#: Seconds any single git query may take before it is treated as unavailable.
_GIT_TIMEOUT = 10

#: Refs tried in order when the caller names no base.
#:
#: ``upstream/*`` comes first deliberately. In a fork-based workflow — which this
#: project mandates, with ``origin`` pointing at the contributor's fork — ``origin/HEAD``
#: tracks the *fork's* default branch, which lags the canonical one. Measured on this
#: checkout, ``origin/HEAD`` was five weeks behind ``upstream/main``, so preferring it
#: would silently widen every window to include work that shipped long ago.
#: A fresh clone of the canonical repo has no ``upstream`` remote, so it falls through
#: to ``origin/HEAD`` and is still correct.
#: ``upstream/HEAD`` leads because it is the canonical remote's *own* symbolic default
#: — right even when that default is neither ``main`` nor ``master``. Guessing branch
#: names first would skip it and fall through to a stale fork ref, which is the same
#: failure this ordering exists to prevent, one level deeper.
_DEFAULT_BASE_REFS = (
    "upstream/HEAD",
    "upstream/main",
    "upstream/master",
    "origin/HEAD",
    "origin/main",
    "main",
    "origin/master",
    "master",
)

# Reasons are module constants so the renderer and the tests share one vocabulary
# instead of matching on prose that can drift.
REASON_OK = ""
REASON_NOT_A_REPO = "not a git repository"
REASON_GIT_UNAVAILABLE = "git unavailable"
#: Kept free of an enumerated candidate list on purpose: the first version of this
#: string named the refs it tried, and went stale the moment the list changed.
REASON_NO_BASE_REF = "no default base ref found"
REASON_BASE_REF_UNKNOWN = "requested base ref not found"
REASON_NO_MERGE_BASE = "no merge base with the base ref"
REASON_NO_BASE_TIME = "base commit has no readable timestamp"
REASON_NOT_A_PROJECT = "not inside a Studio project"
REASON_LOG_DISABLED = "decision log disabled"
REASON_LOG_ABSENT = "no decision log yet"
REASON_LOG_UNREADABLE = "decision log unreadable"

#: Bucket name for selected events carrying no ``run_id``. Grouping used to build
#: buckets only for truthy ids, so such events sat in ``events`` and in no group and a
#: renderer under-reported them. Run ids are hex, so this cannot collide with a real one.
RUN_UNATTRIBUTED = "(unattributed)"


@dataclass
class ChangeWindow:
    """The span of work a digest covers.

    ``available`` false means no git-derived window could be established; ``reason``
    then says which of the failure modes applied. ``since`` is the base commit's own
    commit time, which is what makes the window "everything after the branch point".
    """

    project_root: str = ""
    base_ref: str = ""
    base_sha: str = ""
    since: str = ""
    available: bool = False
    reason: str = REASON_NOT_A_REPO


@dataclass
class EventSelection:
    """Decision-log events falling inside a window.

    ``skipped_lines`` is derived, not observed: :func:`decision_log.read_events` drops
    unparseable lines without reporting a count, so this compares the log's non-empty
    line count against the events actually returned. It is therefore a lower bound on
    corruption, and it is reported rather than hidden.
    """

    events: List[Dict[str, Any]] = field(default_factory=list)
    runs: List[str] = field(default_factory=list)
    scanned: int = 0
    undated: int = 0
    runless: int = 0
    skipped_lines: int = 0
    available: bool = False
    reason: str = REASON_NOT_A_PROJECT
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-datamodel


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-git-query
def _git_query(project_root: Path, args: List[str]) -> Tuple[Optional[str], bool]:
    """Run a read-only git query, returning ``(first line or None, tool_failed)``.

    The two halves of "no answer" are kept apart, because conflating them lets a
    transient tool failure be reported as a conclusion about history — "no merge base"
    when git simply timed out.

    * **Tool failure** is git not launching, or timing out. Nothing was learned.
    * **A non-zero exit is a valid negative**, not a failure: ``merge-base`` exits 1
      when two histories genuinely have no common ancestor, and
      ``rev-parse --verify --quiet`` exits 1 when a ref genuinely does not exist. Those
      are answers, and treating them as breakage would be just as misleading in the
      other direction.

    Never raises.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("change-summary git query could not run: %s", exc)
        return None, True
    if result.returncode:
        logger.debug("change-summary git query exited %d", result.returncode)
        return None, False
    line = result.stdout.strip().splitlines()
    return (line[0].strip() if line else None), False


def _git_line(project_root: Path, args: List[str]) -> Optional[str]:
    """First output line of a read-only git query, or ``None`` for any non-answer.

    For callers that only need the value; use :func:`_git_query` where a tool failure
    must be told apart from a valid negative.
    """
    value, _failed = _git_query(project_root, args)
    return value
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-git-query


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-detect-repo
def _is_git_repo(project_root: Path) -> bool:
    """Report whether ``project_root`` sits inside a git work tree."""
    return _git_line(project_root, ["rev-parse", "--is-inside-work-tree"]) == "true"
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-detect-repo


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-default-base
def _resolve_base_ref(project_root: Path, requested: str = "") -> Optional[str]:
    """Pick the ref the window is measured from.

    An explicitly requested ref is honoured or refused — never silently swapped for a
    fallback, because a digest measured against a different ref than the caller asked
    for is worse than one that says it could not comply.
    """
    if requested:
        resolved = _git_line(project_root, ["rev-parse", "--verify", "--quiet", requested])
        return requested if resolved else None
    for candidate in _DEFAULT_BASE_REFS:
        if _git_line(project_root, ["rev-parse", "--verify", "--quiet", candidate]):
            return candidate
    return None
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-default-base


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-merge-base
def _merge_base(project_root: Path, base_ref: str) -> Tuple[Optional[str], bool]:
    """Return the merge-base sha between ``HEAD`` and ``base_ref``.

    Returns ``(sha, tool_failed)``. Unrelated histories and a missing ref yield a
    ``None`` sha with ``tool_failed`` false — there is genuinely no branch point. A
    ``True`` flag means git never answered, which is a different fact and must not be
    reported as a finding about history.
    """
    return _git_query(project_root, ["merge-base", "HEAD", base_ref])
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-merge-base


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-base-time
def _commit_time(project_root: Path, sha: str) -> Tuple[Optional[str], bool]:
    """Commit time in strict ISO 8601, plus whether git itself failed."""
    return _git_query(project_root, ["show", "-s", "--format=%cI", sha])
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-base-time


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-resolve-window
def resolve_window(
    project_root: Path,
    *,
    base: str = "",
    since: str = "",
) -> ChangeWindow:
    """Resolve the span of work a digest should cover.

    ``since`` short-circuits git entirely — an explicit lower bound is the caller's
    assertion and needs no branch point. Otherwise the window starts at the merge-base
    with ``base`` (or the first of :data:`_DEFAULT_BASE_REFS` that exists).

    Every failure returns an unavailable window carrying its reason. Never raises.
    """
    root = str(project_root)
    if since:
        return ChangeWindow(project_root=root, since=since, available=True, reason=REASON_OK)

    if not _is_git_repo(project_root):
        reason = REASON_NOT_A_REPO if _git_line(project_root, ["--version"]) else REASON_GIT_UNAVAILABLE
        return ChangeWindow(project_root=root, reason=reason)

    base_ref = _resolve_base_ref(project_root, base)
    if base_ref is None:
        # Two different failures, two different reasons: a ref the caller named and
        # git does not have, versus no discoverable default at all.
        return ChangeWindow(
            project_root=root,
            reason=REASON_BASE_REF_UNKNOWN if base else REASON_NO_BASE_REF,
        )
    return _window_from_base_ref(project_root, base_ref)
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-resolve-window


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-window-from-base
def _window_from_base_ref(project_root: Path, base_ref: str) -> ChangeWindow:
    """Walk a known-good base ref down to a window, or the reason it could not.

    Split out of :func:`resolve_window` to keep each function's guard clauses legible;
    the whole point of this stage is that there are several distinct ways to fail and
    each gets its own reported reason rather than a shared shrug.

    Past this point git has already answered once, so a further non-answer is
    ambiguous: it may be a genuine negative about history, or the tool falling over.
    Reporting "no merge base" for a timeout would be a false conclusion, so the failure
    flag takes precedence over the historical reading. Whatever *was* learned — the ref,
    then the sha — stays on the returned window so a reason keeps its subject.
    """
    root = str(project_root)
    base_sha, failed = _merge_base(project_root, base_ref)
    if failed:
        return ChangeWindow(project_root=root, base_ref=base_ref, reason=REASON_GIT_UNAVAILABLE)
    if base_sha is None:
        return ChangeWindow(project_root=root, base_ref=base_ref, reason=REASON_NO_MERGE_BASE)

    base_time, failed = _commit_time(project_root, base_sha)
    if failed or base_time is None:
        return ChangeWindow(
            project_root=root, base_ref=base_ref, base_sha=base_sha,
            reason=REASON_GIT_UNAVAILABLE if failed else REASON_NO_BASE_TIME,
        )

    return ChangeWindow(
        project_root=root,
        base_ref=base_ref,
        base_sha=base_sha,
        since=base_time,
        available=True,
        reason=REASON_OK,
    )
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-window-from-base


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-parse-ts
def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp to an aware ``datetime``, or ``None``.

    A trailing ``Z`` is normalised because git and the log writer disagree about it.
    A naive timestamp is refused rather than assumed to be UTC: guessing an offset
    would silently move events across the window boundary.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        logger.debug("change-summary could not parse timestamp %r: %s", value, exc)
        return None
    return parsed if parsed.tzinfo is not None else None
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-parse-ts


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-log-state
def _log_unavailable(path: Path) -> str:
    """Return the reason the decision log cannot be read, or :data:`REASON_OK`.

    Absent and unreadable are reported separately, and readability is proved by
    opening the file rather than inferred from :meth:`Path.is_file`. ``is_file`` only
    needs ``stat``, so a mode-000 log passes it — and
    :func:`decision_log.read_events` swallows the subsequent open failure and yields
    nothing. Together those turned an unreadable log into an *available, empty*
    selection: "no decisions in this window" when in truth nothing was read. That is
    the exact failure this module exists to prevent, so the probe is explicit.
    """
    if not decision_log.is_enabled():
        return REASON_LOG_DISABLED
    try:
        exists = path.is_file()
    except OSError as exc:
        logger.debug("change-summary log probe failed: %s", exc)
        return REASON_LOG_UNREADABLE
    if not exists:
        return REASON_LOG_ABSENT
    try:
        # The bytes are *decoded*, not merely opened. Opening alone proved only that
        # the descriptor could be acquired; `read_events` then performed the first
        # strict UTF-8 read and catches only OSError, so an invalid byte sequence
        # raised UnicodeDecodeError straight out of `select_events` and broke the
        # never-raises contract. Decoding here answers the question the probe claims to.
        with path.open("r", encoding="utf-8") as handle:
            handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("change-summary log is unreadable: %s", exc)
        return REASON_LOG_UNREADABLE
    return REASON_OK
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-log-state


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-count-lines
def _count_log_lines(path: Path) -> int:
    """Count non-empty lines in the log, for deriving how many failed to parse.

    Returns 0 on any read error: an unreadable log is already reported through the
    availability reason, and a wrong skip count must not be invented on top of it.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        logger.debug("change-summary log line count failed: %s", exc)
        return 0
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-count-lines


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-default-log
def _default_log_for(window: ChangeWindow) -> Optional[Path]:
    """Resolve the decision log belonging to *the window's* project.

    ``decision_log.default_log_path()`` defaults to the cwd, which is correct for the
    writer — it logs whichever project the command runs in. A reader reporting on an
    explicitly named project must not inherit that default, or the digest describes one
    project's changes alongside another project's decisions.
    """
    if not window.project_root:
        return decision_log.default_log_path()
    return decision_log.default_log_path(Path(window.project_root))
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-default-log


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-select-events
def select_events(
    window: ChangeWindow,
    *,
    path: Optional[Path] = None,
) -> EventSelection:
    """Select the decision-log events recorded inside ``window``.

    An event whose timestamp cannot be parsed is **excluded and counted** in
    ``undated`` rather than guessed into or out of the window — the caller can then
    say so instead of presenting a quietly incomplete list.

    An unavailable window yields an unavailable selection carrying the window's own
    reason, so the caller reports one cause rather than two.

    The default log is resolved from **the window's own project**, not from the current
    working directory. Those are independent inputs, so a window built for project A
    while the process sits in project B used to select B's decisions — a digest about
    one project carrying another's history. Never raises.
    """
    if not window.available:
        return EventSelection(reason=window.reason)

    target = path or _default_log_for(window)
    if target is None:
        return EventSelection(reason=REASON_NOT_A_PROJECT)
    reason = _log_unavailable(target)
    if reason:
        return EventSelection(reason=reason)

    boundary = _parse_ts(window.since)
    if boundary is None:
        return EventSelection(reason=REASON_NO_BASE_TIME)

    selected, runs, scanned, undated, runless = [], [], 0, 0, 0
    try:
        for event in decision_log.read_events(target):
            scanned += 1
            stamp = _parse_ts(event.get("ts"))
            if stamp is None:
                undated += 1
                continue
            if stamp < boundary:
                continue
            selected.append(event)
            run_id = str(event.get("run_id") or "")
            if not run_id:
                runless += 1
                run_id = RUN_UNATTRIBUTED
            if run_id not in runs:
                runs.append(run_id)
    except (OSError, UnicodeDecodeError) as exc:
        # Belt and braces: the probe already decoded the file, but it could change
        # between probe and read. A read that dies mid-way must not surface as a
        # partial selection presented as complete.
        logger.debug("change-summary log became unreadable while reading: %s", exc)
        return EventSelection(reason=REASON_LOG_UNREADABLE)

    return EventSelection(
        events=selected,
        runs=runs,
        scanned=scanned,
        undated=undated,
        runless=runless,
        skipped_lines=max(0, _count_log_lines(target) - scanned),
        available=True,
        reason=REASON_OK,
    )
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-select-events


# @cpt-begin:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-group-runs
def group_by_run(selection: EventSelection) -> Dict[str, List[Dict[str, Any]]]:
    """Group a selection's events by ``run_id``, preserving first-seen run order.

    This is the role ``run_id`` keeps once the window stops being derived from it: a
    subdivision *within* the branch's span, so a digest can say "three invocations"
    without treating the last one as the whole story.

    **Every selected event lands in exactly one bucket.** Events carrying a blank or
    missing ``run_id`` go to :data:`RUN_UNATTRIBUTED`; previously buckets were built
    only for truthy ids, so such an event sat in ``events`` and in no group at all and
    a renderer summing the groups under-reported without saying so.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {run: [] for run in selection.runs}
    for event in selection.events:
        run_id = str(event.get("run_id") or "") or RUN_UNATTRIBUTED
        grouped.setdefault(run_id, []).append(event)
    return grouped
# @cpt-end:cpt-studio-algo-developer-experience-change-summary:p1:inst-change-summary-group-runs
