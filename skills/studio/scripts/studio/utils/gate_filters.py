"""The three safety filters -- reversibility, scope, blocker -- for the gate chain.

Each is a ``SafetyFilter`` (``gate_chain``): it may only **add** a stop. It returns ``CLEAR`` when
it sees no reason to stop, ``ADD_STOP`` when it requires one, and ``INDETERMINATE`` when it cannot
tell -- which the chain reads as a stop, so every "cannot tell" fails **closed**. None of them
removes a stop; that is the economy class's job, in a later increment.

The order the chain runs them in, and the fact an added stop is final, is the whole safety
argument and lives in ``gate_chain.resolve_gate``; these are the individual judgements it composes.

They are pure of the runtime: each reads only its ``Gate``'s fields (populated by the runtime seam,
a later increment), so they are tested against constructed ``Gate``s. The one exception is that
reversibility asks git through ``armed_reversal`` and scope reads the plan off disk -- both are
read-only, never-raising probes whose failure is treated as ``INDETERMINATE``.

A ``Gate`` field left at its default carries **no information**, and every filter treats "no
information" as a stop, never as permission: an undetermined write set, an unknown checkout, an
unknown plan or active phase, an empty action all return ``INDETERMINATE``.
"""
from __future__ import annotations

import logging
import posixpath
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import FrozenSet, Iterator, List

from .armed_reversal import armed_reversal
from .gate_chain import Gate, SafetyVerdict

logger = logging.getLogger(__name__)


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-normalize
def _normalize(name: str) -> str:
    """One canonical form for comparing a written path against a declared one: stripped,
    forward-slashed, and with ``.``/``..`` segments and duplicate slashes collapsed. An absolute
    path keeps its leading slash, so it can never equal a declared relative one. Case is
    **preserved** -- two paths differing only in case are two files on the platforms this must be
    safe on, and folding them would clear a write the plan did not declare."""
    text = name.strip().replace("\\", "/")
    if not text:
        return ""
    return posixpath.normpath(text)  # collapses `.`, `..` and duplicate slashes, keeps case
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-normalize


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-reversibility
class ReversibilityFilter:  # pylint: disable=too-few-public-methods
    """Stop an autonomous edit unless a reversal mechanism is armed for the checkout.

    Asserts a fact about the environment -- is there a way back? -- rather than judging whether
    the action *looks* undoable; that distinction, and the probe, belong to ``armed_reversal``.
    An undetermined write set stops (fail closed); an option known to write nothing clears; a
    probe that cannot run is read as ``INDETERMINATE``, never as permission.
    """

    def check(self, gate: Gate) -> SafetyVerdict:
        """``CLEAR`` when a reversal is armed (or nothing is written); a stop otherwise."""
        if gate.target_files is None:
            return SafetyVerdict.INDETERMINATE   # write set undetermined -> fail closed
        if not gate.target_files:
            return SafetyVerdict.CLEAR           # known to write nothing
        if gate.project_root is None:
            return SafetyVerdict.INDETERMINATE
        try:
            result = armed_reversal(gate.project_root)
        except Exception as exc:  # pylint: disable=broad-except
            # A probe that raises must not open the gate: unknown is refusal here, and a guard
            # that stops running without saying so is worth a line.
            logger.warning("reversibility filter: the reversal probe raised, so the gate is "
                           "held rather than cleared: %s", type(exc).__name__)
            return SafetyVerdict.INDETERMINATE
        if result.refused:  # the consequence reading the caller acts on, not the bare fact
            return SafetyVerdict.ADD_STOP
        logger.debug("reversibility filter: cleared, a reversal is armed via %s", result.mechanism)
        return SafetyVerdict.CLEAR
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-reversibility


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-read-declared
#: The frontmatter field the phase-file template (``requirements/plan-template.md``) uses for the
#: **created or modified project files** a phase declares -- as opposed to ``outputs``, which the
#: template reserves for intermediate files under ``out/``. Scope is about the former.
_OUTPUT_FILES_KEY = "output_files"


def read_declared_outputs(plan_dir: Path, up_to_phase: int) -> FrozenSet[str]:
    """The union of ``output_files`` declared by phases numbered ``<= up_to_phase``, normalized.

    Only the active phase and the ones before it -- the approved-so-far surface -- count; a file
    only a *later* phase declares is **not** in scope for an option running now, which closes the
    "write any file any future phase will ever touch" hole. Reads ``plan.toml``'s ``[[phases]]``,
    then each in-range phase file's fenced ``[phase]`` frontmatter. **Never raises** -- an
    unreadable or malformed plan yields the empty set, which scope treats as "cannot confirm" and
    fails closed on. Empty is "the plan told us nothing", never "the plan permits everything".
    """
    plan = _load_toml(plan_dir / "plan.toml")
    outputs: set = set()
    phases = plan.get("phases", []) if isinstance(plan, dict) else []
    if not isinstance(phases, list):
        # A `phases` that is a scalar (a corrupt or hand-edited plan.toml) is not iterable;
        # returning nothing keeps the documented non-raising contract, and an empty declared
        # set fails scope closed rather than raising a TypeError into the safety path.
        return frozenset()
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        number = phase.get("number")
        # A phase we cannot place is fail-closed OUT of the approved-so-far set: excluding its
        # outputs means a write to them is stopped, never cleared. A missing, non-int, or bool
        # number (bool is an int subclass) cannot be confirmed <= up_to_phase, so it is skipped
        # -- otherwise a later or unplaceable phase's outputs would broaden the approved set and
        # defeat the future-phase exclusion this reader guarantees. `number < 1` is the lower
        # half of the same guard: phase numbers are 1-based, so a 0 or negative value is not a
        # legitimate phase, and without this it would satisfy `<= up_to_phase` for every real
        # active phase and fail OPEN -- unioning a corrupt phase's outputs into the approved set.
        if (not isinstance(number, int) or isinstance(number, bool)
                or number < 1 or number > up_to_phase):
            continue
        name = phase.get("file")
        if not isinstance(name, str) or not _is_safe_relative(name):
            continue
        outputs.update(_declared_outputs_of(_phase_frontmatter(plan_dir / name)))
    return frozenset(outputs)
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-read-declared


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-scope
class ScopeFilter:  # pylint: disable=too-few-public-methods
    """Stop if the option writes a project file the approved plan did not declare it would.

    Checks the option's ``target_files`` against the outputs declared by the active phase and the
    ones before it. An undetermined write set, an unknown plan or an unknown active phase all stop
    (fail closed); an option known to write nothing clears; a plan that declares no outputs in
    range is read as ``INDETERMINATE`` -- "no declaration" is not "declared to write anywhere".
    """

    def check(self, gate: Gate) -> SafetyVerdict:
        """``CLEAR`` when every written file is declared in range; a stop otherwise."""
        if gate.target_files is None:
            return SafetyVerdict.INDETERMINATE
        if not gate.target_files:
            return SafetyVerdict.CLEAR
        if gate.plan_dir is None or gate.active_phase is None:
            return SafetyVerdict.INDETERMINATE
        declared = read_declared_outputs(gate.plan_dir, gate.active_phase)
        if not declared:
            return SafetyVerdict.INDETERMINATE
        outside = [name for name in gate.target_files if _normalize(name) not in declared]
        return SafetyVerdict.ADD_STOP if outside else SafetyVerdict.CLEAR
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-scope


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-blocker
#: The never-auto-answer categories, each a pattern over an option's action text. Derived from the
#: eleven ``NEVER`` invariant lines at ``brave-new-world-eligibility.md`` -- referenced, not
#: copied: a per-operation test asserts a realistic, *inflected* phrasing of each forbidden
#: operation matches here, so the two cannot drift into a filter that clears something the
#: invariants forbid. Verbs use a **prefix** boundary (``\bremov``, not ``\bremove\b``) so
#: "removes/removing/removal" are caught, not only the bare word. Matching an option's visible
#: action is a proxy, so this is a floor, not a ceiling: the same gates are typed ``blocking`` and
#: caught by the chain's ceiling too. Where a forbidden word is also a common benign one
#: (``update``, ``security``, ``account``, ``merge``, ``replace``), the pattern targets the
#: dangerous sense to keep the floor from stopping ordinary options. Case-insensitive.
_BLOCKED_PATTERNS = (
    # destructive / deletion / overwrite / truncate / drop / artifact replacement (lines 33, 38)
    r"\bdelet|overwrit|\bremov|\brm\b|\bdrop\b|truncat|destroy|irreversib|regenerat"
    r"|replace.{0,25}(generated|artifact|snapshot|output)|artifact.{0,25}replac",
    # session / history / forget / unload / disable / weaken the invariants (lines 40, 43)
    r"shutdown|shut down|\bforget|\bunload|\bdisabl|history[- ]?rewrit|rewrite history"
    r"|\bweaken|STOP_TURN|dispatch gate|terminal output shape",
    # git mutation (line 37)
    r"\bcommit|\bpush|force[- ]?push|git add|staging|\bcheckout|branch delet|reset --hard"
    r"|\brebase|\brevert|git merge|\bmerge (branch|into|main|onto)|git mode",
    # secrets / permissions / install / update / setup / security / escalation (line 35)
    r"credential|secret|token|password|authenticat|authoriz|privacy|payment|billing"
    r"|sandbox|escalat|filesystem.{0,3}permission|chmod|\bsudo|\binstall"
    r"|dependency (upgrad|updat)|update.{0,15}(dependenc|packag|runtime|system|engine)"
    r"|\bsetup\b|security.{0,3}(impact|gate|escalat|sensitive)",
    # external / deploy / publish / network / account / cross-repo (lines 34, 36)
    r"\bexternal|cross[- ]?repo|network[- ]?access|\bdeploy|publication|\bpublish"
    r"|account.{0,3}(change|owner|delet|modif|plan)",
    # broad / unknown blast radius / migration (lines 38, 39)
    r"migrat|mechanical rewrit|blast radius|broad.*rewrit",
    # debugger (line 34)
    r"debugger|breakpoint|step/continue|debug[- ]?gate|cf-debug",
    # review-fix (line 39)
    r"ReviewFindingsNavigation|ReviewFixScope|review[- ]?fix",
)
_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)


class BlockerFilter:  # pylint: disable=too-few-public-methods
    """Stop an option whose action names a never-auto-answer operation.

    The list of such operations is PDSL prose (``brave-new-world-eligibility.md``); this matches
    the option's visible action against patterns derived from it, kept honest by the test that
    asserts every forbidden operation is covered. An empty action is read as ``INDETERMINATE``
    (fail closed) -- an unreadable action is exactly the case where clearing would be dangerous.
    """

    def check(self, gate: Gate) -> SafetyVerdict:
        """``ADD_STOP`` when the action matches a blocked category; ``CLEAR`` when it is clean."""
        if not gate.option_action.strip():
            return SafetyVerdict.INDETERMINATE
        if _BLOCKED_RE.search(gate.option_action):
            return SafetyVerdict.ADD_STOP
        return SafetyVerdict.CLEAR
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-filter-blocker


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-readers
def _is_safe_relative(name: str) -> bool:
    """Whether a phase ``file=`` entry stays under the plan directory: a plain relative path with
    no absolute root, drive letter or ``..`` escape. A plan that points the reader at ``/etc`` or
    ``../../x`` is ignored, not followed -- read-only or not, the reader answers about the plan."""
    if not name or name.startswith(("/", "\\")) or ":" in name:
        return False
    return ".." not in PurePosixPath(name.replace("\\", "/")).parts


def _declared_outputs_of(front: dict) -> List[str]:
    """The normalized ``output_files`` a phase declares. A lone string -- a common authoring slip
    for a one-file phase -- is accepted as a single entry rather than silently dropped; anything
    that is neither a list nor a string is warned about, not swallowed."""
    value = front.get(_OUTPUT_FILES_KEY)
    if isinstance(value, str):
        value = [value]
    elif value is not None and not isinstance(value, (list, tuple)):
        logger.warning("scope filter: a phase's output_files is a %s, not a list of paths, so "
                       "it declares nothing", type(value).__name__)
        return []
    if not isinstance(value, (list, tuple)):
        return []
    return [_normalize(item) for item in value if isinstance(item, str) and item.strip()]
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-readers


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-toml
def _load_toml(path: Path) -> dict:
    """Parse a TOML file, bounded and non-raising to the caller (`{}` on any failure)."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, ValueError) as exc:  # TOMLDecodeError is a ValueError subclass
        logger.warning("scope filter: %s could not be read as TOML, so it declares nothing: %s",
                       path.name, type(exc).__name__)
        return {}
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-toml


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-fences
def _fenced_bodies(text: str) -> Iterator[str]:
    """Yield the body of each ```` ``` ```` fenced block in order, closing only on a proper
    CommonMark closing-fence **line** -- its own line, backticks only, at least as long as the
    opener.

    Closing on the next ``` **substring** anywhere (the earlier scan) truncated a body early on a
    ``` inside a value or a shorter nested fence -- the very case the frontmatter scan cares about
    (an example or ``pdsl`` fence ahead of the real ``[phase]`` block) -- and desynchronised the
    rest of the scan. Line-based, so still one linear pass and no backtracking regex over
    author-controlled input (S8786). The opener's info string (```` ```toml ````) is ignored; a
    closer must carry nothing but its backticks.
    """
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        head = lines[i].lstrip(" ")
        opener = len(head) - len(head.lstrip("`"))
        if opener < 3:  # not a fence-opening line
            i += 1
            continue
        body: List[str] = []
        i += 1
        while i < n:
            close = lines[i].lstrip(" ")
            run = len(close) - len(close.lstrip("`"))
            if run >= opener and not close[run:].strip():  # a bare closing fence, long enough
                break
            body.append(lines[i])
            i += 1
        yield "\n".join(body)
        i += 1  # step past the closing fence line (or past end-of-text)


def _phase_frontmatter(path: Path) -> dict:
    """The ``[phase]`` table from the **first fenced block that actually declares one**, or ``{}``.

    Scanning until a ``[phase]`` table appears -- rather than blindly taking the first fence --
    means an example block or a ``pdsl`` fence ahead of the real frontmatter (the template itself
    shows example ``[phase]`` blocks) does not mis-declare a phase's outputs. Non-raising: a
    missing file, no such fence, or malformed TOML all yield ``{}``.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except (OSError, ValueError) as exc:
        # ValueError as well as OSError: a plan `file=` with an embedded NUL passes the lexical
        # `_is_safe_relative` check but makes `read_text` raise ValueError ("embedded null byte"),
        # not OSError -- caught here so the documented non-raising contract holds, matching the
        # sibling `_load_toml`'s (OSError, ValueError) handling.
        logger.warning("scope filter: phase file %s could not be read: %s",
                       path.name, type(exc).__name__)
        return {}
    for body in _fenced_bodies(text):
        try:
            data = tomllib.loads(body)
        except ValueError:  # TOMLDecodeError; a non-TOML fence (e.g. a ```pdsl block) -- try next
            logger.debug("scope filter: a fence in %s is not TOML, trying the next", path.name)
            continue
        phase = data.get("phase")
        if isinstance(phase, dict):
            return phase
    return {}
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-scope-fences
