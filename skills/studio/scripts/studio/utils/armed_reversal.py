"""Assert that a reversal mechanism is armed before an autonomous edit, or refuse.

**Assert, never judge.** This takes no description of the action it guards, and it cannot:
nothing about the action is in scope. It answers one question about the *environment* --
if this goes wrong, is there something that puts it back? -- rather than the question the
shipped eligibility modules answer, which is whether an action *looks* reversible.

That distinction is the point. Judging asks "does this seem safe?" and is a guess about an
action; asserting asks "is there a way back?" and is a fact about the checkout. The task
this implements rules the first out in those words: the filter must assert that a reversal
mechanism is armed, never judge whether an action looks undoable.

Three mechanisms, weakest last, first answer wins:

``worktree``
    the edit happens in a linked git worktree, so discarding it costs nothing.
``branch``
    a named branch is checked out, so the edit can be committed and reset.
``snapshot``
    per-file copies taken before the edit. **Not implemented** -- it is the mechanism for
    a project that is not a git repository at all, which is the case that prompted this,
    and it is deliberately out of scope here. Until it exists a non-git project gets a
    refusal, which is the safe direction and a real gap rather than a hidden one.

**Unknown is refusal, never permission.** A git command that fails teaches nothing, and
the one direction this must never fail is open.

This reports; it does not gate. The caller decides what a refusal means, the same way the
plan lookup answers without dispatching.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .redaction import home_collapsed

logger = logging.getLogger(__name__)

# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-vocab
#: The mechanisms, in the order they are tried. Not a preference: each is strictly weaker
#: than the one before, so the first that answers is the strongest available. A test pins
#: the order, because "try them in any order" is how the weakest becomes the default.
WORKTREE = "worktree"
BRANCH = "branch"
SNAPSHOT = "snapshot"

#: Every mechanism this knows about, named in the refusal so an operator is told what
#: would have satisfied it rather than only that nothing did. Pinned to literals by a test:
#: the refusal tests iterate this tuple, so shrinking it left them green while checking
#: one mechanism -- a constant a test draws its cases from has to be pinned somewhere, or
#: it is testing itself.
MECHANISMS: Tuple[str, ...] = (WORKTREE, BRANCH, SNAPSHOT)
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-vocab


# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-outcome
@dataclass(frozen=True)
class ReversalCheck:
    """Whether a way back exists, which one, and what to do when none does.

    ``why`` is written for the person who has to act on it. "No reversal is armed" tells
    them nothing; naming the three mechanisms tells them three things, one of which they
    can usually do in ten seconds.
    """

    armed: bool
    mechanism: Optional[str]
    why: str

    @property
    def refused(self) -> bool:
        """The reading the caller acts on. `armed` is the fact; this is the consequence."""
        return not self.armed
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-outcome


# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-git
def _check_borrowed_helper_exists() -> None:
    """Fail loudly at import if the bounded git query this borrows was renamed.

    `_git_query` is private to its module and carries no stability promise, so a rename
    there would leave every probe here reporting "the tool failed" — and this module turns
    that into a refusal. The result is a permanent, silent refusal of every autonomous
    edit, arriving as a behaviour change with no error anywhere.

    The same guard the gate-surface walk uses against a renamed PDSL keyword. That lesson
    was applied there and not here; review found the gap.

    **Both halves raise**, matching that guard exactly. The first version warned and
    returned when its own import failed, which review flagged as two guards against the
    same class of failure disagreeing on how bad it is. Warning is the weaker of the two
    and it was the wrong one: a guard that cannot run, and carries on anyway, leaves the
    thing it guards unguarded and looking fine -- which is the defect it exists to catch,
    one level up. The module it reaches for imports nothing but the standard library and
    its own siblings, so an `ImportError` here is a broken installation rather than a
    missing optional dependency, and a broken installation should stop.
    """
    try:
        from . import change_summary  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "armed reversal: the module carrying the bounded git query could not be "
            "imported, so a rename of `_git_query` would go unnoticed and every reversal "
            "probe would report a tool failure and refuse every edit, silently") from exc
    if not hasattr(change_summary, "_git_query"):
        raise RuntimeError(
            "armed reversal: change_summary._git_query is gone, so every reversal probe "
            "would report a tool failure and refuse every edit, silently")


_check_borrowed_helper_exists()


def _git(project_root: Path, args: List[str]) -> Tuple[Optional[str], bool]:
    """Ask git one read-only question: ``(answer or None, the tool failed)``.

    Reuses the repo's bounded query rather than running a second subprocess of its own.
    A duplicate implementation is how a fix gets lost -- that helper already carries a
    timeout, the never-raises contract, and the filesystem codec a ref needs, and each of
    those was added in response to something.

    It also keeps apart the two halves of "no answer", which matters here even though both
    end in a refusal: "there is no branch" and "git did not run" are different things to
    tell someone, and only one of them is about their project.
    """
    try:
        from .change_summary import _git_query  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        # Loud, and refusing. A missing helper is not evidence that a reversal is armed,
        # and this is the module where guessing in the permissive direction is worst.
        #
        # Broad for that reason. Naming `ImportError` and `AttributeError` covered the
        # failures imagined, and importing a module runs it: a `SyntaxError` after an
        # edit, or anything raised at its import time, went straight through a probe
        # whose whole contract is that unknown is refusal. `_said` below already had to
        # be broadened for the same reason, and an asymmetry inside one file is how the
        # second one gets missed.
        logger.warning("armed reversal: the shared git query is unavailable, so no "
                       "mechanism can be confirmed: %s", type(exc).__name__)
        return None, True
    return _git_query(project_root, args)


def _inside_work_tree(project_root: Path) -> Tuple[bool, bool]:
    """``(inside a git work tree, the tool failed)``.

    Asked first because both git mechanisms rest on it, and because a project that is not
    a repository at all is the case the missing snapshot mechanism exists for -- naming it
    precisely is the difference between "no reversal" and "this is not a git project".
    """
    answer, failed = _git(project_root, ["rev-parse", "--is-inside-work-tree"])
    return answer == "true", failed


def _in_linked_worktree(project_root: Path) -> Tuple[bool, bool]:
    """Whether this checkout is a linked worktree rather than the main one.

    A linked worktree has its own git dir under the common one, so the two paths differ.
    That is the whole test: in the main checkout both answer `.git`.

    Both flags are asked for absolutely, because the bare forms answer relative in the
    main checkout and absolute in a linked one -- comparing *those* reports a main
    checkout as linked from some directories and not others.

    `resolve()` on top of that is belt-and-braces: with both answers absolute I could not
    construct a case where it changes the verdict, including through a symlinked project
    root. It stays because comparing two paths by string is a known footgun and the call
    costs nothing, but it is not carrying a defect I can point at -- said plainly rather
    than dressed up as a fix, since a guard justified by an invented scenario reads as
    evidence the scenario is real.
    """
    git_dir, failed_a = _git(project_root, ["rev-parse", "--absolute-git-dir"])
    common, failed_b = _git(project_root, ["rev-parse", "--path-format=absolute",
                                           "--git-common-dir"])
    if failed_a or failed_b or git_dir is None or common is None:
        return False, True
    return Path(git_dir).resolve() != Path(common).resolve(), False


def _on_named_branch(project_root: Path) -> Tuple[bool, bool]:
    """Whether a named branch is checked out, so an edit can be committed and reset.

    `symbolic-ref --quiet` answers nothing on a detached HEAD and exits non-zero, which is
    a valid negative rather than a tool failure -- the distinction the shared helper keeps.
    """
    answer, failed = _git(project_root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if failed or not answer:
        return False, failed
    # A named branch is not enough: `git init` with no commits reports one, and there is
    # nothing behind it to reset to. Arming on that is the worst failure this module has --
    # a way back that does not exist, reported as one that does. Raised in review as Major.
    head, failed = _git(project_root, ["rev-parse", "--verify", "--quiet", "HEAD"])
    return bool(head) and not failed, failed
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-git


# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-refusal
def _said(value: object) -> str:
    """A path on its way into a refusal, with the home directory collapsed and capped.

    A project root is very often under `$HOME`, so the refusal read
    `/home/<username>/project is not inside a git work tree` and put a username into
    whatever logs it.

    The collapse moved to `redaction.home_collapsed`, and moving it fixed two live leaks
    this copy still had: it trusted a clean return from `capped_text`, which returns its
    input unchanged when `Path.home()` raises inside it, and its own fallback then returned
    the raw path truncated when `Path.home()` raised here too -- under a comment saying that
    second one had been fixed. Both were found in the sibling copy and repaired only there.
    Raised in review as duplication; the duplication is how the repair was lost.
    """
    text = home_collapsed(value, logger=logger, subject="armed reversal")[:200]
    # Stripped and delimited, not merely redacted and capped. A directory named with a
    # newline forged a second line in the refusal reading `ARMED: yes — reversal
    # confirmed`: a fabricated verdict, saying the opposite of the real one, inside the
    # check whose whole job is to refuse. Collapsing whitespace removes the forgery and
    # the quotes show a reader where the path ends, which a bare interpolation cannot.
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    # `json.dumps` for the delimiting, not a hand-rolled pair of quotes. A `"` is printable,
    # survives the strip, and closes a hand-rolled delimiter early -- so the forged text
    # lands *outside* the quotes and reads as part of the sentence. That is the same
    # forgery the whitespace strip was added to stop, walking back in through the fix.
    return json.dumps(" ".join(printable.split()))


def _refusal(reason: str) -> ReversalCheck:
    """Not armed, with what would have satisfied it.

    Every refusal names all three mechanisms. An operator reading "no reversal is armed"
    learns that something stopped and nothing about what to do; one that names a worktree,
    a branch and a snapshot gives them a list to act on, and the first two usually take
    seconds.
    """
    # `snapshot` is named as what it is. Listing it beside two mechanisms a reader can
    # actually arm reads as three options, and sends them looking for a switch that does
    # not exist -- raised in review, and only the non-git refusal said so before.
    offered = ", ".join(m if m != SNAPSHOT else f"{m} (not built)" for m in MECHANISMS)
    return ReversalCheck(
        armed=False, mechanism=None,
        why=(f"{reason}; no reversal is armed, so an autonomous edit is refused rather "
             f"than attempted. Any of these would arm one: {offered}"))
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-refusal


# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-assert
def armed_reversal(project_root: Path) -> ReversalCheck:
    """Whether a way back exists for an edit about to be made under ``project_root``.

    Takes no description of the edit, deliberately and permanently. A parameter carrying
    what the action does is all it would take for this to slide back into judging whether
    the action looks safe, which is the behaviour it replaces.

    Ordered worktree, then branch, then snapshot, and the first that answers wins.
    """
    inside, failed = _inside_work_tree(project_root)
    if failed:
        # Nothing was learned, so nothing is armed. The permissive reading of a failed
        # probe -- "git is not saying no, carry on" -- is the single worst mistake this
        # module could make, because it fires exactly when the environment is unusual.
        return _refusal("git could not be consulted, so no mechanism could be confirmed")
    if not inside:
        # Named precisely rather than folded into a generic refusal: this is the case the
        # snapshot mechanism exists for, and the one a reader of the refusal can do least
        # about today.
        return _refusal(f"{_said(project_root)} is not inside a git work tree, and the "
                        "snapshot "
                        "mechanism that would cover that is not built")

    linked, failed = _in_linked_worktree(project_root)
    if linked and not failed:
        return ReversalCheck(armed=True, mechanism=WORKTREE,
                             why="the edit runs in a linked git worktree, so discarding "
                                 "it costs nothing")

    branched, failed = _on_named_branch(project_root)
    if branched and not failed:
        return ReversalCheck(armed=True, mechanism=BRANCH,
                             why="a named branch is checked out, so the edit can be "
                                 "committed and reset")

    # Inside a repository, but on a detached HEAD with no linked worktree: there is a git
    # directory and still nothing that puts a file back by itself.
    return _refusal("this checkout is not a linked worktree and has no named branch "
                    "checked out")
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-reversal-assert
