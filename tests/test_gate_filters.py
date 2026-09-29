"""The three safety filters add stops only where they should, and fail closed when unsure.

Load-bearing checks: an undetermined write set, an unknown checkout/plan/phase, and an empty
action all STOP (never clear); the blocker matches a realistic phrasing of *every* forbidden
operation (a per-operation table, not one cherry-picked word per line); scope never lets a write
declared only by a future phase through.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.utils.armed_reversal import ReversalCheck
from studio.utils.gate_chain import Gate, SafetyVerdict
from studio.utils import gate_filters
from studio.utils.gate_filters import (
    _BLOCKED_RE,
    _OUTPUT_FILES_KEY,
    BlockerFilter,
    ReversibilityFilter,
    ScopeFilter,
    read_declared_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_INVARIANTS = REPO_ROOT / "skills/studio/modules/brave-new-world-eligibility.md"


# --- reversibility ------------------------------------------------------------------------

def test_reversibility_clears_when_the_option_writes_nothing() -> None:
    assert ReversibilityFilter().check(Gate(target_files=())) is SafetyVerdict.CLEAR


def test_reversibility_fails_closed_when_the_write_set_is_undetermined() -> None:
    # The default Gate() -- what an unpopulated seam produces -- must NOT clear.
    assert ReversibilityFilter().check(Gate()) is SafetyVerdict.INDETERMINATE
    assert ReversibilityFilter().check(Gate(option_action="WRITE x")) is SafetyVerdict.INDETERMINATE


def test_reversibility_fails_closed_without_a_root() -> None:
    out = ReversibilityFilter().check(Gate(target_files=("a.py",), project_root=None))
    assert out is SafetyVerdict.INDETERMINATE


def test_reversibility_clears_when_armed_and_stops_when_not(monkeypatch) -> None:
    monkeypatch.setattr(gate_filters, "armed_reversal",
                        lambda root: ReversalCheck(armed=True, mechanism="branch", why=""))
    armed = ReversibilityFilter().check(Gate(target_files=("a.py",), project_root=Path(".")))
    assert armed is SafetyVerdict.CLEAR

    monkeypatch.setattr(gate_filters, "armed_reversal",
                        lambda root: ReversalCheck(armed=False, mechanism=None, why="no reversal"))
    unarmed = ReversibilityFilter().check(Gate(target_files=("a.py",), project_root=Path(".")))
    assert unarmed is SafetyVerdict.ADD_STOP


def test_reversibility_fails_closed_when_the_probe_raises(monkeypatch) -> None:
    def _boom(root):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(gate_filters, "armed_reversal", _boom)
    out = ReversibilityFilter().check(Gate(target_files=("a.py",), project_root=Path(".")))
    assert out is SafetyVerdict.INDETERMINATE


# --- scope --------------------------------------------------------------------------------

def _write_plan(plan_dir: Path, outputs_by_phase) -> None:
    """A plan.toml + phase files declaring the given output_files per phase (1-indexed)."""
    phases = "\n".join(f'[[phases]]\nnumber = {i+1}\nfile = "phase-{i+1}.md"'
                       for i in range(len(outputs_by_phase)))
    (plan_dir / "plan.toml").write_text(f"[plan]\ntask = \"t\"\n{phases}\n", encoding="utf-8")
    for i, outs in enumerate(outputs_by_phase):
        arr = ", ".join(f'"{o}"' for o in outs)
        body = f"```toml\n[phase]\nnumber = {i+1}\n{_OUTPUT_FILES_KEY} = [{arr}]\n```\n\n# Phase {i+1}\n"
        (plan_dir / f"phase-{i+1}.md").write_text(body, encoding="utf-8")


def test_scope_clears_when_the_option_writes_nothing() -> None:
    assert ScopeFilter().check(Gate(target_files=())) is SafetyVerdict.CLEAR


def test_scope_fails_closed_when_the_write_set_is_undetermined() -> None:
    assert ScopeFilter().check(Gate()) is SafetyVerdict.INDETERMINATE


def test_scope_fails_closed_without_a_plan_or_active_phase(tmp_path: Path) -> None:
    assert ScopeFilter().check(
        Gate(target_files=("a.py",), plan_dir=None, active_phase=1)) is SafetyVerdict.INDETERMINATE
    assert ScopeFilter().check(
        Gate(target_files=("a.py",), plan_dir=tmp_path, active_phase=None)) is SafetyVerdict.INDETERMINATE


def test_scope_clears_an_in_plan_write(tmp_path: Path) -> None:
    _write_plan(tmp_path, [["src/a.py"], ["src/b.py", "README.md"]])
    out = ScopeFilter().check(Gate(target_files=("src/b.py",), plan_dir=tmp_path, active_phase=2))
    assert out is SafetyVerdict.CLEAR


def test_scope_stops_an_out_of_plan_write(tmp_path: Path) -> None:
    _write_plan(tmp_path, [["src/a.py"]])
    out = ScopeFilter().check(Gate(target_files=("src/evil.py",), plan_dir=tmp_path, active_phase=1))
    assert out is SafetyVerdict.ADD_STOP


def test_scope_stops_when_any_target_is_out_of_plan(tmp_path: Path) -> None:
    _write_plan(tmp_path, [["src/a.py"]])
    out = ScopeFilter().check(
        Gate(target_files=("src/a.py", "src/evil.py"), plan_dir=tmp_path, active_phase=1))
    assert out is SafetyVerdict.ADD_STOP


def test_scope_excludes_a_future_phases_declared_outputs(tmp_path: Path) -> None:
    # phase 2 declares src/b.py; an option running in phase 1 may not write it.
    _write_plan(tmp_path, [["src/a.py"], ["src/b.py"]])
    in_phase1 = ScopeFilter().check(Gate(target_files=("src/b.py",), plan_dir=tmp_path, active_phase=1))
    assert in_phase1 is SafetyVerdict.ADD_STOP


def test_scope_excludes_a_phase_whose_number_is_not_an_int(tmp_path: Path) -> None:
    """A phase we cannot place (missing/non-int number) is fail-closed OUT of the approved set.

    Otherwise a later or unplaceable phase's outputs would broaden the approved-so-far set and
    defeat the future-phase exclusion the reader guarantees.
    """
    (tmp_path / "plan.toml").write_text(
        '[plan]\ntask = "t"\n[[phases]]\nnumber = 1\nfile = "phase-1.md"\n'
        '[[phases]]\nnumber = "two"\nfile = "phase-2.md"\n', encoding="utf-8")
    (tmp_path / "phase-1.md").write_text(
        f'```toml\n[phase]\n{_OUTPUT_FILES_KEY} = ["src/a.py"]\n```\n', encoding="utf-8")
    (tmp_path / "phase-2.md").write_text(
        f'```toml\n[phase]\n{_OUTPUT_FILES_KEY} = ["src/sneaky.py"]\n```\n', encoding="utf-8")
    declared = read_declared_outputs(tmp_path, up_to_phase=2)
    assert "src/a.py" in declared
    assert "src/sneaky.py" not in declared  # the string-numbered phase is excluded
    assert ScopeFilter().check(
        Gate(target_files=("src/sneaky.py",), plan_dir=tmp_path, active_phase=2)) is SafetyVerdict.ADD_STOP


def test_scope_excludes_a_phase_whose_number_is_zero_or_negative(tmp_path: Path) -> None:
    """A non-positive phase number is not a legitimate phase, so it fails closed OUT of the set.

    Phase numbers are 1-based; a 0 or negative value would otherwise satisfy `<= up_to_phase`
    for every real active phase and fail OPEN, unioning a corrupt phase's outputs into scope.
    """
    (tmp_path / "plan.toml").write_text(
        '[plan]\ntask = "t"\n[[phases]]\nnumber = 1\nfile = "phase-1.md"\n'
        '[[phases]]\nnumber = 0\nfile = "phase-0.md"\n'
        '[[phases]]\nnumber = -1\nfile = "phase-neg.md"\n', encoding="utf-8")
    (tmp_path / "phase-1.md").write_text(
        f'```toml\n[phase]\n{_OUTPUT_FILES_KEY} = ["src/a.py"]\n```\n', encoding="utf-8")
    (tmp_path / "phase-0.md").write_text(
        f'```toml\n[phase]\n{_OUTPUT_FILES_KEY} = ["src/zero.py"]\n```\n', encoding="utf-8")
    (tmp_path / "phase-neg.md").write_text(
        f'```toml\n[phase]\n{_OUTPUT_FILES_KEY} = ["src/neg.py"]\n```\n', encoding="utf-8")
    declared = read_declared_outputs(tmp_path, up_to_phase=1)
    assert "src/a.py" in declared
    assert "src/zero.py" not in declared  # phase 0 is excluded
    assert "src/neg.py" not in declared   # phase -1 is excluded


def test_read_declared_outputs_survives_a_nul_in_a_phase_filename(tmp_path: Path) -> None:
    """A `file=` with an embedded NUL passes the lexical safe-relative check but makes read_text
    raise ValueError, not OSError. The reader must stay non-raising and yield nothing for it."""
    (tmp_path / "plan.toml").write_text(
        '[plan]\ntask = "t"\n[[phases]]\nnumber = 1\nfile = "\\u0000bad.md"\n', encoding="utf-8")
    assert read_declared_outputs(tmp_path, up_to_phase=1) == frozenset()


def test_read_declared_outputs_survives_a_non_list_phases(tmp_path: Path) -> None:
    """A `phases` that is a scalar, not a list of tables (a corrupt plan.toml), must not raise a
    TypeError into the safety path -- the documented non-raising contract yields the empty set."""
    (tmp_path / "plan.toml").write_text('[plan]\ntask = "t"\nphases = 5\n', encoding="utf-8")
    assert read_declared_outputs(tmp_path, up_to_phase=1) == frozenset()


def test_a_nested_fence_does_not_close_the_outer_block_early(tmp_path: Path) -> None:
    """A shorter inner fence (an example ```toml inside a ```` block) must not close the longer
    outer fence, so the real `[phase]` frontmatter is read, not a decoy inside the example."""
    (tmp_path / "plan.toml").write_text(
        '[plan]\ntask = "t"\n[[phases]]\nnumber = 1\nfile = "phase-1.md"\n', encoding="utf-8")
    phase = (
        "````markdown\n"                       # 4-backtick outer fence (an example block)
        "Example of a phase file:\n"
        "```toml\n"                            # inner 3-backtick — must NOT close the outer
        "[phase]\n"
        'output_files = ["src/DECOY.py"]\n'
        "```\n"
        "````\n"                               # 4-backtick closer for the outer block
        "\n"
        "```toml\n"                            # the REAL frontmatter
        "[phase]\n"
        'output_files = ["src/real.py"]\n'
        "```\n")
    (tmp_path / "phase-1.md").write_text(phase, encoding="utf-8")
    declared = read_declared_outputs(tmp_path, up_to_phase=1)
    assert "src/real.py" in declared
    assert "src/DECOY.py" not in declared


@pytest.mark.parametrize("written", ["./src/a.py", "src/../src/a.py", "src/a.py "])
def test_scope_normalizes_trivially_equal_paths(tmp_path: Path, written: str) -> None:
    _write_plan(tmp_path, [["src/a.py"]])
    out = ScopeFilter().check(Gate(target_files=(written,), plan_dir=tmp_path, active_phase=1))
    assert out is SafetyVerdict.CLEAR


def test_scope_fails_closed_when_the_plan_declares_no_outputs(tmp_path: Path) -> None:
    _write_plan(tmp_path, [[]])
    out = ScopeFilter().check(Gate(target_files=("src/a.py",), plan_dir=tmp_path, active_phase=1))
    assert out is SafetyVerdict.INDETERMINATE


def test_read_declared_outputs_accepts_a_lone_string(tmp_path: Path) -> None:
    (tmp_path / "plan.toml").write_text('[plan]\ntask="t"\n[[phases]]\nnumber=1\nfile="p1.md"\n',
                                        encoding="utf-8")
    (tmp_path / "p1.md").write_text('```toml\n[phase]\noutput_files = "src/a.py"\n```\n', encoding="utf-8")
    assert read_declared_outputs(tmp_path, 1) == frozenset({"src/a.py"})


def test_reader_skips_a_non_phase_fence_before_the_real_one(tmp_path: Path) -> None:
    (tmp_path / "plan.toml").write_text('[plan]\ntask="t"\n[[phases]]\nnumber=1\nfile="p1.md"\n',
                                        encoding="utf-8")
    (tmp_path / "p1.md").write_text(
        '```pdsl\nUNIT Example\n```\n\n```toml\n[phase]\noutput_files = ["src/real.py"]\n```\n',
        encoding="utf-8")
    assert read_declared_outputs(tmp_path, 1) == frozenset({"src/real.py"})


def test_reader_ignores_a_phase_file_pointer_that_escapes_the_plan_dir(tmp_path: Path) -> None:
    (tmp_path / "plan.toml").write_text(
        '[plan]\ntask="t"\n[[phases]]\nnumber=1\nfile="/etc/hostname"\n'
        '[[phases]]\nnumber=2\nfile="../escape.md"\n', encoding="utf-8")
    assert read_declared_outputs(tmp_path, 9) == frozenset()


def test_read_declared_outputs_never_raises_on_a_missing_or_malformed_plan(tmp_path: Path) -> None:
    assert read_declared_outputs(tmp_path, 1) == frozenset()
    (tmp_path / "plan.toml").write_text("this is : not [ toml", encoding="utf-8")
    assert read_declared_outputs(tmp_path, 1) == frozenset()


# --- blocker ------------------------------------------------------------------------------

def test_blocker_fails_closed_on_an_empty_action() -> None:
    assert BlockerFilter().check(Gate(option_action="   ")) is SafetyVerdict.INDETERMINATE


#: (a realistic, often inflected, phrasing of a forbidden operation, an invariant-prose phrase the
#: operation's category is named by). The test asserts BOTH that the anchor is present in the
#: invariants (so representatives are real, not cherry-picked) AND that the blocker stops the
#: action -- so deleting any pattern makes a specific dangerous operation fail and the test red.
_FORBIDDEN_OPERATIONS = (
    ("RUN delete the stale files", "deletion prompts"),
    ("RUN removes the cache directory", "deleting files"),           # inflection
    ("RUN overwrite the output artifact", "overwrite prompts"),
    ("RUN truncate the audit log", "destructive-operation"),
    ("RUN drop the users table", "destructive-operation"),
    ("RUN destroy the workspace", "destructive-operation"),
    ("this step is irreversible", "irreversible-operation"),
    ("RUN replace the generated snapshots", "generated artifact replacement"),
    ("RUN shutdown the session", "shutdown confirmations"),
    ("RUN forgets the loaded rules", "forget"),                      # inflection
    ("RUN unload the session state", "unload"),
    ("the rule is disabled for this run", "disable"),                # inflection
    ("RUN weaken the STOP_TURN gate", "STOP_TURN"),
    ("RUN commit the changes", "committing"),
    ("RUN pushing the release tag", "pushing"),                      # inflection
    ("RUN force-push the branch", "force-pushing"),
    ("RUN git add the staging area", "staging"),
    ("RUN checkout over uncommitted work", "checkout over uncommitted"),
    ("RUN branch deletion of feature/x", "branch deletion"),
    ("RUN git rebase onto main", "history rewrite"),
    ("SET API_TOKEN = rotate now", "token"),
    ("RUN read the credential store", "credential"),
    ("RUN authenticate to the external service", "authentication"),
    ("RUN installation of the runtime", "install"),                  # inflection
    ("RUN update all dependencies to latest", "update"),
    ("RUN the first-time project setup", "setup"),
    ("SET SECURITY_GATE = off; security-impacting", "security"),
    ("RUN escalate to the host sandbox", "sandbox-escalation"),
    ("SET filesystem-permission = 0777", "filesystem-permission"),
    ("RUN sudo install a package", "install"),
    ("RUN external delegation to a worker", "external delegation"),
    ("RUN deploy to production", "deployment"),
    ("RUN publish the release", "publication"),
    ("RUN change the account owner email", "account changes"),
    ("RUN a cross-repository change", "cross-repository"),
    ("RUN a migration of the schema", "migrations"),
    ("RUN a broad mechanical rewrite of the tree", "broad mechanical rewrites"),
    ("EMIT_MENU DebuggerMenu", "debugger"),
    ("CONTINUE ReviewFixScope", "ReviewFixScope"),
)


@pytest.mark.parametrize("action,anchor", _FORBIDDEN_OPERATIONS)
def test_blocker_stops_every_forbidden_operation_whose_category_is_in_the_invariants(
        action: str, anchor: str) -> None:
    prose = _INVARIANTS.read_text(encoding="utf-8").lower()
    assert anchor.lower() in prose, (
        f"'{anchor}' is no longer in the invariants prose; re-anchor this operation to the "
        "current wording of the line it belongs to")
    assert BlockerFilter().check(Gate(option_action=action)) is SafetyVerdict.ADD_STOP, (
        f"the blocker cleared a forbidden operation: {action!r}")


@pytest.mark.parametrize("action", [
    "CONTINUE cf-generate-author",              # 'author' must not hit the auth patterns
    "SET REVIEW_REPORT_VIEW = table",
    "RUN the discovery step and continue",
    "RUN merge the review findings into one",   # 'merge findings', not a git merge
    "RUN a security review of the changes",     # 'security review', not a security operation
    "RUN update the progress counter",          # benign 'update', not a dependency/system update
])
# ('truncated diff preview' and 'DEPLOY_TARGET = staging' are deliberately NOT here: the floor
# over-blocks them toward safety -- 'truncate' and 'deploy/staging' are genuinely in the never-list,
# and a floor erring toward a stop on an ambiguous word is the correct direction.)
def test_blocker_does_not_over_block_common_studio_terms(action: str) -> None:
    assert BlockerFilter().check(Gate(option_action=action)) is SafetyVerdict.CLEAR
