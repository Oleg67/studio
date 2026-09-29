"""The chain orders safety before economy, and a safety-added stop cannot be removed.

The one invariant a future contributor is most tempted to relax is that an autonomous
resolution is reachable only with every safety filter clear. It is bound here with fake
filters, because the real filters (scope, reversibility, blocker; already-answered,
plan-and-ledger) arrive in later increments and this skeleton must hold before them.
"""
from __future__ import annotations

import pytest

from studio.utils import pdsl
from studio.utils.decision_log import GateRuling
from studio.utils.gate_chain import (
    BLOCKING,
    CONFIRMATION,
    DECISION,
    ChainOutcome,
    EconomyDecision,
    EconomyVerdict,
    Gate,
    OutcomeKind,
    SafetyVerdict,
    _ECONOMY_ELIGIBLE,
    resolve_gate,
)

_RULING = GateRuling(decision_key="k", value="v", provenance="plan", status="resolved")


class _Safety:
    """A safety filter that always returns the verdict it was built with."""

    def __init__(self, verdict: SafetyVerdict) -> None:
        self._verdict = verdict

    def check(self, gate: object) -> SafetyVerdict:
        return self._verdict


class _Economy:
    """An economy filter that always returns the verdict it was built with."""

    def __init__(self, verdict: EconomyVerdict) -> None:
        self._verdict = verdict
        self.called = False

    def check(self, gate: object) -> EconomyVerdict:
        self.called = True
        return self._verdict


def test_the_blocking_token_matches_the_pdsl_vocabulary() -> None:
    # The ceiling reads a bare token; if pdsl renames it, this catches the drift here.
    assert BLOCKING in pdsl.GATE_TYPES


def test_an_empty_chain_asks_every_gate() -> None:
    # Increment 1 registers no filters, so behaviour is identical to today.
    assert resolve_gate(Gate(), "confirmation", [], []).kind is OutcomeKind.ASK


def test_safety_wins_over_economy() -> None:
    # The load-bearing invariant: a stop a safety filter added is final, and economy — which
    # here would remove it — never even runs.
    out = resolve_gate(
        Gate(),
        "confirmation",
        [_Safety(SafetyVerdict.ADD_STOP)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_indeterminate_safety_fails_closed() -> None:
    out = resolve_gate(
        Gate(),
        "confirmation",
        [_Safety(SafetyVerdict.INDETERMINATE)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_the_ceiling_holds_a_blocking_gate_never_resolves() -> None:
    out = resolve_gate(
        Gate(),
        BLOCKING,
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_resolves_only_when_all_safety_clear_and_type_permits() -> None:
    out = resolve_gate(
        Gate(),
        "confirmation",
        [_Safety(SafetyVerdict.CLEAR), _Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.no_opinion()), _Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.RESOLVE
    assert out.ruling is _RULING


def test_indeterminate_economy_defers_with_a_reason() -> None:
    out = resolve_gate(
        Gate(),
        "decision",
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.indeterminate())],
    )
    assert out.kind is OutcomeKind.DEFER
    assert out.defer_reason


def test_a_remove_wins_over_a_later_indeterminate() -> None:
    # A gate a declared source answered is resolved, whatever another economy filter could not tell.
    out = resolve_gate(
        Gate(),
        "decision",
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.indeterminate()), _Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.RESOLVE
    assert out.ruling is _RULING


def test_all_no_opinion_leaves_the_declared_stop_standing() -> None:
    out = resolve_gate(
        Gate(),
        "confirmation",
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.no_opinion())],
    )
    assert out.kind is OutcomeKind.ASK


def test_a_resolve_outcome_must_carry_a_ruling() -> None:
    with pytest.raises(ValueError):
        ChainOutcome(OutcomeKind.RESOLVE)
    with pytest.raises(ValueError):
        ChainOutcome(OutcomeKind.ASK, ruling=_RULING)


def test_a_defer_outcome_must_state_a_reason() -> None:
    with pytest.raises(ValueError):
        ChainOutcome(OutcomeKind.DEFER)


def test_a_remove_verdict_must_carry_a_ruling() -> None:
    with pytest.raises(ValueError):
        EconomyVerdict(EconomyDecision.REMOVE)
    with pytest.raises(ValueError):
        EconomyVerdict(EconomyDecision.NO_OPINION, ruling=_RULING)


def test_an_unknown_declared_type_fails_closed() -> None:
    # The one input that decides whether the ceiling applies must not fail open: an
    # unrecognised or misspelled type is treated like `blocking`, not as economy-eligible.
    out = resolve_gate(
        Gate(),
        "confimation",  # a typo, not in GATE_TYPES
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_a_non_string_declared_type_fails_closed_without_raising() -> None:
    # A caller violating the `str` type hint with a non-hashable value (a list, a dict) must
    # degrade to ASK, not raise TypeError from the frozenset membership test -- the module's
    # never-raises contract holds for that input too. The `isinstance` guard runs first.
    out = resolve_gate(
        Gate(),
        ["confirmation"],  # type: ignore[arg-type]  # non-string, unhashable
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_economy_eligible_plus_blocking_is_the_whole_vocabulary() -> None:
    # Binds the local constants to pdsl's set: if a fourth gate type is added, this fails so the
    # ceiling's fail-closed set is reconsidered rather than silently excluding the new type.
    assert _ECONOMY_ELIGIBLE | {BLOCKING} == set(pdsl.GATE_TYPES)
    assert CONFIRMATION in _ECONOMY_ELIGIBLE
    assert DECISION in _ECONOMY_ELIGIBLE


def test_an_ask_or_resolve_outcome_must_not_carry_a_defer_reason() -> None:
    with pytest.raises(ValueError):
        ChainOutcome(OutcomeKind.ASK, defer_reason="stray")
    with pytest.raises(ValueError):
        ChainOutcome(OutcomeKind.RESOLVE, ruling=_RULING, defer_reason="stray")


def test_a_safety_stop_short_circuits_before_economy_is_consulted() -> None:
    # Not just that the outcome is ASK, but that economy is never even asked — the short-circuit
    # itself, which a refactor could break while leaving the outcome accidentally right.
    economy = _Economy(EconomyVerdict.remove(_RULING))
    out = resolve_gate(Gate(), CONFIRMATION, [_Safety(SafetyVerdict.ADD_STOP)], [economy])
    assert out.kind is OutcomeKind.ASK
    assert economy.called is False


def test_a_later_safety_filter_still_blocks_economy() -> None:
    # A stop from a non-first safety filter is as final as one from the first: the chain checks
    # every safety filter, not just the head.
    economy = _Economy(EconomyVerdict.remove(_RULING))
    out = resolve_gate(
        Gate(),
        CONFIRMATION,
        [_Safety(SafetyVerdict.CLEAR), _Safety(SafetyVerdict.ADD_STOP)],
        [economy],
    )
    assert out.kind is OutcomeKind.ASK
    assert economy.called is False


def test_real_filters_fail_closed_end_to_end() -> None:
    # Not fakes: a gate the seam under-populated (writes a file but carries no root/plan/phase)
    # runs the REAL filters, which return INDETERMINATE, and the chain must turn that into ASK —
    # even with an economy filter that would remove. Proves the filter->chain seam.
    from studio.utils.gate_filters import BlockerFilter, ReversibilityFilter, ScopeFilter
    out = resolve_gate(
        Gate(option_action="WRITE src/x.py", target_files=("src/x.py",)),
        CONFIRMATION,
        [ReversibilityFilter(), ScopeFilter(), BlockerFilter()],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_real_filters_allow_a_resolve_when_all_clear() -> None:
    # Nothing written and a benign action: all three real safety filters CLEAR, so an economy
    # REMOVE is honoured end-to-end.
    from studio.utils.gate_filters import BlockerFilter, ReversibilityFilter, ScopeFilter
    out = resolve_gate(
        Gate(option_action="CONTINUE NextPhase", target_files=()),
        CONFIRMATION,
        [ReversibilityFilter(), ScopeFilter(), BlockerFilter()],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.RESOLVE
