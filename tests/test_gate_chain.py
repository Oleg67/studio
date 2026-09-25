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
    ChainOutcome,
    EconomyDecision,
    EconomyVerdict,
    OutcomeKind,
    SafetyVerdict,
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

    def check(self, gate: object) -> EconomyVerdict:
        return self._verdict


def test_the_blocking_token_matches_the_pdsl_vocabulary() -> None:
    # The ceiling reads a bare token; if pdsl renames it, this catches the drift here.
    assert BLOCKING in pdsl.GATE_TYPES


def test_an_empty_chain_asks_every_gate() -> None:
    # Increment 1 registers no filters, so behaviour is identical to today.
    assert resolve_gate(object(), "confirmation", [], []).kind is OutcomeKind.ASK


def test_safety_wins_over_economy() -> None:
    # The load-bearing invariant: a stop a safety filter added is final, and economy — which
    # here would remove it — never even runs.
    out = resolve_gate(
        object(),
        "confirmation",
        [_Safety(SafetyVerdict.ADD_STOP)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_indeterminate_safety_fails_closed() -> None:
    out = resolve_gate(
        object(),
        "confirmation",
        [_Safety(SafetyVerdict.INDETERMINATE)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_the_ceiling_holds_a_blocking_gate_never_resolves() -> None:
    out = resolve_gate(
        object(),
        BLOCKING,
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.ASK


def test_resolves_only_when_all_safety_clear_and_type_permits() -> None:
    out = resolve_gate(
        object(),
        "confirmation",
        [_Safety(SafetyVerdict.CLEAR), _Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.no_opinion()), _Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.RESOLVE
    assert out.ruling is _RULING


def test_indeterminate_economy_defers_with_a_reason() -> None:
    out = resolve_gate(
        object(),
        "decision",
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.indeterminate())],
    )
    assert out.kind is OutcomeKind.DEFER
    assert out.defer_reason


def test_a_remove_wins_over_a_later_indeterminate() -> None:
    # A gate a declared source answered is resolved, whatever another economy filter could not tell.
    out = resolve_gate(
        object(),
        "decision",
        [_Safety(SafetyVerdict.CLEAR)],
        [_Economy(EconomyVerdict.indeterminate()), _Economy(EconomyVerdict.remove(_RULING))],
    )
    assert out.kind is OutcomeKind.RESOLVE
    assert out.ruling is _RULING


def test_all_no_opinion_leaves_the_declared_stop_standing() -> None:
    out = resolve_gate(
        object(),
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
