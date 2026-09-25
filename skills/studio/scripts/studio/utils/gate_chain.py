"""The chain that decides whether a gate stops, from its declared type and two filter classes.

Everything before this built the vocabulary -- a gate carries a declared type, a plan can
answer a decision key, a reversal can be asserted armed, the ledger can record how a gate
resolved. None of it was wired to anything. This is the wiring, and the first place that can
change what a user sees -- which also makes it the first that can make things worse.

The order of the two classes is the whole safety argument, so it lives here, not in a caller:

**Safety runs first, and its result cannot be removed.** A safety filter may only *add* a
stop; an economy filter may only *remove* one. Running safety first and then letting economy
remove a stop it added would defeat the point, so an added stop is final -- economy removes
only a stop the *declared type* created, never one a safety filter added. Every path that
resolves autonomously is reachable only with all safety filters clear.

**The declared type is a ceiling, never a floor.** Nothing here promotes ``blocking`` to
auto-proceed; a ``blocking`` gate always asks.

This is the skeleton (increment 1): the two filter classes, the outcome, and the chain that
orders them. **No filters ship in it** -- with none registered every gate returns ``ASK``,
which is identical to today. The filters (scope, reversibility, blocker; already-answered,
plan-and-ledger) and the ledger write path are later increments, wired then. Nothing here is
called by a runtime yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol

from .decision_log import GateRuling


#: The one declared type the ceiling refuses to auto-resolve. Kept as a bare token and bound
#: to the PDSL vocabulary by a test rather than imported at module load, so reading the
#: ceiling costs no parser import -- the same reason ``decision_log`` reads ``GATE_TYPES``
#: lazily. If the token is renamed in ``pdsl.GATE_TYPES`` the binding test fails here.
BLOCKING = "blocking"


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-outcome
class OutcomeKind(Enum):
    """The three ways the chain can end (design note §4)."""

    ASK = "ask"          # stop and ask the user, exactly as today
    RESOLVE = "resolve"  # an economy filter answered it; the ruling is recorded
    DEFER = "defer"      # neither answered nor safe to answer; raise as an open question


@dataclass(frozen=True)
class ChainOutcome:
    """What the chain decided, and the evidence that belongs with it.

    Built through ``ask`` / ``resolve`` / ``defer`` so an ill-formed outcome -- a ``RESOLVE``
    with no ruling, an ``ASK`` carrying one -- cannot be constructed. ``__post_init__``
    enforces the same, because a frozen dataclass can still be instantiated directly and a
    guard that only the named constructors honour is not a guard.
    """

    kind: OutcomeKind
    ruling: Optional[GateRuling] = None
    defer_reason: str = ""

    def __post_init__(self) -> None:
        if self.kind is OutcomeKind.RESOLVE and self.ruling is None:
            raise ValueError("a RESOLVE outcome must carry the GateRuling that answered the gate")
        if self.kind is not OutcomeKind.RESOLVE and self.ruling is not None:
            raise ValueError(f"a {self.kind.name} outcome must not carry a ruling")
        if self.kind is OutcomeKind.DEFER and not self.defer_reason:
            raise ValueError("a DEFER outcome must say why it could not resolve")

    @classmethod
    def ask(cls) -> "ChainOutcome":
        """The gate stops and asks the user, exactly as today."""
        return cls(OutcomeKind.ASK)

    @classmethod
    def resolve(cls, ruling: GateRuling) -> "ChainOutcome":
        """The gate resolves autonomously, carrying the ruling that answered it."""
        return cls(OutcomeKind.RESOLVE, ruling=ruling)

    @classmethod
    def defer(cls, reason: str) -> "ChainOutcome":
        """The gate is raised as an open question, with the reason it could not resolve."""
        return cls(OutcomeKind.DEFER, defer_reason=reason)
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-outcome


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-verdicts
class SafetyVerdict(Enum):
    """A safety filter may only add a stop; indeterminate is read as a stop (fail closed)."""

    CLEAR = "clear"                  # this filter sees no reason to stop
    ADD_STOP = "add_stop"            # this filter requires a stop
    INDETERMINATE = "indeterminate"  # cannot tell -- the chain reads it as ADD_STOP


class EconomyDecision(Enum):
    """An economy filter may only remove a stop, or decline to."""

    NO_OPINION = "no_opinion"        # nothing to say; the declared stop stands
    REMOVE = "remove"                # a declared source answered the gate; carries the ruling
    INDETERMINATE = "indeterminate"  # could not rule from a declared source -- the chain defers


@dataclass(frozen=True)
class EconomyVerdict:
    """An economy filter's answer. ``REMOVE`` carries the ruling that justifies removing the stop."""

    decision: EconomyDecision
    ruling: Optional[GateRuling] = None

    def __post_init__(self) -> None:
        if self.decision is EconomyDecision.REMOVE and self.ruling is None:
            raise ValueError("a REMOVE verdict must carry the GateRuling that answered the gate")
        if self.decision is not EconomyDecision.REMOVE and self.ruling is not None:
            raise ValueError(f"a {self.decision.name} verdict must not carry a ruling")

    @classmethod
    def no_opinion(cls) -> "EconomyVerdict":
        """This filter has nothing to say; the declared stop stands."""
        return cls(EconomyDecision.NO_OPINION)

    @classmethod
    def remove(cls, ruling: GateRuling) -> "EconomyVerdict":
        """A declared source answered the gate; remove the stop, carrying the ruling."""
        return cls(EconomyDecision.REMOVE, ruling=ruling)

    @classmethod
    def indeterminate(cls) -> "EconomyVerdict":
        """This filter could not rule from a declared source; the chain defers."""
        return cls(EconomyDecision.INDETERMINATE)
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-verdicts


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-filters
class SafetyFilter(Protocol):  # pylint: disable=too-few-public-methods
    """May only add a stop. Return ``INDETERMINATE`` when unsure -- the chain fails it closed."""

    def check(self, gate: object) -> SafetyVerdict:
        """Report whether this filter clears the gate, requires a stop, or cannot tell."""


class EconomyFilter(Protocol):  # pylint: disable=too-few-public-methods
    """May only remove a stop the declared type created. Return ``NO_OPINION`` to abstain."""

    def check(self, gate: object) -> EconomyVerdict:
        """Report whether this filter removes the stop, abstains, or cannot rule."""
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-filters


# @cpt-begin:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-resolve
def resolve_gate(
    gate: object,
    declared_type: str,
    safety: List[SafetyFilter],
    economy: List[EconomyFilter],
) -> ChainOutcome:
    """Decide whether ``gate`` stops, in the fixed order of design note §4.

    Safety first: any filter that is not ``CLEAR`` -- including one that cannot tell -- ends
    the chain at ``ASK``, and that stop is final. Only with every safety filter clear does the
    ceiling apply and then economy run: a ``blocking`` gate always asks; otherwise the first
    economy filter that removes the stop resolves it, an economy filter that cannot rule from a
    declared source defers, and silence leaves the declared stop standing.

    With no filters registered -- the whole of increment 1 -- every gate returns ``ASK``, which
    is what runs today. ``gate`` is opaque here; the filters that read its fields arrive with
    the later increments, and the chain itself needs only ``declared_type`` for the ceiling.
    """
    for safety_filter in safety:
        if safety_filter.check(gate) is not SafetyVerdict.CLEAR:
            # ADD_STOP, or INDETERMINATE read as a stop: the safety class fails closed, and an
            # added stop is not something economy is allowed to reach, let alone remove.
            return ChainOutcome.ask()

    if declared_type == BLOCKING:
        # The ceiling. A blocking gate's stop is not one the type lets anything remove, so
        # economy never runs for it. Which economy filter may act on a confirmation vs a
        # decision gate is settled when the economy filters land (increment 3).
        return ChainOutcome.ask()

    verdicts = [economy_filter.check(gate) for economy_filter in economy]
    for verdict in verdicts:
        if verdict.decision is EconomyDecision.REMOVE:
            # A REMOVE verdict is built only with a ruling (EconomyVerdict.__post_init__), and it
            # wins over a later INDETERMINATE: a gate a declared source answered is resolved,
            # whatever another economy filter could not tell.
            return ChainOutcome.resolve(verdict.ruling)
    if any(verdict.decision is EconomyDecision.INDETERMINATE for verdict in verdicts):
        return ChainOutcome.defer("an economy filter could not rule the gate from a declared source")
    return ChainOutcome.ask()
# @cpt-end:cpt-studio-algo-core-infra-gate-chain:p1:inst-chain-resolve
