"""ROUTING-1 as code; pre-registration §12's decision table is rendered from this module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Iterator, Mapping, Sequence

__all__ = [
    "GATE_THRESHOLD",
    "GATE_THRESHOLD_TEXT",
    "Gate",
    "Validity",
    "Th17Authoring",
    "Th17b",
    "T12",
    "Conclusion",
    "State",
    "Rule",
    "RoutingRow",
    "AmbiguousRouting",
    "VARIABLES",
    "RULES",
    "DEFAULT_RULE",
    "TABLE_ID",
    "THESIS_REFUTING",
    "ARM_UNFUNDED_VALUES",
    "UNMEASURED_VALUES",
    "all_states",
    "states",
    "route",
    "deciding_rule",
    "matching_rules",
    "reachable_conclusions",
    "parse_constraint",
    "routing_rows",
    "render_routing_section",
]


# =================================================================================================
# The state space.
# =================================================================================================
class Gate(Enum):
    """§6 step 5's verdict."""

    PASS = "pass"
    FAIL = "fail"


class Validity(Enum):
    """§6 step 0's validity conclusion, over the F1 diagnostics."""

    CLEAR = "clear"
    BREACHED = "breached"


class Th17Authoring(Enum):
    """TH-17's authoring share of the iterative arm's scored agent failures, against 0.50."""

    GE_050 = "ge_050"
    LT_050 = "lt_050"


class Th17b(Enum):
    """TH-17b, named by what it indicates: an F2 indicator, with the polarity in the names."""

    #: The convergence rule holds: revised and improved. Does not indicate F2.
    F2_NOT_INDICATED = "f2_not_indicated"
    #: The arm revised without converging. This DOES indicate F2.
    F2_INDICATED = "f2_indicated"
    #: The revising set is smaller than TH-17b's 20-run floor, so the rule cannot be evaluated.
    NOT_EVALUABLE = "not_evaluable"


class T12(Enum):
    """T12's typed DSP-graph arm; stated without the number so GATE_THRESHOLD stays single."""

    #: A ledger fact fixed when N is frozen, before any of this is measured (§3).
    UNFUNDED = "unfunded"
    #: Funded and not yet run: the state F2's action is written for.
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


class Conclusion(Enum):
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F_UNDETERMINED = "F-undetermined"
    NO_FAILURE_CONCLUSION = "no-failure-conclusion"


#: The only thesis-refuting conclusion.
THESIS_REFUTING = Conclusion.F3


# =================================================================================================
# The join to the ledger, stated once.
# =================================================================================================
#: Routing variables that are a statistic of a separately ticketed ledger arm, mapped to the
#: value each takes when no ledger line funds that arm. The ticket is the variable's own name
#: upper-cased; tests derive what each value costs from the ledger, and the named value may
#: appear in no rule guard.
ARM_UNFUNDED_VALUES: dict[str, Enum] = {"t12": T12.UNFUNDED}

#: Values recording a failure to measure rather than a measured outcome.
UNMEASURED_VALUES: frozenset[Enum] = frozenset(
    {Th17b.NOT_EVALUABLE, *ARM_UNFUNDED_VALUES.values()}
)

#: What makes validity=breached, stated once; three prose sites are checked against it.
VALIDITY_DIAGNOSTICS: tuple[str, ...] = (
    "TH-8",
    "TH-9",
    "TH-10",
    "TH-11",
    "TH-18",
    "TH-19",
    "TH-20",
    "TH-21",
)
#: Not triggers: under-coverage makes the stated level wrong, not the comparison uninformative,
#: so TH-22's consequence is verdict-conditional rather than an F1 route.
VALIDITY_NON_TRIGGERS: tuple[str, ...] = ("TH-22",)

TABLE_ID = "ROUTING-1"

#: TH-1, and the only place this repository writes the number down. Every statement of the
#: threshold is interpolated from here, including the code that applies it
#: (arms.bestofn.margin_verdict), so the published number cannot move without the applied one.
GATE_THRESHOLD = 0.15

#: How the threshold is spelled in prose; rendering is part of the single source.
GATE_THRESHOLD_TEXT = f"+{GATE_THRESHOLD:.2f}"


@dataclass(frozen=True)
class State:
    """One campaign's result, over exactly the variables ROUTING-1 is written on."""

    gate: Gate
    validity: Validity
    th17_authoring: Th17Authoring
    th17b: Th17b
    t12: T12

    def value(self, variable: str) -> Enum:
        return getattr(self, variable)

    def tokens(self) -> dict[str, str]:
        return {name: self.value(name).value for name, _enum, _gloss in VARIABLES}

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return ", ".join(f"{name}={token}" for name, token in self.tokens().items())


#: `(attribute name, enum, what the variable is)`, in the order §12 prints them.
VARIABLES: tuple[tuple[str, type[Enum], str], ...] = (
    (
        "gate",
        Gate,
        "§6 step 5's verdict: the BCa 2.5th percentile of Δ̂ is at or above "
        f"{GATE_THRESHOLD_TEXT}, or it is not.",
    ),
    (
        "validity",
        Validity,
        "§6 step 0's validity conclusion over the F1 diagnostics.",
    ),
    (
        "th17_authoring",
        Th17Authoring,
        "TH-17's authoring share of the iterative arm's `scored_agent_failure` runs, against 0.50.",
    ),
    (
        "th17b",
        Th17b,
        "TH-17b's convergence rule over the runs that actually revised. It is an F2 **indicator**, "
        "and the value names say which way it points.",
    ),
    (
        "t12",
        T12,
        f"T12's typed DSP-graph arm against the same {GATE_THRESHOLD_TEXT} CI-lower-bound rule "
        "on the same signals.",
    ),
)

#: One line per value, printed as §12's `means` column. Written here so §12 cannot restate it.
GLOSSES: dict[Enum, str] = {
    Gate.PASS: f"the lower bound cleared {GATE_THRESHOLD_TEXT}.",
    Gate.FAIL: "it did not.",
    Validity.CLEAR: "none of the F1 diagnostics fired.",
    Validity.BREACHED: "PLACEHOLDER",  # filled below, from VALIDITY_DIAGNOSTICS
    Th17Authoring.GE_050: "the authoring share is at or above 0.50 — the taxonomy points at the "
    "action space.",
    Th17Authoring.LT_050: "it is below 0.50 — the taxonomy does not point at the action space.",
    Th17b.F2_NOT_INDICATED: (
        "the convergence rule **holds** — the arm revised and improved — which does *not* "
        "indicate F2."
    ),
    Th17b.F2_INDICATED: "the arm revised **without** converging, which **does** indicate F2.",
    Th17b.NOT_EVALUABLE: (
        "the revising set is smaller than TH-17b's 20-run floor, so the rule has no value. F2 is "
        "unavailable; that is what closes it, not a judgement about the action space."
    ),
    T12.UNFUNDED: (
        "no ledger line funds the arm. A fact fixed when N is frozen, before any of this is "
        "measured (§3)."
    ),
    T12.NOT_RUN: "funded and not yet run — the state F2's action is written for.",
    T12.PASSED: f"run, and it cleared {GATE_THRESHOLD_TEXT}.",
    T12.FAILED: "run, and it failed the same threshold on the same signals.",
}

GLOSSES[Validity.BREACHED] = (
    "**any** of "
    + ", ".join(VALIDITY_DIAGNOSTICS[:-1])
    + f" or {VALIDITY_DIAGNOSTICS[-1]} fired (TH-11 including §10's not-obtainable rate). "
    + ", ".join(VALIDITY_NON_TRIGGERS)
    + " is deliberately not among them — see F1 below."
)


def all_states() -> Iterator[State]:
    """Every point of the state space, in declaration order."""
    domains = [tuple(enum) for _name, enum, _gloss in VARIABLES]
    for combination in itertools.product(*domains):
        yield State(*combination)


def states(partial: Mapping[str, Sequence[Enum]] | None = None) -> Iterator[State]:
    """Every state consistent with a partial assignment of variables to admissible values."""
    partial = dict(partial or {})
    unknown = set(partial) - {name for name, _enum, _gloss in VARIABLES}
    if unknown:
        raise KeyError(f"no such routing variable(s): {sorted(unknown)}")
    domains = [
        tuple(partial[name]) if name in partial else tuple(enum)
        for name, enum, _gloss in VARIABLES
    ]
    for combination in itertools.product(*domains):
        yield State(*combination)


# =================================================================================================
# The rules.
# =================================================================================================
class AmbiguousRouting(Exception):
    """Two rules matched one state."""


@dataclass(frozen=True)
class Rule:
    """One guarded rule. `guard` is a conjunction of `variable == value`, never a disjunction."""

    id: str
    guard: tuple[tuple[str, Enum], ...]
    conclusion: Conclusion
    because: str

    def matches(self, state: State) -> bool:
        return all(state.value(name) is value for name, value in self.guard)

    def disjoint_from(self, other: "Rule") -> bool:
        """True iff no state can satisfy both guards; decided symbolically."""
        mine = dict(self.guard)
        theirs = dict(other.guard)
        return any(
            mine[name] is not theirs[name] for name in mine.keys() & theirs.keys()
        )

    def when(self) -> str:
        return ", ".join(f"{name}={value.value}" for name, value in self.guard) or "*"


def _rule(id: str, conclusion: Conclusion, because: str, **guard: Enum) -> Rule:
    order = [name for name, _enum, _gloss in VARIABLES]
    ordered = tuple((name, guard[name]) for name in order if name in guard)
    assert len(ordered) == len(guard), f"{id}: guard names a variable ROUTING-1 does not have"
    return Rule(id=id, guard=ordered, conclusion=conclusion, because=because)


#: The guarded rules, pairwise disjoint: pre-emptions are written into the guards, route()
#: refuses to pick between two matches, and re-ordering this tuple changes nothing.
RULES: tuple[Rule, ...] = (
    _rule(
        "R1",
        Conclusion.F1,
        "F1 pre-empts everything, including a PASS: an uninformative instrument supports no "
        "thesis claim in either direction.",
        validity=Validity.BREACHED,
    ),
    _rule(
        "R2",
        Conclusion.NO_FAILURE_CONCLUSION,
        "A clean PASS is not a failure conclusion. `validity=clear` is written into the guard "
        "rather than left to rule order, because F1 attaches to a PASS too.",
        validity=Validity.CLEAR,
        gate=Gate.PASS,
    ),
    _rule(
        "R3",
        Conclusion.F2,
        "The taxonomy points at the action space and T12 is funded and not yet run — the state "
        "F2's action (run T12) is written for.",
        gate=Gate.FAIL,
        validity=Validity.CLEAR,
        th17_authoring=Th17Authoring.GE_050,
        th17b=Th17b.F2_INDICATED,
        t12=T12.NOT_RUN,
    ),
    _rule(
        "R4",
        Conclusion.F2,
        "Same taxonomy, T12 already run and passing: the typed-graph arm clearing the bar is "
        "evidence for the action-space reading, not against it.",
        gate=Gate.FAIL,
        validity=Validity.CLEAR,
        th17_authoring=Th17Authoring.GE_050,
        th17b=Th17b.F2_INDICATED,
        t12=T12.PASSED,
    ),
    _rule(
        "R5",
        Conclusion.F2,
        "Same taxonomy, T12 run and failed. F2 pre-empts F3 by disjointness, not by line order: "
        "F3's guards require `th17_authoring=lt_050`, which this rule excludes.",
        gate=Gate.FAIL,
        validity=Validity.CLEAR,
        th17_authoring=Th17Authoring.GE_050,
        th17b=Th17b.F2_INDICATED,
        t12=T12.FAILED,
    ),
    _rule(
        "R6",
        Conclusion.F3,
        "Gate FAIL, instrument clean, the taxonomy does not indicate F2 on either half, and "
        "T12's arm failed the same threshold on the same signals.",
        gate=Gate.FAIL,
        validity=Validity.CLEAR,
        th17_authoring=Th17Authoring.LT_050,
        th17b=Th17b.F2_NOT_INDICATED,
        t12=T12.FAILED,
    ),
    _rule(
        "R7",
        Conclusion.F3,
        "The same, with TH-17b unevaluable rather than evaluated: unevaluability closes F2, "
        "which is exactly what makes F3's own precondition — F2 not indicated — true "
        "(resolution 1).",
        gate=Gate.FAIL,
        validity=Validity.CLEAR,
        th17_authoring=Th17Authoring.LT_050,
        th17b=Th17b.NOT_EVALUABLE,
        t12=T12.FAILED,
    ),
)

#: The complement of the union above, named separately from the disjointness argument.
DEFAULT_RULE = Rule(
    id="R8",
    guard=(),
    conclusion=Conclusion.F_UNDETERMINED,
    because=(
        "Everything else: a FAIL with a clean instrument that reaches neither F2's nor F3's "
        "guard. This is what makes \"neither F2 nor F3\" computable rather than a phrase."
    ),
)


def matching_rules(state: State) -> tuple[Rule, ...]:
    return tuple(rule for rule in RULES if rule.matches(state))


def deciding_rule(state: State) -> Rule:
    """The rule that decides `state`, or DEFAULT_RULE if no guarded rule matches."""
    matched = matching_rules(state)
    if len(matched) > 1:
        raise AmbiguousRouting(
            f"{state} matches {[rule.id for rule in matched]}. The guards are meant to be "
            "pairwise disjoint; with an overlap the answer would depend on rule order, which is "
            "what round 5's table did and nothing tested."
        )
    return matched[0] if matched else DEFAULT_RULE


def route(state: State) -> Conclusion:
    """Where a result lands. Total over the state space, and independent of rule order."""
    return deciding_rule(state).conclusion


def reachable_conclusions() -> frozenset[Conclusion]:
    return frozenset(route(state) for state in all_states())


# =================================================================================================
# Reading the documents' `when:` clauses.
# =================================================================================================
#: Pre-polarity spellings still used by pre-registration §3; scheduled for deletion.
LEGACY_VALUE_SPELLINGS: dict[tuple[str, str], Enum] = {
    ("th17b", "satisfied"): Th17b.F2_NOT_INDICATED,
    ("th17b", "not_satisfied"): Th17b.F2_INDICATED,
}


def parse_constraint(text: str) -> dict[str, tuple[Enum, ...]]:
    """`"gate=fail, validity=clear"` -> `{"gate": (Gate.FAIL,), "validity": (Validity.CLEAR,)}`.

    Raises :class:`ValueError` with the reason, so a document's `when:` clause that names a
    variable or value the routing does not have fails loudly instead of matching nothing.
    """
    domains = {name: enum for name, enum, _gloss in VARIABLES}
    partial: dict[str, tuple[Enum, ...]] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        name, _, token = piece.partition("=")
        name, token = name.strip(), token.strip()
        if name not in domains:
            raise ValueError(f"{name!r} is not a routing variable; they are {sorted(domains)}")
        legacy = LEGACY_VALUE_SPELLINGS.get((name, token))
        if legacy is not None:
            value = legacy
        else:
            try:
                value = domains[name](token)
            except ValueError:
                raise ValueError(
                    f"{name}={token!r} is not a value of {name}; they are "
                    f"{[member.value for member in domains[name]]}"
                ) from None
        if name in partial and partial[name] != (value,):
            raise ValueError(f"{name} is constrained twice, to different values")
        partial[name] = (value,)
    return partial


# =================================================================================================
# The renderer. `docs/pre-registration.md` §12's decision table is this function's output.
# =================================================================================================
GENERATED_BEGIN = "<!-- BEGIN GENERATED: modembench.conclusions.render_routing_section() -->"
GENERATED_END = "<!-- END GENERATED -->"


@dataclass(frozen=True)
class RoutingRow:
    """One rule as §12 states it, computed once; both of §12's renderings read `when` only."""

    rule: Rule
    #: The canonical guard, `variable=value, …`, or `*` for the catch-all's empty guard.
    when: str
    conclusion: str
    #: How many of the state space's points this rule decides.
    decides: int

    @property
    def declaration(self) -> str:
        """The machine-readable `rule:` line."""
        return f"rule: {self.when} -> {self.conclusion}"

    @property
    def cell(self) -> str:
        """The table's `when` column; the catch-all's empty guard renders as prose."""
        return "*every other state*" if self.when == "*" else f"`{self.when}`"


def routing_rows() -> tuple[RoutingRow, ...]:
    """The rules as §12 prints them. The single source both of §12's renderings read."""
    space = tuple(all_states())
    return tuple(
        RoutingRow(
            rule=rule,
            when=rule.when(),
            conclusion=rule.conclusion.value,
            decides=sum(1 for state in space if deciding_rule(state) is rule),
        )
        for rule in RULES + (DEFAULT_RULE,)
    )


def _declaration() -> list[str]:
    lines = ["```modembench-check", "kind: conclusion-routing", f"id: {TABLE_ID}"]
    for name, enum, _gloss in VARIABLES:
        lines.append(f"variable: {name} = " + " | ".join(member.value for member in enum))
    lines.extend(row.declaration for row in routing_rows())
    lines.append("```")
    return lines


def _variable_table() -> list[str]:
    lines = [
        "| variable | value | means |",
        "| --- | --- | --- |",
    ]
    for name, enum, purpose in VARIABLES:
        for index, member in enumerate(enum):
            first = f"`{name}` — {purpose}" if index == 0 else ""
            lines.append(f"| {first} | `{member.value}` | {GLOSSES[member]} |")
    return lines


def _rule_table() -> list[str]:
    """The human-readable table; it states no guard of its own."""
    lines = [
        "| rule | when | conclusion | why |",
        "| --- | --- | --- | --- |",
    ]
    total = sum(1 for _ in all_states())
    for row in routing_rows():
        lines.append(
            f"| {row.rule.id} | {row.cell} | **{row.conclusion}** | {row.rule.because} "
            f"*(decides {row.decides} of {total} states.)* |"
        )
    return lines


def _census() -> list[str]:
    total = sum(1 for _ in all_states())
    counts = {conclusion: 0 for conclusion in Conclusion}
    for state in all_states():
        counts[route(state)] += 1
    census = ", ".join(
        f"**{conclusion.value}** {count}"
        for conclusion, count in counts.items()
        if count or conclusion in reachable_conclusions()
    )
    return [
        f"**The census, so the table's shape is visible rather than trusted.** The state space is "
        f"{total} points and every one of them routes: {census}. No state matches two rules — the "
        "guards are pairwise disjoint, so this table means the same thing in any order, and "
        "`route()` raises rather than choosing if that ever stops being true. The one conclusion "
        "that refutes the thesis is "
        f"{THESIS_REFUTING.value}, reachable from "
        f"{sum(1 for state in all_states() if route(state) is THESIS_REFUTING)} of them.",
    ]


def render_routing_section() -> str:
    """§12's decision table, as the markdown the document must contain, between its markers."""
    blocks: list[list[str]] = [
        _declaration(),
        [
            "**The state variables, defined once, and the polarity of TH-17b written into the "
            "value names.** Round 5's table spelled the F2-indicating state `not_satisfied` and "
            "the non-indicating one `satisfied`, which is the reverse of how TH-17b's own row "
            "reads; nothing mechanical held the token to the threshold, and a checker is equally "
            "green with the polarity flipped. The names now carry it, and `route()` is what makes "
            "them true — F2 is reachable from no other value of `th17b`."
        ],
        _variable_table(),
        [
            "**The rules.** Each is a conjunction of `variable = value`; there is no rule order to "
            "read, because no two guards can be satisfied at once."
        ],
        _rule_table(),
        _census(),
    ]
    return "\n\n".join("\n".join(block) for block in blocks)


if __name__ == "__main__":  # pragma: no cover - a convenience for regenerating §12
    print(render_routing_section())
