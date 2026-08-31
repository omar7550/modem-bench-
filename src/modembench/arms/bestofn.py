"""Budget-matched best-of-N one-shot arm and the best-of-n curve."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import random
import time
from typing import Any, Callable, Mapping, Sequence

from ..agent.accounting import TokenUsage, cost_usd
from ..agent.harness import OUTCOME_RUN_INVALID
from ..agent.provider import HEADLINE_MODEL
from ..conclusions import GATE_THRESHOLD
from ..gate import wilson_lower_bound
from .budget import SELECTOR_PREREGISTRATION, validate_campaign_n
# rerun_cap lives in ledger to avoid a circular import; re-exported as published surface.
from .ledger import (  # noqa: F401  (MIN_RERUN_CAP / RERUN_CAP_FRACTION are re-exports)
    LIST_RATE_REFERENCE_DATE,
    MIN_RERUN_CAP,
    RERUN_CAP_FRACTION,
    raw_tokens,
    rerun_cap,
)

BESTOFN_POLICY_VERSION = "modembench-bestofn-v1"

SELECTOR = "crc_pass"
SELECTION_TIMING = "harness-side, after all N attempts have completed"
TIE_BREAK_RULE = "lowest attempt index, among the attempts passing the selector"
NO_SELECTOR_PASS_RULE = (
    "when no valid attempt passes the selector, the lowest-index valid attempt is submitted; "
    "a deployable system must still return something"
)

BER_IS_NOT_DEPLOYABLE = (
    "aligned_ber is TRUTH. Every BER in this module — including the 1/64-quantized "
    "public.feedback.aligned_ber that feedback.forward_feedback hands the agent — comes from "
    "evaluator.evaluate_bits(bits, manifest, payload), which compares the receiver's bits "
    "against the HIDDEN payload. Quantizing truth coarsens it; it does not stop being truth. "
    "Agent-visible during benchmark feedback is NOT available to a deployed receiver: in the "
    "field there is no payload to compare against, so no BER exists to rank on, at any "
    "resolution. There is therefore NO honest way to use true BER as a deployable tie-break, "
    "and Attempt.selection_key must not contain one. crc_pass is admissible for the opposite "
    "reason: evaluator._crc_passes reads only the bits the receiver itself produced, so a "
    "fielded receiver can compute it unaided."
)

MEAN_BER_RULE = (
    "BER is REPORTED, never SELECTED ON. Two quantities are published and neither is an input "
    "to selection: (1) mean_ber_all_attempts, the mean over signals of the mean BER of that "
    "signal's N valid attempts — no selection is involved, so it is not a mixture and its "
    "estimator is not a function of the primary metric; (2) oracle_min_ber_at_n, the mean "
    "over signals of the LOWEST BER among the N attempts, which is TRUTH-AIDED and reported "
    "as an unachievable bound exactly as pass_at_n_upper_bound is for the primary metric. The "
    "BER of the SELECTED attempt is deliberately NOT reported: under a crc_pass-then-index "
    "selector it is a mixture whose weight is the primary metric, and the alternative — "
    "ranking ties on BER to make it a clean min-over-N — would make the arm truth-aided."
)

SECONDARY_METRIC_INVARIANT = (
    "Attempt.selection_key is (0 if crc_pass else 1, index) and contains no BER at any "
    "resolution, so no reported BER can reach the submitted result. Reported BER therefore "
    "cannot move packet_success, and the two reported BER quantities are computed by "
    "report_ber directly from the retained attempts rather than from Selection, which carries "
    "no BER field at all. That is structural, not a convention: there is nothing on Selection "
    "for a future caller to average."
)


class RerunCapExceeded(RuntimeError):
    """A signal could not produce N valid attempts within its re-run cap.

    An exception, not a short arm: a short arm silently lowers N on the failing signals.
    """


class AttemptRecordError(ValueError):
    """A run record that is not shaped like ``run_agent``'s return value."""


@dataclass(frozen=True)
class Attempt:
    """One issued attempt. Carries only the public capture id, so serializing cannot leak a sealed capture."""

    index: int
    run_id: str
    capture_id: str
    outcome_kind: str
    outcome_status: str
    outcome_reason: str | None
    crc_pass: bool
    packet_success: bool
    aligned_ber: float | None
    ber_source: str | None
    cost_usd: float
    raw_tokens: int
    output_tokens: int
    tool_calls: int
    is_rerun: bool = False
    wall_clock_s: float | None = None
    # public.feedback.aligned_ber, 1/64-quantized. Record-only; never a selection input.
    agent_visible_ber: float | None = None

    @property
    def valid(self) -> bool:
        """run_invalid attempts are re-run, never counted."""
        return self.outcome_kind != OUTCOME_RUN_INVALID

    @property
    def selection_key(self) -> tuple[int, int]:
        """(crc_pass, index), lower is better. No BER at any resolution; see BER_IS_NOT_DEPLOYABLE."""
        return (0 if self.crc_pass else 1, self.index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "run_id": self.run_id,
            "capture_id": self.capture_id,
            "outcome_kind": self.outcome_kind,
            "outcome_status": self.outcome_status,
            "outcome_reason": self.outcome_reason,
            "valid": self.valid,
            "crc_pass": self.crc_pass,
            "packet_success": self.packet_success,
            "aligned_ber": self.aligned_ber,
            "ber_source": self.ber_source,
            "agent_visible_ber": self.agent_visible_ber,
            "cost_usd": self.cost_usd,
            "raw_tokens": self.raw_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "is_rerun": self.is_rerun,
            "wall_clock_s": self.wall_clock_s,
        }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    # run_agent hands the outcome over both as a dict and as an Outcome dataclass; accept either.
    if hasattr(value, "as_dict") and not isinstance(value, Mapping):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise AttemptRecordError(f"run record field {name!r} is not a mapping")
    return dict(value)


def attempt_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
    is_rerun: bool = False,
    wall_clock_s: float | None = None,
) -> Attempt:
    """Reduce one run_agent record to an Attempt. Prefers the unrounded BER; records which was used."""
    if not isinstance(record, Mapping):
        raise AttemptRecordError("run record is not a mapping")
    public = _mapping(record.get("public"), "public")
    internal = _mapping(record.get("internal"), "internal")
    outcome = _mapping(public.get("outcome") or record.get("outcome"), "outcome")
    if not outcome:
        raise AttemptRecordError("run record carries no outcome")
    feedback = _mapping(public.get("feedback"), "public.feedback")
    usage = _mapping(internal.get("usage"), "internal.usage")
    cost = _mapping(internal.get("cost"), "internal.cost")

    forwarded = feedback.get("aligned_ber")
    agent_visible = (
        float(forwarded)
        if isinstance(forwarded, (int, float)) and not isinstance(forwarded, bool)
        else None
    )

    unrounded = internal.get("aligned_ber_unrounded")
    if isinstance(unrounded, (int, float)) and not isinstance(unrounded, bool):
        ber: float | None = float(unrounded)
        ber_source: str | None = "internal.aligned_ber_unrounded"
    elif agent_visible is not None:
        ber = agent_visible
        ber_source = "public.feedback.aligned_ber (quantized to the 1/64 grid)"
    else:
        ber = None
        ber_source = None

    prompt_size = usage.get("prompt_tokens_total_SIZE_NOT_COST")
    output_tokens = int(usage.get("output_tokens") or 0)
    total_raw = int(prompt_size or 0) + output_tokens
    return Attempt(
        index=int(index),
        run_id=str(record.get("run_id") or public.get("run_id") or ""),
        capture_id=str(public.get("capture_id") or ""),
        outcome_kind=str(outcome.get("kind")),
        outcome_status=str(outcome.get("status")),
        outcome_reason=(
            str(outcome["reason"]) if outcome.get("reason") is not None else None
        ),
        crc_pass=feedback.get("crc_pass") is True,
        packet_success=outcome.get("packet_success") is True,
        aligned_ber=ber,
        ber_source=ber_source,
        agent_visible_ber=agent_visible,
        cost_usd=float(cost.get("total_usd") or 0.0),
        raw_tokens=total_raw,
        output_tokens=output_tokens,
        tool_calls=len(public.get("tool_calls") or ()),
        is_rerun=bool(is_rerun),
        wall_clock_s=wall_clock_s,
    )


# --- selection --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Selection:
    """What the arm submits. Carries no BER by design; see MEAN_BER_RULE and report_ber."""

    selected_index: int
    selected_run_id: str
    selected_by: str
    selector_passed: bool
    packet_success: bool
    pass_at_n_upper_bound: bool
    first_attempt_packet_success: bool
    candidates_passing_selector: tuple[int, ...]
    n_considered: int
    # True when no valid attempt passed crc_pass, so the index alone chose.
    selector_was_uninformative: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector": SELECTOR,
            "selection_timing": SELECTION_TIMING,
            "tie_break": TIE_BREAK_RULE,
            "selected_index": self.selected_index,
            "selected_run_id": self.selected_run_id,
            "selected_by": self.selected_by,
            "selector_passed": self.selector_passed,
            "candidates_passing_selector": list(self.candidates_passing_selector),
            "n_considered": self.n_considered,
            "packet_success": self.packet_success,
            "pass_at_n_upper_bound": self.pass_at_n_upper_bound,
            "pass_at_n_label": (
                "UPPER BOUND: max over N attempts under a ground-truth selector. Not "
                "achievable by any deployable system; reported for comparison only."
            ),
            "first_valid_attempt_packet_success": self.first_attempt_packet_success,
            "selector_was_uninformative": self.selector_was_uninformative,
            "ber_is_not_reported_here": (
                "a selection carries no BER: the selected attempt's BER is a mixture whose "
                "weight is the primary metric. See report_ber for the two reported quantities."
            ),
            "selection_key": "(0 if crc_pass else 1, index) — contains no BER at any resolution",
            "mean_ber_rule": MEAN_BER_RULE,
            "secondary_metric_invariant": SECONDARY_METRIC_INVARIANT,
            "ber_is_not_deployable": BER_IS_NOT_DEPLOYABLE,
        }


def select_best(attempts: Sequence[Attempt]) -> Selection:
    """Pick one attempt on Attempt.selection_key. Called after all attempts complete, never by run_best_of_n."""
    valid = sorted((a for a in attempts if a.valid), key=lambda a: a.index)
    if not valid:
        raise ValueError("selection requires at least one valid attempt")
    passing = tuple(a.index for a in valid if a.crc_pass)
    chosen = min(valid, key=lambda a: a.selection_key)
    if passing:
        selected_by = f"{SELECTOR}, tie-break: {TIE_BREAK_RULE}"
    else:
        selected_by = NO_SELECTOR_PASS_RULE
    return Selection(
        selected_index=chosen.index,
        selected_run_id=chosen.run_id,
        selected_by=selected_by,
        selector_passed=chosen.crc_pass,
        packet_success=chosen.packet_success,
        pass_at_n_upper_bound=any(a.packet_success for a in valid),
        first_attempt_packet_success=valid[0].packet_success,
        candidates_passing_selector=passing,
        n_considered=len(valid),
        selector_was_uninformative=not passing,
    )


# --- running the arm ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BestOfNResult:
    """One signal's arm: every attempt issued, the N retained, and both spend figures."""

    signal_label: str
    n_requested: int
    attempts: tuple[Attempt, ...]
    rerun_cap: int

    @property
    def valid_attempts(self) -> tuple[Attempt, ...]:
        return tuple(a for a in self.attempts if a.valid)

    @property
    def attempts_issued(self) -> int:
        return len(self.attempts)

    @property
    def attempts_valid(self) -> int:
        return len(self.valid_attempts)

    @property
    def reruns(self) -> int:
        return sum(1 for a in self.attempts if a.is_rerun)

    @property
    def matched_spend_usd(self) -> float:
        """Spend charged against B: the N valid attempts only."""
        return sum(a.cost_usd for a in self.valid_attempts)

    @property
    def gross_spend_usd(self) -> float:
        """Every dollar billed, re-runs included."""
        return sum(a.cost_usd for a in self.attempts)

    @property
    def rerun_spend_usd(self) -> float:
        return self.gross_spend_usd - self.matched_spend_usd

    @property
    def matched_raw_tokens(self) -> int:
        return sum(a.raw_tokens for a in self.valid_attempts)

    @property
    def matched_output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.valid_attempts)

    @property
    def tool_calls(self) -> int:
        return sum(a.tool_calls for a in self.attempts)

    @property
    def wall_clock_s(self) -> float | None:
        measured = [a.wall_clock_s for a in self.attempts if a.wall_clock_s is not None]
        return sum(measured) if measured else None

    def selection(self) -> Selection:
        return select_best(self.valid_attempts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bestofn_policy_version": BESTOFN_POLICY_VERSION,
            "signal_label": self.signal_label,
            "n_requested": self.n_requested,
            "attempts_issued": self.attempts_issued,
            "attempts_valid": self.attempts_valid,
            "reruns": self.reruns,
            "rerun_cap": self.rerun_cap,
            "matched_spend_usd": self.matched_spend_usd,
            "gross_spend_usd": self.gross_spend_usd,
            "rerun_spend_usd": self.rerun_spend_usd,
            "matched_raw_tokens": self.matched_raw_tokens,
            "matched_output_tokens": self.matched_output_tokens,
            "tool_calls": self.tool_calls,
            "wall_clock_s": self.wall_clock_s,
            "mutual_blindness": (
                "each attempt is issued with nothing but its own index; no attempt sees "
                "another's existence, output or score"
            ),
            "selection": self.selection().as_dict(),
            "attempts": [a.as_dict() for a in self.attempts],
        }


def run_best_of_n(
    *,
    run_attempt: Callable[[int], Mapping[str, Any]],
    n: int,
    signal_label: str = "",
    max_reruns: int | None = None,
    enforce_campaign_ceiling: bool = True,
) -> BestOfNResult:
    """Issue attempts until exactly n valid ones exist, or raise RerunCapExceeded.

    run_attempt gets only the zero-based attempt index (mutual blindness) and returns a
    run_agent record.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}")
    if enforce_campaign_ceiling:
        validate_campaign_n(n)
    cap = rerun_cap(n) if max_reruns is None else int(max_reruns)
    if cap < 0:
        raise ValueError("the re-run cap must not be negative")

    attempts: list[Attempt] = []
    valid = 0
    issued = 0
    reruns = 0
    while valid < n:
        is_rerun = issued >= n
        if is_rerun and reruns >= cap:
            raise RerunCapExceeded(
                f"signal {signal_label!r} reached its re-run cap of {cap} with only {valid} "
                f"of {n} valid attempts after {issued} issued. Reporting a short arm would "
                "make effective N adaptive on exactly the signals where something is wrong."
            )
        started = time.perf_counter()
        record = run_attempt(issued)
        elapsed = time.perf_counter() - started
        attempt = attempt_from_record(
            record, index=issued, is_rerun=is_rerun, wall_clock_s=elapsed
        )
        attempts.append(attempt)
        issued += 1
        if is_rerun:
            reruns += 1
        if attempt.valid:
            valid += 1
    return BestOfNResult(
        signal_label=signal_label,
        n_requested=n,
        attempts=tuple(attempts),
        rerun_cap=cap,
    )


#: The only BER quantities this module reports; the selected attempt's BER is deliberately absent.
REPORTED_BER_ESTIMATORS: dict[str, str] = {
    "mean_ber_all_attempts": (
        "mean over signals of the mean aligned_ber of that signal's N valid attempts. No "
        "attempt is chosen, so this is a plain per-draw average: it is not a mixture, its "
        "weight is not the primary metric, and it is the same estimator whatever the arm's "
        "success rate. It is a property of the arm's DRAWS, not of what the arm submitted."
    ),
    "oracle_min_ber_at_n": (
        "mean over signals of the LOWEST aligned_ber among that signal's N valid attempts. "
        "TRUTH-AIDED LOWER BOUND: picking the minimum requires comparing against the hidden "
        "payload, which no deployable receiver can do. Reported for comparison only — the "
        "BER-side analogue of pass_at_n_upper_bound — and never an input to selection."
    ),
}


def report_ber(results: Sequence[BestOfNResult]) -> dict[str, Any]:
    """Report mean_ber_all_attempts and oracle_min_ber_at_n; see REPORTED_BER_ESTIMATORS."""
    per_signal_means: list[float] = []
    per_signal_minima: list[float] = []
    sources: set[str] = set()
    missing_signals = 0
    attempts_with_ber = 0
    attempts_without_ber = 0
    with_selector_pass = 0
    uninformative = 0
    for result in results:
        selection = result.selection()
        if selection.selector_passed:
            with_selector_pass += 1
        if selection.selector_was_uninformative:
            uninformative += 1
        values = []
        for attempt in result.valid_attempts:
            if attempt.aligned_ber is None:
                attempts_without_ber += 1
                continue
            attempts_with_ber += 1
            values.append(attempt.aligned_ber)
            if attempt.ber_source:
                sources.add(attempt.ber_source)
        if not values:
            missing_signals += 1
            continue
        per_signal_means.append(sum(values) / len(values))
        per_signal_minima.append(min(values))
    signals = len(results)
    measured = len(per_signal_means)
    return {
        "mean_ber_all_attempts": (
            sum(per_signal_means) / measured if measured else None
        ),
        "mean_ber_all_attempts_estimator": REPORTED_BER_ESTIMATORS["mean_ber_all_attempts"],
        "oracle_min_ber_at_n": (
            sum(per_signal_minima) / measured if measured else None
        ),
        "oracle_min_ber_at_n_estimator": REPORTED_BER_ESTIMATORS["oracle_min_ber_at_n"],
        "oracle_min_ber_at_n_label": (
            "TRUTH-AIDED: ranks attempts on a BER computed against the hidden payload. Not "
            "achievable by any deployable receiver; reported for comparison only, never used "
            "to choose the submitted attempt."
        ),
        "selection_inputs": (
            "crc_pass, then the attempt index (Attempt.selection_key). No BER, quantized or "
            "unrounded, is an input to selection."
        ),
        "selected_attempt_ber_reported": False,
        "selected_attempt_ber_omitted_because": (
            "under a crc_pass-then-index selector the selected attempt's BER is a mixture of "
            "a CRC-selected draw and an index-0 draw, and the mixture weight is the primary "
            "metric, so it cannot corroborate the primary metric. Making it a clean "
            "min-over-N requires ranking ties on truth, which is not deployable."
        ),
        "signals": signals,
        "signals_with_ber": measured,
        "signals_without_ber": missing_signals,
        "attempts_with_ber": attempts_with_ber,
        "attempts_without_ber": attempts_without_ber,
        "ber_sources": sorted(sources),
        "signals_with_selector_pass": with_selector_pass,
        "signals_without_selector_pass": signals - with_selector_pass,
        "selector_pass_fraction": (with_selector_pass / signals) if signals else None,
        "signals_where_selector_was_uninformative": uninformative,
        "rule": MEAN_BER_RULE,
        "ber_is_not_deployable": BER_IS_NOT_DEPLOYABLE,
    }


def mean_selected_ber(results: Sequence[BestOfNResult]) -> dict[str, Any]:
    """Deprecated alias for report_ber; kept because cli.py and docs/pre-registration.md name it."""
    payload = report_ber(results)
    payload["name_is_historical"] = (
        "mean_selected_ber is a legacy alias for report_ber. There is no mean of SELECTED "
        "BERs in this payload and there will not be one: see selected_attempt_ber_omitted_"
        "because."
    )
    return payload


# --- the best-of-n curve -----------------------------------------------------------------------
def _selected_index_in_subset(subset: Sequence[int], crc: Sequence[bool]) -> int:
    """The shipped selector on a subset: first CRC pass by index, else lowest index.

    Must match select_best on the subset; tests assert the agreement over every subset.
    """
    ordered = sorted(subset)
    for index in ordered:
        if crc[index]:
            return index
    return ordered[0]


def exact_best_of_n_success(
    crc: Sequence[bool], packet: Sequence[bool], n: int
) -> float:
    """P(the selected attempt of a uniformly random n-subset recovered the packet), exact.

    Counting: a passing attempt a is selected in C(M - 1 - passing_below(a), n - 1) subsets;
    a failing attempt m in C(failing_above(m), n - 1). subset_bootstrap_success converges here.
    """
    total = len(crc)
    if total != len(packet):
        raise ValueError("crc and packet_success sequences must be the same length")
    if not 1 <= n <= total:
        raise ValueError(f"n must lie in 1..{total}; got {n}")
    denominator = math.comb(total, n)
    numerator = 0
    passing_below = 0
    failing_indices = [i for i in range(total) if not crc[i]]
    for index in range(total):
        if crc[index]:
            subsets = math.comb(total - 1 - passing_below, n - 1)
            passing_below += 1
        else:
            failing_above = sum(1 for j in failing_indices if j > index)
            subsets = math.comb(failing_above, n - 1)
        if packet[index]:
            numerator += subsets
    return numerator / denominator


def exact_pass_at_n(packet: Sequence[bool], n: int) -> float:
    """P(at least one of n drawn attempts recovered the packet): the upper bound."""
    total = len(packet)
    if not 1 <= n <= total:
        raise ValueError(f"n must lie in 1..{total}; got {n}")
    successes = sum(1 for value in packet if value)
    return 1.0 - (math.comb(total - successes, n) / math.comb(total, n))


def subset_bootstrap_success(
    crc: Sequence[bool],
    packet: Sequence[bool],
    n: int,
    *,
    draws: int = 4000,
    seed: int = 20260806,
) -> float:
    """Seeded resampling of n-subsets; kept as an independent check on exact_best_of_n_success."""
    total = len(crc)
    if not 1 <= n <= total:
        raise ValueError(f"n must lie in 1..{total}; got {n}")
    rng = random.Random(seed)
    hits = 0
    for _ in range(int(draws)):
        subset = rng.sample(range(total), n)
        hits += bool(packet[_selected_index_in_subset(subset, crc)])
    return hits / int(draws)


def _attempts_of(signal: Any) -> tuple[Attempt, ...]:
    if isinstance(signal, BestOfNResult):
        return signal.valid_attempts
    attempts = tuple(signal)
    if not all(isinstance(a, Attempt) for a in attempts):
        raise TypeError("a signal must be a BestOfNResult or a sequence of Attempt")
    return tuple(a for a in sorted(attempts, key=lambda a: a.index) if a.valid)


REPLICATION_UNIT = "signal (capture), not signal x replicate"
CLUSTER_DESIGN_EFFECT_NOTE = (
    "replicates of one signal are not independent draws: they share the capture, the "
    "impairment realisation and the difficulty. The bootstrap resamples SIGNAL CLUSTERS with "
    "replacement, taking every replicate of a drawn signal, and never resamples attempts or "
    "replicates on their own. Resampling the signal x replicate cells independently would "
    "understate the interval by sqrt(1 + (replicates - 1) * intra-cluster correlation)."
)


def _cluster_key_of(unit: Any, attempts: Sequence[Attempt], position: int) -> str:
    """Cluster key: capture_id, not signal_label, which callers may make unique per replicate."""
    captures = {a.capture_id for a in attempts if a.capture_id}
    if len(captures) == 1:
        return next(iter(captures))
    if isinstance(unit, BestOfNResult) and unit.signal_label:
        return str(unit.signal_label)
    # No usable identity: its own cluster.
    return f"__unkeyed_unit_{position}__"


def best_of_n_curve(
    units: Sequence[Any],
    *,
    cluster_keys: Sequence[Any] | None = None,
    max_n: int | None = None,
    bootstrap_draws: int = 2000,
    seed: int = 20260806,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Best-of-n success for every n in 1..N, from attempts already paid for.

    units are (signal, replicate) cells, one BestOfNResult each. The interval is a cluster
    bootstrap over signals (see CLUSTER_DESIGN_EFFECT_NOTE); cluster_keys overrides the
    default capture_id key.
    """
    signals = units
    resolved = [(_attempts_of(unit), unit) for unit in signals]
    if cluster_keys is not None:
        if len(cluster_keys) != len(resolved):
            raise ValueError(
                f"cluster_keys has {len(cluster_keys)} entries for {len(resolved)} units"
            )
        keys = [
            str(key)
            for (attempts, _), key in zip(resolved, cluster_keys)
            if attempts
        ]
        key_source = "caller-supplied cluster_keys"
    else:
        keys = [
            _cluster_key_of(unit, attempts, position)
            for position, (attempts, unit) in enumerate(resolved)
            if attempts
        ]
        key_source = "Attempt.capture_id (the campaign's signal identity)"
    per_signal = [attempts for attempts, _ in resolved if attempts]
    if not per_signal:
        raise ValueError("the curve needs at least one signal with a valid attempt")
    # Uneven attempt counts: define the curve on the common support and report the truncation.
    available = min(len(attempts) for attempts in per_signal)
    truncated = sum(1 for attempts in per_signal if len(attempts) > available)
    ceiling = available if max_n is None else min(int(max_n), available)
    if ceiling < 1:
        raise ValueError("no signal retained an attempt")

    crc = [[a.crc_pass for a in attempts] for attempts in per_signal]
    packet = [[a.packet_success for a in attempts] for attempts in per_signal]
    unit_count = len(per_signal)

    # Order preserved so the curve is deterministic under a fixed seed.
    clusters: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        clusters.setdefault(key, []).append(position)
    cluster_members = list(clusters.values())
    cluster_count = len(cluster_members)
    sizes = [len(members) for members in cluster_members]
    clustered = max(sizes) > 1

    rng = random.Random(seed)
    lower_quantile = (1.0 - float(confidence)) / 2.0
    upper_quantile = 1.0 - lower_quantile

    points: dict[int, dict[str, Any]] = {}
    for n in range(1, ceiling + 1):
        per_signal_success = [
            exact_best_of_n_success(crc[i][:available], packet[i][:available], n)
            for i in range(unit_count)
        ]
        per_signal_upper = [
            exact_pass_at_n(packet[i][:available], n) for i in range(unit_count)
        ]
        rate = sum(per_signal_success) / unit_count
        cluster_sums = [
            sum(per_signal_success[i] for i in members) for members in cluster_members
        ]
        draws: list[float] = []
        for _ in range(int(bootstrap_draws)):
            total = 0.0
            cells = 0
            for _ in range(cluster_count):
                pick = rng.randrange(cluster_count)
                total += cluster_sums[pick]
                cells += sizes[pick]
            draws.append(total / cells)
        draws.sort()
        # Wilson needs an integer count over independent trials; report None with the reason
        # where the count would be fabricated (fractional cells, or clustered cells).
        realized = all(value in (0.0, 1.0) for value in per_signal_success)
        if not realized:
            wilson: float | None = None
            wilson_reason: str | None = (
                "at n below the retained attempt count the per-cell values are probabilities; "
                "rounding them into a success count would manufacture a binomial that was "
                "never observed"
            )
        elif clustered:
            wilson = None
            wilson_reason = (
                f"{unit_count} cells fall in {cluster_count} signal clusters, so they are not "
                "independent binomial trials; read the cluster bootstrap interval instead"
            )
        else:
            wilson = wilson_lower_bound(int(round(sum(per_signal_success))), unit_count)
            wilson_reason = None
        points[n] = {
            "n": n,
            "success_rate": rate,
            "pass_at_n_upper_bound": sum(per_signal_upper) / unit_count,
            "ci_low": _quantile(draws, lower_quantile),
            "ci_high": _quantile(draws, upper_quantile),
            "ci_width": _quantile(draws, upper_quantile) - _quantile(draws, lower_quantile),
            "wilson_lower_bound": wilson,
            "wilson_lower_bound_unavailable_reason": wilson_reason,
        }
    return {
        "bestofn_policy_version": BESTOFN_POLICY_VERSION,
        "selector": SELECTOR,
        "selection_timing": SELECTION_TIMING,
        "units": unit_count,
        "signals": cluster_count,
        "signals_are_clusters_of_units": clustered,
        "cluster_key_source": key_source,
        "min_units_per_signal": min(sizes),
        "max_units_per_signal": max(sizes),
        "mean_units_per_signal": unit_count / cluster_count,
        "replication_unit": REPLICATION_UNIT,
        "cluster_design_effect_note": CLUSTER_DESIGN_EFFECT_NOTE,
        "attempts_per_signal": available,
        "signals_truncated_to_common_support": truncated,
        "max_n": ceiling,
        "additional_spend_usd": 0.0,
        "estimator": (
            "exact over n-subsets of the retained attempts; the cluster bootstrap resamples "
            "SIGNAL CLUSTERS with replacement, taking every replicate of a drawn signal, and "
            "never resamples attempts or replicates on their own"
        ),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(seed),
        "confidence": float(confidence),
        "naive_single_call_rate": points[1]["success_rate"],
        "points": points,
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no bootstrap draws")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


# --- the currency sensitivity --------------------------------------------------------------
# Measured N_raw / N_cost by iterative round count; recompute with raw_to_cost_ratio.
RAW_TO_COST_RATIO_BY_ROUNDS: dict[int, float] = {3: 1.31, 6: 1.65, 12: 2.06}


def raw_to_cost_ratio(
    one_shot: TokenUsage,
    iterative: TokenUsage,
    *,
    model: str = HEADLINE_MODEL,
    on: date | None = None,
) -> float:
    """N_raw / N_cost for a pair of run shapes.

    N_cost = B / cost(one_shot); N_raw = B * tokens(iter) / (cost(iter) * tokens(one)).
    B cancels: the ratio is (cost_one/tokens_one) / (cost_iter/tokens_iter).
    """
    priced_on = on or LIST_RATE_REFERENCE_DATE
    one_tokens = raw_tokens(one_shot)
    iter_tokens = raw_tokens(iterative)
    if one_tokens <= 0 or iter_tokens <= 0:
        raise ValueError("both run shapes must move a positive number of tokens")
    one_price = cost_usd(one_shot, model=model, on=priced_on) / one_tokens
    iter_price = cost_usd(iterative, model=model, on=priced_on) / iter_tokens
    if iter_price <= 0:
        raise ValueError("the iterative shape prices at zero dollars per token")
    return one_price / iter_price


def n_raw_from_n_cost(n_cost: int, *, rounds: int, ratio: float | None = None) -> int:
    """N_raw at a given iterative round count. Floored: attempts are whole."""
    if ratio is None:
        try:
            ratio = RAW_TO_COST_RATIO_BY_ROUNDS[int(rounds)]
        except KeyError:
            raise ValueError(
                f"no measured raw/cost ratio for {rounds} rounds; measured rounds are "
                f"{sorted(RAW_TO_COST_RATIO_BY_ROUNDS)}"
            ) from None
    return int(math.floor(int(n_cost) * float(ratio)))


def gate_sensitivity(
    curve: Mapping[str, Any],
    *,
    n_cost: int,
    n_raw: int,
    verdict_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the gate verdict at N_cost and N_raw.

    verdict_fn is injected; the gate analysis owns the actual statistical test. n_raw above
    max_n is reported not computable, never clamped.
    """
    points = curve.get("points") or {}
    max_n = int(curve.get("max_n") or 0)

    def _read(n: int) -> dict[str, Any]:
        # JSON round-trips turn the integer keys into strings; accept both.
        entry = points.get(n, points.get(str(n)))
        if n < 1:
            return {"n": n, "computable": False, "reason": "n below 1"}
        if n > max_n or entry is None:
            return {
                "n": n,
                "computable": False,
                "reason": (
                    f"only {max_n} attempts per signal were retained, so best-of-{n} cannot "
                    "be read off this curve without buying more attempts"
                ),
            }
        point = dict(entry)
        return {
            "n": n,
            "computable": True,
            "point": point,
            "verdict": dict(verdict_fn(point)),
        }

    cost_side = _read(int(n_cost))
    raw_side = _read(int(n_raw))
    both = cost_side.get("computable") and raw_side.get("computable")
    delta = (
        raw_side["point"]["success_rate"] - cost_side["point"]["success_rate"]
        if both
        else None
    )
    return {
        "pre_registered": True,
        "question": (
            "does the cost-vs-raw-token matching currency move the gate verdict? The design "
            "measured the baseline shift at +7.4 / +14.4 / +22.1 points at 3 / 6 / 12 "
            f"iterative rounds, against a {GATE_THRESHOLD * 100:.0f}-point gate."
        ),
        "n_cost": cost_side,
        "n_raw": raw_side,
        "one_shot_delta_points": (delta * 100.0) if delta is not None else None,
        "verdicts_agree": (
            (cost_side["verdict"].get("passed") == raw_side["verdict"].get("passed"))
            if both
            else None
        ),
        "currency_is_material": (
            (cost_side["verdict"].get("passed") != raw_side["verdict"].get("passed"))
            if both
            else None
        ),
    }


def margin_verdict(
    iterative_success_rate: float, *, threshold: float = GATE_THRESHOLD
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Provisional verdict for the sensitivity harness: point estimates vs a fixed margin, not the gate's test."""

    def verdict(point: Mapping[str, Any]) -> dict[str, Any]:
        one_shot = float(point["success_rate"])
        difference = float(iterative_success_rate) - one_shot
        return {
            "provisional": True,
            "not_the_gate_test": (
                "point estimates against a fixed margin; the gate verdict is computed on "
                "the CI lower bound of a paired comparison"
            ),
            "iterative_success_rate": float(iterative_success_rate),
            "one_shot_success_rate": one_shot,
            "difference": difference,
            "threshold": float(threshold),
            "passed": difference >= float(threshold),
        }

    return verdict


def selector_preregistration() -> dict[str, Any]:
    """The selector's pre-registration, as pinned into the frozen budget config."""
    return dict(SELECTOR_PREREGISTRATION)
