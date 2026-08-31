"""The budget ledger: total dollars in, the affordable N out. See docs/budget-ledger.md."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import math
from typing import Any, Iterable, Sequence

from ..agent.accounting import (
    ACCOUNTING_POLICY_VERSION,
    PRICE_TABLE_DATE,
    PROJECTED_RUN_USAGE,
    SEALED_ARMS,
    SEALED_ATTEMPT_UNITS_PER_ACCESS,
    SEALED_REPLICATES,
    SEALED_SIGNALS as ACCOUNTING_SEALED_SIGNALS,
    SONNET_5_INTRO_THROUGH,
    TokenUsage,
    cost_usd,
    rates_for,
    sealed_campaign_attempts,
)
from ..agent.provider import HEADLINE_MODEL

LEDGER_POLICY_VERSION = "modembench-budget-ledger-v1"


class LedgerError(RuntimeError):
    """The ledger cannot be satisfied: an unpriceable line, or a budget that buys nothing."""


# --- the one constant that sizes everything -------------------------------------------------
# List-price USD stays the matching unit even on subscription accounts; see
# docs/n-r-rederivation.md for the derivation ($1,100 is the smallest hundred affording N=5).
TOTAL_AVAILABLE_USD = 1_100.0

# 10% ON TOP of the headline campaign: residual = TOTAL / (1 + fraction), not TOTAL * 0.9.
CROSS_MODEL_SLICE_FRACTION = 0.10

# --- campaign shape -------------------------------------------------------------------------
DEV_SIGNALS = 40  # the dev split
SEALED_SIGNALS = ACCOUNTING_SEALED_SIGNALS  # 60 sealed signals per access
ARM_COUNT = SEALED_ARMS  # iterative and budget-matched one-shot
# The dev gate's own seed count; equals SEALED_REPLICATES and is checked for agreement,
# not collapsed into it.
GATE_REPLICATES = 3
SEALED_ACCESSES = 2  # two counted accesses, ~2026-11-08 and ~2026-12-23

T7_CALIBRATION_SETTINGS = 5

# The dev gate is a paired comparison: both arms run, doubling the line.
T9_GATE_ARMS = 2

# --- the pivot arm the only refuting conclusion rests on -------------------------------------
# The T12 typed-DSP-graph arm: 40 dev signals x 3 replicates, one arm (t9-dev-gate's one-shot
# arm is its comparator). Unfunded, §12's F3 (t12=failed) is unreachable, so the flag is
# named here and priced by t12_pivot_funding_report().
FUND_T12_PIVOT_ARM = True
T12_PIVOT_ARMS = 1
# Billed after the dev gate, before the first sealed access; the whole window is one list
# period, asserted in tests/test_arms.py.
T12_PIVOT_DATE = date(2026, 10, 6)

# 50% off every sealed-campaign component; dev iteration stays unbatched.
BATCH_SEALED_CAMPAIGN = True

# --- re-runs are real money, and B is not where they are paid for --------------------------
# Re-run spend is excluded from B (a matching quantity) but provisioned in the ledger (money):
# obtaining N valid attempts at invalid rate p costs N / (1 - p) issued attempts, so the
# provision is a multiplier. A hard spend cap instead would truncate a counted sealed access.
INVALID_RATE_PROVISION = 0.10

# Bounds the tail the provision does not cover; worst case is (N + rerun_cap(N)) / N times
# matched spend. Lives here, not in bestofn, to avoid a circular import.
# Written as a denominator, not 0.5: test_the_ledger_holds_no_rate_of_its_own forbids float
# literals matching published per-MTok rates, and 0.50 is Opus 5's cache-read rate.
RERUN_CAP_DENOMINATOR = 2  # one re-run allowed per this many attempts
MIN_RERUN_CAP = 2
RERUN_CAP_FRACTION = 1 / RERUN_CAP_DENOMINATOR


def rerun_cap(n: int, *, fraction: float = RERUN_CAP_FRACTION, floor: int = MIN_RERUN_CAP) -> int:
    """The re-run allowance for a signal at ``n`` attempts."""
    return max(int(floor), int(math.ceil(int(n) * float(fraction))))

# --- when each line is billed ----------------------------------------------------------------
# Sonnet 5's introductory rate expires 2026-08-31; dates are estimates, and
# allocate(pricing="list") prices the slip case.
T6_DEV_SWEEP_DATE = PRICE_TABLE_DATE
T7_CALIBRATION_DATE = date(2026, 8, 20)
T9_DEV_GATE_DATE = date(2026, 9, 15)
SEALED_ACCESS_DATES: tuple[date, ...] = (date(2026, 11, 8), date(2026, 12, 23))
# B is quoted at list: both sealed accesses land after the intro-rate expiry.
LIST_RATE_REFERENCE_DATE = T9_DEV_GATE_DATE

SCHEDULE_MAX_N = 32  # covers the adversarial sizings quoted at N=11 and N=31


def raw_tokens(usage: TokenUsage) -> int:
    """Total tokens moved: prompt (all components) plus output. A size, never a cost basis."""
    return usage.prompt_tokens_total + usage.output_tokens


@dataclass(frozen=True)
class LedgerLine:
    """One claim on the budget, in units of one attempt on one signal for one arm.

    attempt_units x N x unit_usd gives dollars; e.g. 40 signals x 3 replicates x 2 arms = 240.
    """

    name: str
    ticket: str
    signals: int
    replicates: int = 1
    arms: int = 1
    settings: int = 1
    priced_on: date = PRICE_TABLE_DATE
    batch: bool = False
    note: str = ""

    @property
    def attempt_units(self) -> int:
        return int(self.signals) * int(self.replicates) * int(self.arms) * int(self.settings)

    def unit_usd(self, *, model: str, attempt_usage: TokenUsage, on: date | None = None) -> float:
        """Dollars for one attempt on this line, at this line's date and batch setting."""
        return cost_usd(
            attempt_usage, model=model, on=on or self.priced_on, batch=self.batch
        )

    def usd_per_n(
        self, *, model: str, attempt_usage: TokenUsage, on: date | None = None
    ) -> float:
        """Dollars this line costs for each unit of ``N``."""
        return self.attempt_units * self.unit_usd(
            model=model, attempt_usage=attempt_usage, on=on
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ticket": self.ticket,
            "signals": self.signals,
            "replicates": self.replicates,
            "arms": self.arms,
            "settings": self.settings,
            "attempt_units": self.attempt_units,
            "priced_on": self.priced_on.isoformat(),
            "batch": self.batch,
            "note": self.note,
        }


def campaign_lines(
    *,
    calibration_settings: int = T7_CALIBRATION_SETTINGS,
    gate_arms: int = T9_GATE_ARMS,
    sealed_accesses: int = SEALED_ACCESSES,
    batch_sealed: bool = BATCH_SEALED_CAMPAIGN,
    fund_t12_pivot: bool = FUND_T12_PIVOT_ARM,
) -> tuple[LedgerLine, ...]:
    """The claims on the budget, in the order the campaign incurs them."""
    if sealed_accesses > len(SEALED_ACCESS_DATES):
        raise LedgerError(
            f"{sealed_accesses} sealed accesses requested but only "
            f"{len(SEALED_ACCESS_DATES)} access dates are budgeted"
        )
    lines: list[LedgerLine] = [
        LedgerLine(
            name="t6-dev-sweep",
            ticket="T6",
            signals=DEV_SIGNALS,
            priced_on=T6_DEV_SWEEP_DATE,
            note="40 x N: this ticket's own dev sweep, which measures the run shape",
        ),
        LedgerLine(
            name="t7-calibration",
            ticket="T7",
            signals=DEV_SIGNALS,
            settings=int(calibration_settings),
            priced_on=T7_CALIBRATION_DATE,
            note="S settings x 40 x N: the impairment sweep that lands 25-40% one-shot",
        ),
        LedgerLine(
            name="t9-dev-gate",
            ticket="T9",
            signals=DEV_SIGNALS,
            replicates=GATE_REPLICATES,
            arms=int(gate_arms),
            priced_on=T9_DEV_GATE_DATE,
            note=(
                "40 x 3 x N per arm; the dev gate runs both arms, which doubles it"
            ),
        ),
    ]
    if fund_t12_pivot:
        lines.append(
            LedgerLine(
                name="t12-pivot-arm",
                ticket="T12",
                signals=DEV_SIGNALS,
                replicates=GATE_REPLICATES,
                arms=T12_PIVOT_ARMS,
                priced_on=T12_PIVOT_DATE,
                note=(
                    "40 x 3 x N: T12's typed DSP-graph arm at matched budget on the same "
                    "signals. The line the refuting conclusion requires and, until this "
                    "revision, the line no campaign funded"
                ),
            )
        )
    for index in range(int(sealed_accesses)):
        lines.append(
            LedgerLine(
                name=f"sealed-access-{index + 1}",
                ticket="T9",
                signals=SEALED_SIGNALS,
                replicates=GATE_REPLICATES,
                arms=ARM_COUNT,
                priced_on=SEALED_ACCESS_DATES[index],
                batch=bool(batch_sealed),
                note="60 x 3 x N x 2 arms: one counted sealed access",
            )
        )
    return tuple(lines)


def funded_arms(lines: Sequence[LedgerLine] | None = None) -> frozenset[str]:
    """The tickets a plan buys evidence from, upper-cased. Derived from LedgerLine.ticket."""
    plan = campaign_lines() if lines is None else lines
    return frozenset(line.ticket.upper() for line in plan)


def fundable_arms() -> frozenset[str]:
    """Every ticket some setting of this module's flags can fund. A vocabulary, not a plan."""
    return funded_arms(campaign_lines(fund_t12_pivot=True))


@dataclass(frozen=True)
class LineAllocation:
    line: LedgerLine
    unit_usd: float
    usd_per_n: float
    usd_at_n: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.line.as_dict(),
            "unit_usd": self.unit_usd,
            "usd_per_n": self.usd_per_n,
            "usd_at_n": self.usd_at_n,
        }


@dataclass(frozen=True)
class LedgerAllocation:
    """The full allocation and the N the residual affords.

    affordable_n may be 0 here; derive_budget is where a sub-1 N becomes a hard error.
    """

    model: str
    attempt_usage: TokenUsage
    pricing: str
    total_available_usd: float
    cross_model_slice_fraction: float
    cross_model_reserve_usd: float
    residual_usd: float
    lines: tuple[LineAllocation, ...]
    usd_per_n_matched: float
    invalid_rate_provision: float
    usd_per_n: float
    affordable_n: int
    reference_unit_usd: float

    @property
    def attempt_units(self) -> int:
        return sum(entry.line.attempt_units for entry in self.lines)

    @property
    def rerun_provision_usd_per_n(self) -> float:
        """What N valid attempts cost beyond N issued attempts."""
        return self.usd_per_n - self.usd_per_n_matched

    @property
    def worst_case_rerun_multiple(self) -> float:
        """Gross-to-matched spend if every signal exhausted its re-run cap. A ceiling, not an expectation."""
        n = max(1, self.affordable_n)
        return (n + rerun_cap(n)) / n

    @property
    def worst_case_gross_usd_at_n(self) -> float:
        """Campaign dollars if every signal exhausted its re-run cap, slice included."""
        return (
            self.usd_per_n_matched
            * self.affordable_n
            * self.worst_case_rerun_multiple
            * (1.0 + self.cross_model_slice_fraction)
        )

    @property
    def committed_usd(self) -> float:
        """Gross campaign dollars committed at affordable_n, before the slice (re-runs included)."""
        return self.usd_per_n * self.affordable_n

    @property
    def total_committed_usd(self) -> float:
        """Committed campaign dollars plus the cross-model slice they earn."""
        return self.committed_usd * (1.0 + self.cross_model_slice_fraction)

    @property
    def residual_headroom_usd(self) -> float:
        return self.residual_usd - self.committed_usd

    @property
    def budget_per_signal_usd(self) -> float:
        """B: per-signal, per-arm matched budget. B = N * reference_unit_usd by construction."""
        return self.affordable_n * self.reference_unit_usd

    @property
    def uncommitted_budget_per_signal_usd(self) -> float:
        """B before the floor to whole attempts; priced through the same gross usd_per_n."""
        if self.usd_per_n <= 0:
            return 0.0
        return (self.residual_usd / self.usd_per_n) * self.reference_unit_usd

    def usd_required_at_n(self, n: int) -> float:
        """Total dollars, campaign plus cross-model slice, to run at n attempts."""
        return self.usd_per_n * int(n) * (1.0 + self.cross_model_slice_fraction)

    def schedule(self, max_n: int = SCHEDULE_MAX_N) -> dict[int, float]:
        """n -> total dollars required."""
        return {n: self.usd_required_at_n(n) for n in range(1, int(max_n) + 1)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "ledger_policy_version": LEDGER_POLICY_VERSION,
            "accounting_policy_version": ACCOUNTING_POLICY_VERSION,
            "price_table_date": PRICE_TABLE_DATE.isoformat(),
            "sonnet_5_intro_through": SONNET_5_INTRO_THROUGH.isoformat(),
            "model": self.model,
            "pricing": self.pricing,
            "attempt_usage": self.attempt_usage.as_dict(),
            "attempt_raw_tokens": raw_tokens(self.attempt_usage),
            "attempt_output_tokens": self.attempt_usage.output_tokens,
            "total_available_usd": self.total_available_usd,
            "cross_model_slice_fraction": self.cross_model_slice_fraction,
            "cross_model_reserve_usd": self.cross_model_reserve_usd,
            "residual_usd": self.residual_usd,
            "attempt_units": self.attempt_units,
            "usd_per_n_matched": self.usd_per_n_matched,
            "invalid_rate_provision": self.invalid_rate_provision,
            "rerun_provision_usd_per_n": self.rerun_provision_usd_per_n,
            "usd_per_n": self.usd_per_n,
            "usd_per_n_basis": (
                "GROSS: N valid attempts plus the re-runs needed to obtain them at the "
                "provisioned invalid rate. B excludes re-run spend because B is a matching "
                "quantity between the arms; the ledger includes it because it is money."
            ),
            "worst_case_rerun_multiple": self.worst_case_rerun_multiple,
            "worst_case_gross_usd_at_n": self.worst_case_gross_usd_at_n,
            "affordable_n": self.affordable_n,
            "reference_unit_usd": self.reference_unit_usd,
            "reference_date": LIST_RATE_REFERENCE_DATE.isoformat(),
            "budget_per_signal_usd": self.budget_per_signal_usd,
            "uncommitted_budget_per_signal_usd": self.uncommitted_budget_per_signal_usd,
            "committed_usd": self.committed_usd,
            "total_committed_usd": self.total_committed_usd,
            "residual_headroom_usd": self.residual_headroom_usd,
            "lines": [entry.as_dict() for entry in self.lines],
        }


PRICING_MODES = ("dated", "list")


def allocate(
    *,
    total_available_usd: float = TOTAL_AVAILABLE_USD,
    model: str = HEADLINE_MODEL,
    attempt_usage: TokenUsage | None = None,
    lines: Sequence[LedgerLine] | None = None,
    pricing: str = "dated",
    cross_model_slice_fraction: float = CROSS_MODEL_SLICE_FRACTION,
    invalid_rate_provision: float = INVALID_RATE_PROVISION,
) -> LedgerAllocation:
    """Allocate total_available_usd across the campaign and return the affordable N.

    pricing="dated" prices each line on its own date (what will be billed); "list" prices
    everything at LIST_RATE_REFERENCE_DATE (the schedule-slip case, always the smaller N).
    """
    if pricing not in PRICING_MODES:
        raise LedgerError(f"unknown pricing mode {pricing!r}; expected one of {PRICING_MODES}")
    if not isinstance(total_available_usd, (int, float)) or isinstance(
        total_available_usd, bool
    ):
        raise LedgerError("total_available_usd must be a number")
    if total_available_usd <= 0:
        raise LedgerError("total_available_usd must be positive")
    if cross_model_slice_fraction < 0:
        raise LedgerError("cross_model_slice_fraction must not be negative")
    if not 0.0 <= invalid_rate_provision < 1.0:
        raise LedgerError(
            "invalid_rate_provision is a probability in [0, 1); at 1 every issued attempt is "
            "invalid and no finite budget buys a valid one"
        )
    usage = attempt_usage if attempt_usage is not None else PROJECTED_RUN_USAGE
    plan = tuple(lines) if lines is not None else campaign_lines()
    if not plan:
        raise LedgerError("a ledger with no lines cannot derive a budget")
    override = LIST_RATE_REFERENCE_DATE if pricing == "list" else None
    # rates_for raises on an unpriceable model or date, before any arithmetic.
    for line in plan:
        rates_for(model, override or line.priced_on)
    reference_unit_usd = cost_usd(usage, model=model, on=LIST_RATE_REFERENCE_DATE)

    entries: list[LineAllocation] = []
    usd_per_n_matched = 0.0
    for line in plan:
        unit = line.unit_usd(model=model, attempt_usage=usage, on=override)
        per_n = line.attempt_units * unit
        usd_per_n_matched += per_n
        entries.append(LineAllocation(line=line, unit_usd=unit, usd_per_n=per_n, usd_at_n=0.0))
    if usd_per_n_matched <= 0:
        raise LedgerError("the campaign prices at zero dollars per attempt; refusing to plan")

    # N valid attempts at invalid rate p cost N / (1 - p) issued: a campaign-wide multiplier.
    provision = float(invalid_rate_provision)
    usd_per_n = usd_per_n_matched / (1.0 - provision)

    residual = float(total_available_usd) / (1.0 + float(cross_model_slice_fraction))
    reserve = float(total_available_usd) - residual
    affordable = int(math.floor(residual / usd_per_n))
    # Line-level dollars stay matched; the gross figure is committed_usd.
    entries = [replace(entry, usd_at_n=entry.usd_per_n * affordable) for entry in entries]
    return LedgerAllocation(
        model=model,
        attempt_usage=usage,
        pricing=pricing,
        total_available_usd=float(total_available_usd),
        cross_model_slice_fraction=float(cross_model_slice_fraction),
        cross_model_reserve_usd=reserve,
        residual_usd=residual,
        lines=tuple(entries),
        usd_per_n_matched=usd_per_n_matched,
        invalid_rate_provision=provision,
        usd_per_n=usd_per_n,
        affordable_n=affordable,
        reference_unit_usd=reference_unit_usd,
    )


def with_output_tokens(usage: TokenUsage, output_tokens: int) -> TokenUsage:
    """The same prompt shape at a different output estimate (the axis N is most sensitive to)."""
    return TokenUsage(
        input_tokens=usage.input_tokens,
        cache_creation_5m_tokens=usage.cache_creation_5m_tokens,
        cache_creation_1h_tokens=usage.cache_creation_1h_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=int(output_tokens),
    )


def output_sensitivity(
    output_token_counts: Iterable[int],
    *,
    total_available_usd: float = TOTAL_AVAILABLE_USD,
    model: str = HEADLINE_MODEL,
    attempt_usage: TokenUsage | None = None,
    pricing: str = "dated",
) -> dict[int, int]:
    """output tokens per attempt -> affordable N. Zero is a real answer."""
    base = attempt_usage if attempt_usage is not None else PROJECTED_RUN_USAGE
    return {
        int(count): allocate(
            total_available_usd=total_available_usd,
            model=model,
            attempt_usage=with_output_tokens(base, count),
            pricing=pricing,
        ).affordable_n
        for count in output_token_counts
    }


#: Smallest N at which best-of-N is not the single-call arm (budget.Budget.degenerate).
NON_DEGENERATE_N = 2

#: The four settings priced by :func:`t12_pivot_funding_report`, as `(fund_t12_pivot, batch)`.
T12_PIVOT_SETTINGS: tuple[tuple[str, bool, bool], ...] = (
    ("excluded-unbatched", False, False),
    ("included-unbatched", True, False),
    ("excluded-batched", False, True),
    ("included-batched", True, True),
)


def t12_pivot_funding_report(
    *,
    total_available_usd: float = TOTAL_AVAILABLE_USD,
    calibration_settings: int = T7_CALIBRATION_SETTINGS,
    gate_arms: int = T9_GATE_ARMS,
    sealed_accesses: int = SEALED_ACCESSES,
    model: str = HEADLINE_MODEL,
    attempt_usage: TokenUsage | None = None,
    pricing: str = "dated",
) -> dict[str, Any]:
    """Price all four (FUND_T12_PIVOT_ARM, BATCH_SEALED_CAMPAIGN) settings, on the residual basis."""
    settings: dict[str, dict[str, Any]] = {}
    for label, fund, batch in T12_PIVOT_SETTINGS:
        plan = campaign_lines(
            calibration_settings=calibration_settings,
            gate_arms=gate_arms,
            sealed_accesses=sealed_accesses,
            batch_sealed=batch,
            fund_t12_pivot=fund,
        )
        allocation = allocate(
            total_available_usd=total_available_usd,
            model=model,
            attempt_usage=attempt_usage,
            lines=plan,
            pricing=pricing,
        )
        needed = allocation.usd_per_n * NON_DEGENERATE_N
        settings[label] = {
            "funds_the_pivot_arm": fund,
            "batch_sealed": batch,
            "funded_arms": sorted(funded_arms(plan)),
            "lines": [line.name for line in plan],
            "attempt_units": allocation.attempt_units,
            "usd_per_n": allocation.usd_per_n,
            "affordable_n": allocation.affordable_n,
            "committed_usd": allocation.committed_usd,
            "headroom_usd": allocation.residual_headroom_usd,
            # Positive: this many residual dollars short of a non-degenerate N. Negative: spare.
            "usd_short_of_non_degenerate_n": needed - allocation.residual_usd,
            "reaches_non_degenerate_n": allocation.affordable_n >= NON_DEGENERATE_N,
        }
    # "Shipped" must read both flags, or the pair below is differenced across the wrong rows.
    _batch_label = "batched" if BATCH_SEALED_CAMPAIGN else "unbatched"
    shipped = settings[f"excluded-{_batch_label}"]
    with_arm = settings[f"included-{_batch_label}"]
    return {
        "flag": "modembench.arms.ledger.FUND_T12_PIVOT_ARM",
        "default": FUND_T12_PIVOT_ARM,
        "default_setting": (
            f"{'included' if FUND_T12_PIVOT_ARM else 'excluded'}-{_batch_label}"
        ),
        "decision_owner": "project owner",
        "line": LedgerLine(
            name="t12-pivot-arm",
            ticket="T12",
            signals=DEV_SIGNALS,
            replicates=GATE_REPLICATES,
            arms=T12_PIVOT_ARMS,
            priced_on=T12_PIVOT_DATE,
        ).as_dict(),
        "usd_basis": (
            "residual: campaign dollars after the cross-model slice, which is the "
            "basis affordable_n is floored on and the affordability frontier tabulates"
        ),
        "non_degenerate_n": NON_DEGENERATE_N,
        "usd_per_n_for_the_pivot_arm": with_arm["usd_per_n"] - shipped["usd_per_n"],
        "costs_n_at_the_shipped_settings": (
            shipped["affordable_n"] - with_arm["affordable_n"]
        ),
        "settings": settings,
        "why_it_matters": (
            "docs/pre-registration.md §12 reaches F3 — the only thesis-refuting conclusion — "
            "only at t12=failed, and an unfunded arm cannot fail. With this flag False the "
            "campaign cannot present the evidence its own routing requires to conclude the "
            "thesis was wrong, whatever the data say."
        ),
        "recommendation": (
            "funded. This flag was set True together with BATCH_SEALED_CAMPAIGN, which is the "
            "one combination that funds the arm at a non-degenerate N: the arm costs no N at "
            "all at the shipped settings, only headroom, and the batched campaign reaches "
            "N = 2 with room to spare. It cost no additional dollars — the alternatives were "
            "reopening the budget band or discarding a sealed access's replication."
        ),
    }


def sealed_campaign_agreement(n_attempts: int | None = None) -> dict[str, Any]:
    """Check that this module and accounting agree on the sealed-access shape (GATE_REPLICATES)."""
    n = 0 if n_attempts is None else n_attempts
    ledger_units = SEALED_SIGNALS * GATE_REPLICATES * ARM_COUNT
    line_units = {
        line.name: line.attempt_units
        for line in campaign_lines()
        if line.name.startswith("sealed-access-")
    }
    return {
        "constant": "modembench.agent.accounting.SEALED_ATTEMPT_UNITS_PER_ACCESS",
        "retired_constant": "modembench.agent.accounting.SEALED_CAMPAIGN_RUNS",
        "retired_value": 360,
        "why_it_was_wrong": (
            "it priced one sealed access as 360 one-shot-shaped runs. The one-shot arm alone "
            "is 60 x 3 x N runs, so the constant understated that arm by a factor of N, and "
            "the iterative arm is a different run shape entirely rather than a second copy "
            "of the one-shot shape — it is matched to the same budget B, not to the same "
            "token shape."
        ),
        "formula": "60 signals x 3 replicates x 2 arms attempt-units, each costing N attempts",
        "accounting_attempt_units_per_access": SEALED_ATTEMPT_UNITS_PER_ACCESS,
        "ledger_attempt_units_per_access": ledger_units,
        "ledger_sealed_line_attempt_units": line_units,
        "gate_replicates": GATE_REPLICATES,
        "accounting_sealed_replicates": SEALED_REPLICATES,
        "agree": (
            ledger_units == SEALED_ATTEMPT_UNITS_PER_ACCESS
            and GATE_REPLICATES == SEALED_REPLICATES
            and set(line_units.values()) <= {SEALED_ATTEMPT_UNITS_PER_ACCESS}
        ),
        "sealed_accesses": SEALED_ACCESSES,
        "n_attempts": n,
        # Through accounting's own helper, so this exercises the correction.
        "one_shot_shaped_attempts_at_n": sealed_campaign_attempts(n, accesses=SEALED_ACCESSES),
    }


def sealed_campaign_runs_defect() -> dict[str, Any]:
    """Retained alias for sealed_campaign_agreement; arms.__init__ still re-exports the name."""
    return sealed_campaign_agreement()
