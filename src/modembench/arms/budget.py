"""The exogenous budget B, the frozen N, the cap-policy table, and the frozen config."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping

from ..agent.accounting import (
    ACCOUNTING_POLICY_VERSION,
    PRICE_TABLE_DATE,
    PROJECTED_RUN_USAGE,
    PROJECTED_TOOL_TURNS,
    TokenUsage,
    cost_usd,
    uncached_input_tokens_for_tool_turns,
)
from ..agent.feedback import FEEDBACK_POLICY_VERSION
from ..agent.harness import (
    HARNESS_POLICY_VERSION,
    ITERATIVE_RECEIVER_CONTRACT,
    ITERATIVE_ROUND_NOTICE,
    ITERATIVE_SYSTEM_PROMPT,
    ITERATIVE_TOOL_BUDGET_NOTICE,
    OUTCOME_AGENT_FAILURE,
    OUTCOME_RUN_INVALID,
    RECEIVER_CONTRACT,
    ROUND_CAP_REACHED,
    SYSTEM_PROMPT,
    TOOL_BUDGET_NOTICE,
    AgentConfig,
    cached_prefix_sizing,
    classify_outcome,
)
from ..agent.provider import (
    FROZEN_EFFORT,
    FROZEN_THINKING,
    HEADLINE_MODEL,
    PROVIDER_POLICY_VERSION,
    canonical_json,
)
from ..agent.tools import TOOLS_POLICY_VERSION, tools_sha256
from ..sandbox.ast_gate import ALLOW_STDLIB_MATH, AST_POLICY_VERSION
from ..sandbox.profile import POLICY_VERSION as SANDBOX_POLICY_VERSION
from .ledger import (
    LEDGER_POLICY_VERSION,
    LIST_RATE_REFERENCE_DATE,
    TOTAL_AVAILABLE_USD,
    LedgerAllocation,
    allocate,
    raw_tokens,
)
from hashlib import sha256

BUDGET_POLICY_VERSION = "modembench-budget-v1"


class BudgetError(RuntimeError):
    """A budget that cannot be honoured: ``N`` out of range, or a cap that binds wrongly."""


# --- 1. B and N -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Budget:
    """The exogenous per-signal budget and the ``N`` it buys, with its provenance attached."""

    allocation: LedgerAllocation
    budget_per_signal_usd: float
    n_attempts: int
    n_ceiling: int
    mean_attempt_usd: float
    mean_attempt_raw_tokens: int
    mean_attempt_output_tokens: int

    @property
    def degenerate(self) -> bool:
        """N == 1: best-of-N collapses onto the naive single-call arm. Not an error, but loud."""
        return self.n_attempts <= 1

    @property
    def degenerate_note(self) -> str | None:
        if not self.degenerate:
            return None
        return (
            "N == 1: best-of-N is the naive single-call arm, the best-of-n curve has one "
            "point, and no compute confound is removed. Raise TOTAL_AVAILABLE_USD, batch the "
            "sealed campaign, or cut a ledger line before freezing this N."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_policy_version": BUDGET_POLICY_VERSION,
            "budget_per_signal_usd": self.budget_per_signal_usd,
            "budget_currency": (
                "total billed cost under the five-rate table; raw tokens and output tokens "
                "are reported alongside and never matched on"
            ),
            "n_attempts": self.n_attempts,
            "n_ceiling": self.n_ceiling,
            "n_ceiling_source": "the ledger's affordable N, not an arbitrary constant",
            "mean_attempt_usd": self.mean_attempt_usd,
            "mean_attempt_raw_tokens": self.mean_attempt_raw_tokens,
            "mean_attempt_output_tokens": self.mean_attempt_output_tokens,
            "degenerate": self.degenerate,
            "degenerate_note": self.degenerate_note,
            "budget_direction": (
                "exogenous: B is set from the ledger and BOTH arms are matched to it. "
                "Neither arm's budget is derived from the other, which is what breaks the "
                "budget-derivation cycle and honours the plan's matched-budget "
                "stated direction."
            ),
            "allocation": self.allocation.as_dict(),
        }


def derive_budget(
    *,
    total_available_usd: float = TOTAL_AVAILABLE_USD,
    model: str = HEADLINE_MODEL,
    attempt_usage: TokenUsage | None = None,
    pricing: str = "dated",
) -> Budget:
    """Derive B and N from the ledger residual. N < 1 is a hard error, never a clamp."""
    usage = attempt_usage if attempt_usage is not None else PROJECTED_RUN_USAGE
    allocation = allocate(
        total_available_usd=total_available_usd,
        model=model,
        attempt_usage=usage,
        pricing=pricing,
    )
    if allocation.affordable_n < 1:
        raise BudgetError(
            "the ledger affords N = 0 attempts per signal per arm: "
            f"${allocation.residual_usd:.2f} of residual against "
            f"${allocation.usd_per_n:.2f} per unit of N. Raise the budget, cut a ledger "
            "line, batch the sealed campaign, or re-measure the run shape — but do not clamp."
        )
    return Budget(
        allocation=allocation,
        budget_per_signal_usd=allocation.budget_per_signal_usd,
        n_attempts=allocation.affordable_n,
        n_ceiling=allocation.affordable_n,
        mean_attempt_usd=allocation.reference_unit_usd,
        mean_attempt_raw_tokens=raw_tokens(usage),
        mean_attempt_output_tokens=usage.output_tokens,
    )


BUDGET = derive_budget()
BUDGET_PER_SIGNAL_USD = BUDGET.budget_per_signal_usd
N_ATTEMPTS = BUDGET.n_attempts
N_CEILING = BUDGET.n_ceiling


def validate_campaign_n(n: int, *, ceiling: int = N_CEILING) -> int:
    """Hard-error on a campaign N outside [1, ceiling]. Never clamps."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise BudgetError("campaign N must be an integer")
    if n < 1:
        raise BudgetError(f"campaign N must be at least 1; got {n}")
    if n > ceiling:
        raise BudgetError(
            f"campaign N of {n} exceeds the ledger-derived ceiling of {ceiling}. The ceiling "
            "is what the budget affords, so raising it means raising the budget."
        )
    return n


# --- 3. the cap asymmetry --------------------------------------------------------------------
@dataclass(frozen=True)
class CapPolicy:
    """One cap, and the pre-registered reason it is scored the way it is."""

    name: str
    currency: str
    countable_by_model: bool
    disclosed_to_model: bool
    harness_reason: str
    outcome_kind: str
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "currency": self.currency,
            "countable_by_model": self.countable_by_model,
            "disclosed_to_model": self.disclosed_to_model,
            "harness_reason": self.harness_reason,
            "outcome_kind": self.outcome_kind,
            "rationale": self.rationale,
        }


def cap_policies() -> tuple[CapPolicy, ...]:
    """The pre-registered table. Every row is asserted against the shipped harness."""
    return (
        CapPolicy(
            name="max_tool_calls",
            currency="tool_calls",
            countable_by_model=True,
            disclosed_to_model=True,
            harness_reason="tool_budget_exhausted",
            outcome_kind=OUTCOME_AGENT_FAILURE,
            rationale=(
                "a model can count its own tool calls, and TOOL_BUDGET_NOTICE states the "
                "exact number in the cached system block of every request. A model told its "
                "budget that spends all of it without submitting a receiver has failed the "
                "task, not been cut off by the instrument."
            ),
        ),
        CapPolicy(
            name="max_rounds",
            currency="rounds",
            countable_by_model=True,
            disclosed_to_model=True,
            harness_reason=ROUND_CAP_REACHED,
            outcome_kind=OUTCOME_AGENT_FAILURE,
            rationale=(
                "the iterative arm's terminal condition, and it is scored on the same ground "
                "max_tool_calls is: ITERATIVE_ROUND_NOTICE states the number of rounds in the "
                "cached system block of every request, and a round is the most countable unit "
                "in the run. The alternative the ticket text reaches for — 'until success or "
                "budget exhaustion', with the budget in dollars — makes cost_cap_exceeded the "
                "arm's ordinary exit, and that is pre-registered as run_invalid, so every run "
                "terminating by exhausting its budget would be deleted from its own "
                "denominator and re-run. 'Until success or budget exhaustion' would mean "
                "'until success or deletion'."
            ),
        ),
        CapPolicy(
            name="max_run_usd",
            currency="usd",
            countable_by_model=False,
            disclosed_to_model=False,
            harness_reason="cost_cap_exceeded",
            outcome_kind=OUTCOME_RUN_INVALID,
            rationale=(
                "a model cannot observe its token spend, its cache hit rate or the price "
                "table, so a dollar cap is invisible to it. Scoring a dollar cutoff charges "
                "the model for the instrument. Already ships as run_invalid; this row "
                "records why, it does not change it."
            ),
        ),
        CapPolicy(
            name="max_output_tokens",
            currency="tokens",
            countable_by_model=False,
            disclosed_to_model=False,
            harness_reason="max_tokens_truncation",
            outcome_kind=OUTCOME_RUN_INVALID,
            rationale=(
                "token counts are not observable to the model mid-generation and depend on a "
                "tokenizer it is not given. A truncated turn is the transport failing to "
                "carry an answer, not an answer that was wrong."
            ),
        ),
    )


def verify_cap_policy_against_harness() -> None:
    """Raise if any pre-registered row disagrees with the shipped classify_outcome."""
    for policy in cap_policies():
        outcome = classify_outcome(
            sandbox_status=None,
            packet_success=False,
            harness_reason=policy.harness_reason,
        )
        if outcome.kind != policy.outcome_kind:
            raise BudgetError(
                f"cap {policy.name!r} is pre-registered as {policy.outcome_kind!r} but the "
                f"shipped harness classifies {policy.harness_reason!r} as {outcome.kind!r}"
            )
        if policy.countable_by_model != policy.disclosed_to_model:
            raise BudgetError(
                f"cap {policy.name!r} breaks the asymmetry: a cap is disclosed exactly when "
                "the model can count it"
            )


def projected_task_tokens(
    usage: TokenUsage | None = None, tool_turns: int = PROJECTED_TOOL_TURNS
) -> int:
    """Invert uncached_input_tokens_for_tool_turns (affine in task_tokens) to recover task tokens.

    Inverted rather than hard-coded so it tracks accounting.PROJECTED_RUN_USAGE.
    """
    shape = usage if usage is not None else PROJECTED_RUN_USAGE
    at_zero = uncached_input_tokens_for_tool_turns(tool_turns, task_tokens=0)
    per_task = uncached_input_tokens_for_tool_turns(tool_turns, task_tokens=1) - at_zero
    if per_task <= 0:
        raise BudgetError("the projected input model is not increasing in task tokens")
    return max(0, (shape.input_tokens - at_zero) // per_task)


# Backstop sits at 2x the countable worst case so it can never fire before a disclosed cap.
BACKSTOP_SAFETY_FACTOR = 2.0


def worst_case_countable_run(
    *,
    config: AgentConfig | None = None,
    model: str | None = None,
    on: date | None = None,
    usage: TokenUsage | None = None,
) -> dict[str, Any]:
    """The most expensive run this configuration's countable caps still permit.

    Every cap comes from config, not module defaults: a backstop pinned beside a config must
    describe that config's caps, and the rendered prefix depends on the tool cap too.
    """
    settings = config or AgentConfig()
    shape = usage if usage is not None else PROJECTED_RUN_USAGE
    priced_model = model or settings.model
    max_tool_calls = int(settings.max_tool_calls)
    max_output_tokens = int(settings.max_output_tokens)
    requests = max_tool_calls + 1
    input_tokens = uncached_input_tokens_for_tool_turns(
        max_tool_calls, task_tokens=projected_task_tokens(shape)
    )
    prefix_tokens = int(cached_prefix_sizing(settings)["estimated_tokens_nominal"])
    worst = TokenUsage(
        input_tokens=input_tokens,
        # The prefix is written once and re-read on every later request.
        cache_creation_5m_tokens=prefix_tokens,
        cache_read_tokens=prefix_tokens * max_tool_calls,
        output_tokens=max_output_tokens * requests,
    )
    priced_on = on or LIST_RATE_REFERENCE_DATE
    return {
        "config_model": settings.model,
        "requests": requests,
        "max_tool_calls": max_tool_calls,
        "max_output_tokens_per_turn": max_output_tokens,
        "cached_prefix_tokens": prefix_tokens,
        "caps_source": "the AgentConfig this backstop is pinned beside, not module defaults",
        "usage": worst.as_dict(),
        "raw_tokens": raw_tokens(worst),
        "priced_on": priced_on.isoformat(),
        "usd": cost_usd(worst, model=priced_model, on=priced_on),
    }


def undisclosed_cost_cap_usd(
    *,
    config: AgentConfig | None = None,
    model: str | None = None,
    on: date | None = None,
    usage: TokenUsage | None = None,
    safety_factor: float = BACKSTOP_SAFETY_FACTOR,
) -> float:
    """The value for AgentConfig.max_run_usd: a backstop derived from this config's countable caps."""
    if safety_factor <= 1.0:
        raise BudgetError("the backstop must sit strictly above the countable worst case")
    worst = worst_case_countable_run(config=config, model=model, on=on, usage=usage)
    return float(worst["usd"]) * float(safety_factor)


# --- the arm-invariant / arm-specific split -------------------------------------------------
# Keys that describe the arm rather than the experiment. An arm-specific key missing here
# lands in the invariant digest and the paired analysis refuses the arms (safe direction).
ARM_SPECIFIC_KEYS: tuple[str, ...] = (
    "arm",
    "max_tool_calls",
    "max_executions",
    "max_rounds",
    "max_run_usd",
    "system_prompt_sha256",
    "receiver_contract_sha256",
)


def _config_dict(config: AgentConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return dict(config.as_dict())


def split_config(config: AgentConfig | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (arm_invariant, arm_specific) halves of a configuration."""
    payload = _config_dict(config)
    specific = {key: payload[key] for key in ARM_SPECIFIC_KEYS if key in payload}
    invariant = {key: value for key, value in payload.items() if key not in specific}
    return invariant, specific


def arm_invariant_digest(config: AgentConfig | Mapping[str, Any]) -> str:
    """The digest the paired analysis pairs on: model, effort, thinking, tools, policies, prices."""
    return sha256(canonical_json(split_config(config)[0])).hexdigest()


def arm_specific_digest(config: AgentConfig | Mapping[str, Any]) -> str:
    """The digest the paired analysis reports side by side: arm name, caps, rendered prompt."""
    return sha256(canonical_json(split_config(config)[1])).hexdigest()


# Exact substrings of the one-shot system block; all three are false for the iterative arm.
ONE_SHOT_PROMPT_ASSERTIONS: tuple[tuple[str, str], ...] = (
    ("SYSTEM_PROMPT", "There is no feedback of any kind on your answer."),
    ("TOOL_BUDGET_NOTICE", "One receiver, one execution, no feedback."),
    ("RECEIVER_CONTRACT", "You get exactly one execution."),
)

_PROMPT_SOURCES: dict[str, str] = {
    "SYSTEM_PROMPT": SYSTEM_PROMPT,
    "TOOL_BUDGET_NOTICE": TOOL_BUDGET_NOTICE,
    "RECEIVER_CONTRACT": RECEIVER_CONTRACT,
}


def one_shot_prompt_assertions() -> dict[str, Any]:
    """The pre-registered textual delta between the two arms' system blocks."""
    missing = [
        name
        for name, sentence in ONE_SHOT_PROMPT_ASSERTIONS
        if sentence not in _PROMPT_SOURCES[name]
    ]
    if missing:
        raise BudgetError(
            "the pre-registered one-shot prompt assertions are no longer present in "
            f"{missing}: the textual delta the paired analysis relies on has drifted"
        )
    return {
        "assertions": [
            {"source": name, "text": sentence} for name, sentence in ONE_SHOT_PROMPT_ASSERTIONS
        ],
        "one_shot_system_prompt_sha256": sha256(
            AgentConfig().system_text().encode("utf-8")
        ).hexdigest(),
        "requirement": (
            "the iterative arm must remove or rewrite every sentence above; all three "
            "are false for it. The arms are therefore paired on the arm-invariant digest and "
            "their arm-specific digests are reported side by side, never averaged."
        ),
    }


# Required substrings of the iterative system block. {max_rounds} stays a literal format
# field: the round count is rendered per configuration.
ITERATIVE_PROMPT_ASSERTIONS: tuple[tuple[str, str], ...] = (
    (
        "ITERATIVE_SYSTEM_PROMPT",
        "After each attempt you are shown a small fixed summary of how it scored",
    ),
    ("ITERATIVE_ROUND_NOTICE", "You get {max_rounds} rounds, not one."),
    (
        "ITERATIVE_ROUND_NOTICE",
        "Reaching the round limit without recovering the",
    ),
    ("ITERATIVE_ROUND_NOTICE", "exactly four fields and nothing else"),
    ("ITERATIVE_ROUND_NOTICE", "QUANTIZED to the nearest multiple of 1/64"),
    ("ITERATIVE_TOOL_BUDGET_NOTICE", "available in round 1 only"),
    ("ITERATIVE_RECEIVER_CONTRACT", "Each round you submit is executed once."),
)

_ITERATIVE_PROMPT_SOURCES: dict[str, str] = {
    "ITERATIVE_SYSTEM_PROMPT": ITERATIVE_SYSTEM_PROMPT,
    "ITERATIVE_TOOL_BUDGET_NOTICE": ITERATIVE_TOOL_BUDGET_NOTICE,
    "ITERATIVE_ROUND_NOTICE": ITERATIVE_ROUND_NOTICE,
    "ITERATIVE_RECEIVER_CONTRACT": ITERATIVE_RECEIVER_CONTRACT,
}


def iterative_prompt_assertions(max_rounds: int = 2) -> dict[str, Any]:
    """The iterative half of the textual delta. Also checks no one-shot sentence survives."""
    config = AgentConfig(max_rounds=max_rounds, arm="iterative", max_executions=max_rounds)
    rendered = config.system_text()
    missing = [
        (name, text)
        for name, text in ITERATIVE_PROMPT_ASSERTIONS
        if text not in _ITERATIVE_PROMPT_SOURCES[name]
    ]
    if missing:
        raise BudgetError(
            "the pre-registered iterative prompt assertions are no longer present in "
            f"{[name for name, _ in missing]}: the arm's stated condition has drifted from "
            "the arm it is measuring"
        )
    survived = [
        name for name, sentence in ONE_SHOT_PROMPT_ASSERTIONS if sentence in rendered
    ]
    if survived:
        raise BudgetError(
            f"the iterative system block still asserts the one-shot condition {survived}: "
            "every one of those sentences is FALSE for an arm that runs "
            f"{max_rounds} rounds with feedback between them"
        )
    return {
        "assertions": [
            {"source": name, "text": text} for name, text in ITERATIVE_PROMPT_ASSERTIONS
        ],
        "one_shot_assertions_absent": [name for name, _ in ONE_SHOT_PROMPT_ASSERTIONS],
        "max_rounds": int(max_rounds),
        "iterative_system_prompt_sha256": sha256(rendered.encode("utf-8")).hexdigest(),
        "arm_invariant_digest": arm_invariant_digest(config),
        "arm_specific_digest": arm_specific_digest(config),
        "pairing_requirement": (
            "the two arms MUST report the same arm_invariant_digest and different "
            "arm_specific_digests. The analysis pairs on the first and reports the second side by "
            "side; it never averages them."
        ),
    }


# --- 2. the frozen configuration ------------------------------------------------------------
SELECTOR_PREREGISTRATION: dict[str, Any] = {
    "selector": "crc_pass",
    "selector_is_agent_visible": True,
    "selected_on_ground_truth": False,
    "scored_on": "packet_success of the selected attempt",
    "rationale": (
        "packet_success requires claimed_length == true_length AND aligned_ber == 0 AND "
        "crc_pass, and crc_pass verifies a 32-bit CRC, so the two differ only on a ~2^-32 "
        "event. crc_pass is already forwarded to the agent as Tier-2 feedback and needs no "
        "ground truth, so selecting on it keeps the baseline deployable. Selecting on truth "
        "would make the arm pass@N — an oracle nobody can deploy — and a gate FAIL against "
        "an oracle cannot distinguish 'feedback does not help' from 'max over N draws is a "
        "strong estimator'."
    ),
    "pass_at_n_reported_as": "an explicitly labelled UPPER BOUND, never the gate verdict",
    "tie_break": "lowest attempt index, among attempts passing the selector",
    "tie_break_rationale": (
        "no BER, at any resolution, is admissible as a selection input. Every aligned_ber in "
        "this project — including the 1/64-quantized public.feedback.aligned_ber that "
        "feedback.forward_feedback hands the agent — is computed against the HIDDEN payload, "
        "and quantizing truth coarsens it without stopping it being truth. Agent-visible as "
        "benchmark feedback is not the same as available to a deployed receiver: in the field "
        "there is no payload to compare against, so no BER exists to rank on. An earlier "
        "revision ranked ties on the quantized aligned_ber and argued that doing so 'keeps the "
        "selector deployable'; that argument was wrong and is reverted. "
        "Attempt.selection_key is (not crc_pass, index) and contains no BER. crc_pass is "
        "admissible for the opposite reason: evaluator._crc_passes reads only the bits the "
        "receiver itself produced, so a fielded receiver can compute it unaided."
    ),
    "secondary_metric_selected_by": (
        "nothing: BER is REPORTED, never SELECTED ON. arms.bestofn.report_ber publishes two "
        "named estimators computed directly from the retained attempts — mean_ber_all_attempts "
        "(the mean over signals of the mean BER of that signal's N valid attempts, which "
        "involves no selection and so is not a mixture whose weight is the primary metric) and "
        "oracle_min_ber_at_n (the mean over signals of the lowest BER among the N attempts, "
        "labelled TRUTH-AIDED and reported as an unachievable bound, exactly as "
        "pass_at_n_upper_bound is for the primary metric). The BER of the SELECTED attempt is "
        "deliberately not among them."
    ),
}


def frozen_budget_config(
    budget: Budget | None = None, *, config: AgentConfig | None = None
) -> dict[str, Any]:
    """The object the freeze pins. Verifies the cap table against the harness before returning."""
    resolved = budget or BUDGET
    settings = config or AgentConfig()
    verify_cap_policy_against_harness()
    return {
        "budget_policy_version": BUDGET_POLICY_VERSION,
        "ledger_policy_version": LEDGER_POLICY_VERSION,
        "accounting_policy_version": ACCOUNTING_POLICY_VERSION,
        "harness_policy_version": HARNESS_POLICY_VERSION,
        "provider_policy_version": PROVIDER_POLICY_VERSION,
        "feedback_policy_version": FEEDBACK_POLICY_VERSION,
        "tools_policy_version": TOOLS_POLICY_VERSION,
        "ast_policy_version": AST_POLICY_VERSION,
        "sandbox_policy_version": SANDBOX_POLICY_VERSION,
        "allow_stdlib_math": ALLOW_STDLIB_MATH,
        "price_table_date": PRICE_TABLE_DATE.isoformat(),
        "model": settings.model,
        "effort": FROZEN_EFFORT,
        "thinking": FROZEN_THINKING,
        "tools_sha256": tools_sha256(),
        "budget": resolved.as_dict(),
        "selector": SELECTOR_PREREGISTRATION,
        "caps": [policy.as_dict() for policy in cap_policies()],
        # Both derived from `settings`, the config they are pinned beside.
        "undisclosed_cost_backstop_usd": undisclosed_cost_cap_usd(config=settings),
        "worst_case_countable_run": worst_case_countable_run(config=settings),
        "arm_invariant_sha256": arm_invariant_digest(settings),
        "arm_specific_sha256": arm_specific_digest(settings),
        "prompt_delta": one_shot_prompt_assertions(),
    }


def frozen_budget_hash(
    budget: Budget | None = None, *, config: AgentConfig | None = None
) -> str:
    """Canonical hash of frozen_budget_config."""
    return sha256(canonical_json(frozen_budget_config(budget, config=config))).hexdigest()


def budget_summary(budget: Budget | None = None) -> dict[str, Any]:
    """Everything docs/budget-ledger.md quotes, in one object, for a doc-drift test."""
    resolved = budget or BUDGET
    allocation = resolved.allocation
    return {
        "total_available_usd": allocation.total_available_usd,
        "residual_usd": allocation.residual_usd,
        "cross_model_reserve_usd": allocation.cross_model_reserve_usd,
        "attempt_units": allocation.attempt_units,
        "usd_per_n": allocation.usd_per_n,
        "n_attempts": resolved.n_attempts,
        "n_ceiling": resolved.n_ceiling,
        "budget_per_signal_usd": resolved.budget_per_signal_usd,
        "mean_attempt_usd": resolved.mean_attempt_usd,
        "total_committed_usd": allocation.total_committed_usd,
        "residual_headroom_usd": allocation.residual_headroom_usd,
        "degenerate": resolved.degenerate,
    }


def n_from_budget(budget_usd: float, mean_attempt_usd: float) -> int:
    """N = floor(B / mean cost per one-shot attempt). Returns N_ATTEMPTS for the shipped budget."""
    if mean_attempt_usd <= 0:
        raise BudgetError("mean attempt cost must be positive")
    return int(math.floor(float(budget_usd) / float(mean_attempt_usd)))
