"""One-shot agent harness: tool loop, single sandbox execution, outcome classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..records import (
    ALWAYS,
    SEALED,
    TRACE_POLICY_VERSION,
    Constant,
    Drop,
    Null,
    RecordPolicy,
    Redact,
    Substitute,
    append_record,
    refuse_sealed_identity,
    sealing_of_capture,
    trace_head_sha256,
    write_json_once,
    write_record,
)
from ..sandbox.ast_gate import AST_POLICY_VERSION
from ..sandbox.profile import POLICY_VERSION as SANDBOX_POLICY_VERSION
from ..sandbox.profile import SANDBOX_EXEC
from ..sandbox.runner import RUNS_ROOT, run_receiver
from ..sealed import (
    META_ARTIFACT,
    capture_reference,
    sealed_root_containing,
    sealed_run_artifact_dir,
    sealed_run_record_path,
    write_sealed_run_record,
)
from .accounting import (
    ACCOUNTING_POLICY_VERSION,
    PRICE_TABLE_DATE,
    TokenUsage,
    cache_minimum_tokens,
    cost_breakdown,
    cost_usd,
    rates_for,
)
from .feedback import FEEDBACK_POLICY_VERSION, forward_feedback, FeedbackWallError, feedback_config
from .provider import (
    FROZEN_EFFORT,
    FROZEN_THINKING,
    HEADLINE_MODEL,
    MAX_OUTPUT_TOKENS,
    PROVIDER_POLICY_VERSION,
    SERVER_SIDE_FALLBACKS_ENABLED,
    Provider,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailable,
    canonical_json,
    require_scoring_platform,
)
from .tools import (
    MAX_TOOL_CALLS_PER_RUN,
    MAX_TOOL_RESULT_BYTES,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    TOOLS_POLICY_VERSION,
    CaptureSignal,
    ToolResult,
    call_tool,
    load_capture,
    tools_config,
    tools_sha256,
)

HARNESS_POLICY_VERSION = "modembench-harness-v1"
AGENT_RUNS_SUBDIR = "agent"
_ARTIFACT = "agent run artifact"


# --- outcome taxonomy (pre-registered) ----------------------------------------------------
OUTCOME_SUCCESS = "scored_success"
OUTCOME_AGENT_FAILURE = "scored_agent_failure"
OUTCOME_RUN_INVALID = "run_invalid"

# Sandbox statuses attributable to the receiver; scored as failed packets.
AGENT_FAILURE_STATUSES = frozenset(
    {
        "ast_rejected",
        "crashed",
        "timeout",
        "memory_exceeded",
        "cpu_exceeded",
        "fsize_exceeded",
        "bad_output",
    }
)
# Sandbox statuses attributable to the machine. Excluded from the denominator and re-run.
INFRASTRUCTURE_STATUSES = frozenset({"sandbox_unavailable", "resource_monitor_unavailable"})

# The runner collapses several reasons into `crashed`; only these two are the receiver's
# fault. Everything else (shim bugs, signals, unmapped exits) classifies as run_invalid.
RECEIVER_CRASH_REASONS = frozenset({"missing_receive", "receiver_raised"})
# Non-sandbox invalidity: the model or the transport, never the receiver.
HARNESS_INVALID_REASONS = frozenset(
    {
        "model_identity_mismatch",
        "refusal",
        "max_tokens_truncation",
        "pause_turn_unhandled",
        "transport_error",
        "structured_output_parse_failure",
        "evaluator_invalid",
        "feedback_wall_violation",
        "cost_cap_exceeded",
        "sandbox_unavailable",
        "resource_monitor_unavailable",
    }
)
# Agent-side failures that never reach the sandbox. These are scored (not run_invalid)
# because the caps are disclosed to the model in TOOL_BUDGET_NOTICE.
PRE_SANDBOX_AGENT_FAILURES = frozenset({"tool_budget_exhausted", "no_receiver_submitted"})

# The iterative arm's terminal condition; scored because the round cap is disclosed.
ROUND_CAP_REACHED = "round_cap_reached"

# Harness-side reasons scored against the agent; every member is a disclosed cap.
SCORED_HARNESS_REASONS = PRE_SANDBOX_AGENT_FAILURES | {ROUND_CAP_REACHED}


class SandboxUnavailableAbort(RuntimeError):
    """Sweep-stopping: on a machine where this fires it fires for every run."""


class ExecutionCapExceeded(RuntimeError):
    """A second sandbox execution was attempted inside one run."""


class ExecutionBudget:
    """Caps sandbox executions per run. Spent before the execution, so a crash still counts."""

    def __init__(self, limit: int = 1) -> None:
        if limit < 1:
            raise ValueError("execution budget must allow at least one execution")
        self.limit = int(limit)
        self.spent = 0

    def spend(self) -> None:
        if self.spent >= self.limit:
            raise ExecutionCapExceeded(
                f"refusing a second sandbox execution in one run (cap {self.limit})"
            )
        self.spent += 1

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


def guarded_executor(
    executor: Callable[..., dict[str, Any]], budget: ExecutionBudget
) -> Callable[..., dict[str, Any]]:
    """Wrap an executor so every call spends the budget."""

    def execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        budget.spend()
        return executor(*args, **kwargs)

    return execute


@dataclass(frozen=True)
class Outcome:
    """The pre-registered verdict for one run."""

    kind: str
    status: str
    reason: str | None
    packet_success: bool

    @property
    def scored(self) -> bool:
        return self.kind in (OUTCOME_SUCCESS, OUTCOME_AGENT_FAILURE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "packet_success": self.packet_success,
            "scored": self.scored,
        }


def classify_outcome(
    *,
    sandbox_status: str | None,
    packet_success: bool,
    sandbox_error: str | None = None,
    provider_invalid_reason: str | None = None,
    harness_reason: str | None = None,
    evaluator_error: str | None = None,
) -> Outcome:
    """Map one run's raw signals onto the pre-registered taxonomy. Order is load-bearing:
    provider invalidity outranks everything, then infrastructure, then sandbox status."""
    if provider_invalid_reason is not None:
        return Outcome(OUTCOME_RUN_INVALID, "provider", provider_invalid_reason, False)
    if harness_reason in SCORED_HARNESS_REASONS:
        return Outcome(OUTCOME_AGENT_FAILURE, harness_reason, harness_reason, False)
    if harness_reason is not None:
        return Outcome(OUTCOME_RUN_INVALID, "harness", harness_reason, False)
    if sandbox_status in INFRASTRUCTURE_STATUSES:
        return Outcome(OUTCOME_RUN_INVALID, sandbox_status, sandbox_status, False)
    if evaluator_error == "evaluator_invalid":
        return Outcome(OUTCOME_RUN_INVALID, "evaluator_invalid", "evaluator_invalid", False)
    if sandbox_status == "crashed" and sandbox_error not in RECEIVER_CRASH_REASONS:
        # An unattributable crash must not be booked against the model.
        return Outcome(
            OUTCOME_RUN_INVALID, "crashed", sandbox_error or "crash_reason_unavailable", False
        )
    if sandbox_status in AGENT_FAILURE_STATUSES:
        return Outcome(OUTCOME_AGENT_FAILURE, sandbox_status, sandbox_status, False)
    if sandbox_status == "ok":
        if packet_success:
            return Outcome(OUTCOME_SUCCESS, "ok", None, True)
        return Outcome(OUTCOME_AGENT_FAILURE, "ok", "non_decoding_output", False)
    return Outcome(OUTCOME_RUN_INVALID, str(sandbox_status), "unclassified_status", False)


# Pre-registered round composition: invalid rounds are re-issued in place and do not consume
# a round of R; valid rounds compose and the last one is the run's outcome.
ROUND_COMPOSITION_RULE: tuple[dict[str, str], ...] = (
    {
        "round_outcome": OUTCOME_SUCCESS,
        "loop": "stop",
        "run_outcome": "scored_success",
    },
    {
        "round_outcome": f"{OUTCOME_AGENT_FAILURE} (k < R)",
        "loop": "forward Tier-2 feedback and continue — this is the arm",
        "run_outcome": "none yet",
    },
    {
        "round_outcome": f"{OUTCOME_AGENT_FAILURE} (k = R)",
        "loop": "stop",
        "run_outcome": "that failure, scored",
    },
    {
        "round_outcome": f"{OUTCOME_RUN_INVALID} (any k)",
        "loop": "re-issue the round in place; it does not consume a round of R",
        "run_outcome": "none, if the re-issue produces a valid round",
    },
    {
        "round_outcome": f"{OUTCOME_RUN_INVALID} past MAX_ROUND_RERUNS",
        "loop": "stop",
        "run_outcome": "run_invalid; the whole run is re-run and is never scored",
    },
)

# How many times one round may be re-issued before the run is abandoned.
MAX_ROUND_RERUNS = 3


class RoundRerunCapExceeded(RuntimeError):
    """A single round produced ``run_invalid`` more times than the cap allows."""


def compose_rounds(rounds: Sequence[Outcome], *, round_cap: int) -> Outcome:
    """Fold valid round outcomes into a run outcome per ROUND_COMPOSITION_RULE."""
    if round_cap < 1:
        raise ValueError(f"round cap must be at least 1; got {round_cap}")
    if not rounds:
        return Outcome(OUTCOME_RUN_INVALID, "harness", "no_valid_round", False)
    if len(rounds) > round_cap:
        raise ValueError(
            f"{len(rounds)} valid rounds against a cap of {round_cap}: the execution budget "
            "should have made this unreachable"
        )
    for index, outcome in enumerate(rounds):
        if outcome.kind == OUTCOME_RUN_INVALID:
            raise ValueError(
                f"round {index + 1} is {outcome.kind}: invalid rounds are re-issued, not "
                "composed — passing one here would score the instrument against the model"
            )
        if outcome.kind == OUTCOME_SUCCESS:
            if index != len(rounds) - 1:
                raise ValueError(
                    f"round {index + 1} succeeded but {len(rounds)} rounds ran: the loop must "
                    "stop on success, and a round after a success is a round the model was "
                    "charged for having already finished"
                )
            return outcome
    # All rounds failed; keep the last round's own reason so the status histogram sees it.
    return rounds[-1]


def status_histogram(outcomes: Iterable[Outcome | Mapping[str, Any]]) -> dict[str, Any]:
    """Status counts plus the packet-success numerator and denominator."""
    counts: dict[str, int] = {}
    kinds: dict[str, int] = {OUTCOME_SUCCESS: 0, OUTCOME_AGENT_FAILURE: 0, OUTCOME_RUN_INVALID: 0}
    successes = 0
    scored = 0
    for entry in outcomes:
        record = entry.as_dict() if isinstance(entry, Outcome) else dict(entry)
        status = str(record.get("status"))
        counts[status] = counts.get(status, 0) + 1
        kind = str(record.get("kind"))
        kinds[kind] = kinds.get(kind, 0) + 1
        if record.get("scored"):
            scored += 1
            if record.get("packet_success"):
                successes += 1
    return {
        "status_counts": dict(sorted(counts.items())),
        "kind_counts": dict(sorted(kinds.items())),
        "packet_success_numerator": successes,
        "attributable_denominator": scored,
        "packet_success_rate": (successes / scored) if scored else None,
        "denominator_definition": (
            "runs whose outcome is attributable to the agent; run_invalid outcomes are "
            "excluded and re-run"
        ),
    }


# --- pre-flight ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    platform: str
    sandbox_exec_present: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "platform": self.platform,
            "sandbox_exec_present": self.sandbox_exec_present,
            "error": self.error,
        }


def preflight() -> PreflightReport:
    """Report, without raising, whether this machine can score a receiver."""
    try:
        require_scoring_platform()
    except ProviderUnavailable as exc:
        return PreflightReport(
            ok=False,
            platform=sys.platform,
            sandbox_exec_present=SANDBOX_EXEC.is_file(),
            error=str(exc),
        )
    return PreflightReport(ok=True, platform=sys.platform, sandbox_exec_present=True)


# --- the agent contract -------------------------------------------------------------------
RECEIVER_SUBMISSION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "receiver_source": {
                "type": "string",
                "description": "The complete contents of receiver.py.",
            },
            "approach": {
                "type": "string",
                "description": "One or two sentences describing the receiver's design.",
            },
        },
        "required": ["receiver_source", "approach"],
        "additionalProperties": False,
    },
}

RECEIVER_CONTRACT = """\
Submit one Python module. It must define:

    def receive(iq, sample_rate):
        ...

`iq` is a 1-D complex NumPy array of the whole capture; `sample_rate` is the sample rate in
Hz. Return a 1-D `numpy.uint8` array of bits valued 0 or 1, at most 2**20 of them, with the
frame's sync word starting somewhere in the first 64 returned bits.

Execution environment:
  * `import numpy`, `import scipy.signal`, `import math` and `import cmath`. No other
    imports; no file or network access; no `open`, `eval`, `exec`, `getattr`, `globals`,
    `__import__`; no attribute whose name begins with an underscore; no frame or code
    introspection (`gi_frame`, `f_globals`, `co_consts`, ...); no NumPy IO (`load`, `save`,
    `fromfile`, `memmap`, `loadtxt`, ...). Defining dunder methods in a class is fine
    (`def __init__`), as is `if __name__ == "__main__"`; reading a dunder attribute off an
    object is not. Source is statically checked before it runs and a violation ends the run
    as a failure.
  * The module is imported and `receive` called once, inside a kernel sandbox with no
    network, no subprocesses and no writable filesystem outside its own scratch directory.
  * Wall-clock and CPU limits of a few seconds and a memory limit of a few hundred MiB. An
    exception, a timeout, or a badly-shaped return value ends the run as a failure.

You get exactly one execution. There is no scoring feedback and no second attempt: write the
receiver you believe is correct.
"""

SYSTEM_PROMPT = """\
You are a communications engineer. You are given one recorded complex-baseband capture of a
single packet from a digital radio, and a small set of numeric diagnostic tools that measure
the capture. You cannot see the samples directly; the tools are your only view of them.

Your task is to write a receiver that recovers the packet's bits.

Work like an instrument engineer: measure first, decide from the measurements, then write the
receiver. The tools are characterized instruments with known error, not oracles — cross-check
them against each other and against the published signal family before you commit.

Everything about this specific capture beyond what the tools report and the metadata below is
unknown to you and will not be revealed. There is no feedback of any kind on your answer.

What the instrument does and does not resolve, measured over the whole development split and
published in docs/tool-characterization.md rather than asserted here:

  * The ranked samples-per-symbol from `symbol_rate_candidates` is the strongest single
    measurement available to you. Its ranking is an estimate, not a disclosure: cross-check
    the top candidate against the autocorrelation's symbol-period estimate and the occupied
    bandwidth before you commit to it, and treat a disagreement between them as a finding.
  * The spectral centroid from `spectrum` is a direct carrier-frequency-offset estimate, good
    to roughly a hundred hertz. The occupied bandwidth is a coarse cross-check on the symbol
    rate ONLY: it carries no usable information about the pulse's excess-bandwidth factor, so
    do not try to infer the roll-off from it. Assume a root-raised-cosine matched filter and
    treat the roll-off as unknown within the published family.
  * The first null of the signal autocorrelation is a symbol-period estimate accurate to
    within a couple of samples, which is coarse next to the rate candidates. The nulls of the
    squared-envelope autocorrelation are a different statistic and are NOT at multiples of the
    symbol period; use them only as a shape cross-check.
  * Carrier phase, fractional timing offset and the exact frame position are not measured by
    any tool. Your receiver has to recover them itself from the samples.
"""

# Rendered from the run's configuration so a changed cap cannot ship a stale prompt.
TOOL_BUDGET_NOTICE = """\
Your budget for this run, stated up front so you can plan against it:

  * At most {max_tool_calls} tool calls in total, counted across every turn of the
    conversation. Parallel calls in one turn each count separately.
  * When the budget runs out the run ends immediately, with whatever you have submitted at
    that point. A run that ends without a submitted receiver is scored as a failure, so keep
    calls in reserve and submit before you are close to the limit.
  * Every tool result is capped at {max_tool_result_bytes} bytes. Nothing you can do will
    return the raw samples, so plan to measure, not to browse.
  * One receiver, one execution, no feedback. Write the receiver you believe is correct.
"""


# The iterative arm rewrites (not deletes) the three one-shot sentences pinned by
# arms.budget.ONE_SHOT_PROMPT_ASSERTIONS.
ITERATIVE_ROUND_NOTICE = """\
You get {max_rounds} rounds, not one.

  * Round 1: measure with the tools, then submit a receiver. It is executed immediately.
  * After each execution you are shown the result and asked to submit again, up to the round
    limit. Exactly {max_rounds} executions are available and the count is enforced.
  * The run ends the moment a receiver recovers the packet, or when round {max_rounds} has
    been executed — whichever comes first. Reaching the round limit without recovering the
    packet is scored as a failure, so treat every round as the one that has to work.

The result you are shown after each round is exactly four fields and nothing else:

  * `acquisition_success` — true if your bit stream contained the frame's sync word where the
    evaluator could find it. False means the fault is in burst detection, framing or sync
    search, and NOT in the demodulator downstream of them.
  * `crc_pass` — true if the recovered payload's CRC verified.
  * `aligned_ber` — the bit error rate after alignment, or null when acquisition failed. It is
    QUANTIZED to the nearest multiple of 1/64 before you see it. 0.0 means bit-exact; 0.015625
    is one grid step and may be any small nonzero rate. Do not read precision into it that is
    not there, and do not try to infer the payload length from it.
  * `error` — a short machine code naming the fault when there is one (for example
    `ast_rejected`, `output_wrong_shape`, `timeout`), or null.

There is no other feedback. Nothing tells you which symbols were wrong, where the frame
started, or how close you were.
"""

ITERATIVE_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "There is no feedback of any kind on your answer.",
    "After each attempt you are shown a small fixed summary of how it scored, and you may "
    "revise. What that summary contains is stated in full below; it is deliberately narrow.",
)

ITERATIVE_TOOL_BUDGET_NOTICE = TOOL_BUDGET_NOTICE.replace(
    "One receiver, one execution, no feedback. Write the receiver you believe is correct.",
    "The tools measure the recording, and the recording does not change between rounds. They "
    "are available in round 1 only; later rounds are repairs, with your round-1 measurements "
    "still in front of you.",
)

ITERATIVE_RECEIVER_CONTRACT = RECEIVER_CONTRACT.replace(
    "You get exactly one execution. There is no scoring feedback and no second attempt: "
    "write the\nreceiver you believe is correct.",
    "Each round you submit is executed once. What you are given back is the four-field summary "
    "described in the standing instructions, and nothing more.",
)


def iterative_system_text(
    max_rounds: int,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_RUN,
    max_tool_result_bytes: int = MAX_TOOL_RESULT_BYTES,
) -> str:
    """Rendered iterative system block, same order as the one-shot arm's."""
    if max_rounds < 2:
        raise ValueError(
            f"the iterative system block states a multi-round condition; max_rounds={max_rounds} "
            "would ship a prompt that is false about its own arm"
        )
    return (
        ITERATIVE_SYSTEM_PROMPT
        + "\n"
        + ITERATIVE_TOOL_BUDGET_NOTICE.format(
            max_tool_calls=int(max_tool_calls),
            max_tool_result_bytes=int(max_tool_result_bytes),
        )
        + "\n"
        + ITERATIVE_ROUND_NOTICE.format(max_rounds=int(max_rounds))
        + "\n"
        + ITERATIVE_RECEIVER_CONTRACT
    )


def _withheld_tool_result(name: str) -> "ToolResult":
    from .tools import _error_result

    return _error_result(name, f"unknown tool {name!r}")


def _without_tool_bullets(system: str, withheld: tuple[str, ...]) -> str:
    # A withheld tool must vanish from the prompt entirely, or the ablation measures
    # confusion rather than absence. Drops a `* ` bullet and its continuation lines.
    if not withheld:
        return system
    dropping = False
    kept: list[str] = []
    for line in system.splitlines():
        if line.lstrip().startswith("* "):
            dropping = any(name in line for name in withheld)
        if not dropping:
            kept.append(line)
    return "\n".join(kept)


def system_text(
    max_tool_calls: int = MAX_TOOL_CALLS_PER_RUN,
    max_tool_result_bytes: int = MAX_TOOL_RESULT_BYTES,
) -> str:
    """The full system block: standing instructions, disclosed budget, receiver contract.

    The contract sits in front of the cache breakpoint: it is identical on every run, and
    it lifts the cached prefix clear of the model's minimum cacheable size.
    """
    return (
        SYSTEM_PROMPT
        + "\n"
        + TOOL_BUDGET_NOTICE.format(
            max_tool_calls=int(max_tool_calls),
            max_tool_result_bytes=int(max_tool_result_bytes),
        )
        + "\n"
        + RECEIVER_CONTRACT
    )


# --- cached-prefix sizing ------------------------------------------------------------------
# No tokenizer in this repo, so prefix size is reported in bytes plus a token estimate over
# this bytes-per-token band; the caching claim must hold at the pessimistic end.
BYTES_PER_TOKEN_BAND = (3.5, 5.0)
# The ratio accounting.PROJECTED_RUN_USAGE is built on.
BYTES_PER_TOKEN_NOMINAL = 4.0


def cached_prefix_text(config: "AgentConfig | None" = None) -> str:
    """The exact text that sits in front of the cache breakpoint."""
    settings = config or AgentConfig()
    return (
        canonical_json(list(settings.offered_tool_schemas())).decode("utf-8")
        + settings.system_text()
    )


def cached_prefix_sizing(config: "AgentConfig | None" = None) -> dict[str, Any]:
    """Measured bytes and estimated token range of the cached prefix."""
    settings = config or AgentConfig()
    tools_bytes = len(canonical_json(list(TOOL_SCHEMAS)))
    system_bytes = len(settings.system_text().encode("utf-8"))
    total = tools_bytes + system_bytes
    high_ratio, low_ratio = BYTES_PER_TOKEN_BAND[0], BYTES_PER_TOKEN_BAND[1]
    minimum = cache_minimum_tokens(settings.model)
    low = int(total // low_ratio)
    return {
        "tool_definitions_bytes": tools_bytes,
        "system_block_bytes": system_bytes,
        "cached_prefix_bytes": total,
        "bytes_per_token_band": list(BYTES_PER_TOKEN_BAND),
        "estimated_tokens_low": low,
        "estimated_tokens_high": int(total // high_ratio),
        "estimated_tokens_nominal": int(total // BYTES_PER_TOKEN_NOMINAL),
        "model": settings.model,
        "minimum_cacheable_tokens": minimum,
        "clears_minimum_at_worst_case": low > minimum,
        "margin_over_minimum_at_worst_case": low - minimum,
    }


def _public_metadata(capture_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the public metadata only; never the manifest or private diagnostics."""
    raw = (Path(capture_dir) / META_ARTIFACT).read_text(encoding="utf-8")
    meta = json.loads(raw)
    if not isinstance(meta, dict):
        raise ValueError("public metadata is not a JSON object")
    return meta


def build_task_message(meta: Mapping[str, Any]) -> str:
    """The per-run user turn. Re-serialized field by field so a new metadata field cannot
    ride into the prompt unreviewed."""
    family = dict(meta.get("signal_family") or {})
    framing = dict(meta.get("framing") or {})
    output = dict(meta.get("receiver_output") or {})
    published = {
        "sample_rate_hz": meta.get("sample_rate_hz"),
        "signal_family": {
            "modulation": family.get("modulation"),
            "mapping": family.get("mapping"),
            "pulse_shape": family.get("pulse_shape"),
            "pulse_span_symbols": family.get("pulse_span_symbols"),
            "impairments": family.get("impairments"),
        },
        "framing": {
            "layout": framing.get("layout"),
            "sync_bits": framing.get("sync_bits"),
            "length_bits": framing.get("length_bits"),
            "length_encoding": framing.get("length_encoding"),
            "payload_length_bytes": framing.get("payload_length_bytes"),
            "bit_order": framing.get("bit_order"),
            "crc": framing.get("crc"),
        },
        "receiver_output": output,
    }
    return (
        "Published signal family and framing for this capture (the instance parameters are "
        "not published):\n\n"
        + json.dumps(published, indent=2, sort_keys=True)
        + "\n\nThe sync word's bit pattern is not published; its length is. Use the tools to "
        "measure the capture, then submit one receiver under the contract in your "
        "instructions."
    )


# --- frozen configuration -----------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    """Everything that must be identical for two runs to be pooled or paired.

    The freeze pins frozen_hash(); the tools digest is inside it.
    """

    model: str = HEADLINE_MODEL
    effort: str = FROZEN_EFFORT
    thinking: str = FROZEN_THINKING
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    max_tool_calls: int = MAX_TOOL_CALLS_PER_RUN
    max_executions: int = 1
    # Disclosed round cap R; 1 is the one-shot arm. Arm-specific: must be listed in
    # arms.budget.ARM_SPECIFIC_KEYS or the arms' invariant digests diverge.
    max_rounds: int = 1
    # Tools withheld from BOTH arms; arm-invariant so mismatched sets refuse to pair.
    withheld_tools: tuple[str, ...] = ()
    arm: str = "no-evaluator-feedback"
    # Per-run USD ceiling on projected total cost; None leaves the token caps as the bound.
    max_run_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "model": self.model,
            "effort": self.effort,
            "thinking": self.thinking,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_executions": self.max_executions,
            "max_rounds": self.max_rounds,
            "withheld_tools": list(self.withheld_tools),
            "max_run_usd": self.max_run_usd,
            "server_side_fallbacks_enabled": SERVER_SIDE_FALLBACKS_ENABLED,
            "replicate_semantics": (
                "independent identical requests, clustered within signal; not a seeded "
                "reproduction and not independent across replicates of one signal"
            ),
            "harness_policy_version": HARNESS_POLICY_VERSION,
            "provider_policy_version": PROVIDER_POLICY_VERSION,
            "accounting_policy_version": ACCOUNTING_POLICY_VERSION,
            "price_table_date": PRICE_TABLE_DATE.isoformat(),
            "feedback_policy_version": FEEDBACK_POLICY_VERSION,
            "ast_policy_version": AST_POLICY_VERSION,
            "sandbox_policy_version": SANDBOX_POLICY_VERSION,
            # Hashes the rendered block, budget disclosure included.
            "system_prompt_sha256": sha256(self.system_text().encode("utf-8")).hexdigest(),
            "receiver_contract_sha256": sha256(
                self.receiver_contract().encode("utf-8")
            ).hexdigest(),
            "tool_budget_disclosed_to_model": True,
            **tools_config(),
        }

    def offered_tool_schemas(self) -> tuple[dict[str, Any], ...]:
        """The schemas this configuration's runs actually offer the model."""
        from .tools import TOOL_NAMES, TOOL_SCHEMAS

        unknown = set(self.withheld_tools) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"withheld_tools names tools that do not exist: {sorted(unknown)}")
        return tuple(
            schema for schema in TOOL_SCHEMAS if schema["name"] not in self.withheld_tools
        )

    @property
    def iterative(self) -> bool:
        # Derived from the round cap so the prompt cannot disagree with it.
        return self.max_rounds > 1

    def receiver_contract(self) -> str:
        return ITERATIVE_RECEIVER_CONTRACT if self.iterative else RECEIVER_CONTRACT

    def system_text(self) -> str:
        """Rendered system block, with withheld tools' bullets removed."""
        if self.iterative:
            rendered = iterative_system_text(
                self.max_rounds, self.max_tool_calls, MAX_TOOL_RESULT_BYTES
            )
        else:
            rendered = system_text(self.max_tool_calls, MAX_TOOL_RESULT_BYTES)
        return _without_tool_bullets(rendered, self.withheld_tools)

    def frozen_hash(self) -> str:
        return sha256(canonical_json(self.as_dict())).hexdigest()


# --- the run ------------------------------------------------------------------------------
@dataclass
class _Transcript:
    """The agent-visible surface, in one object so the wall tests can scan all of it."""

    system: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "messages": self.messages,
            "tool_results": self.tool_results,
        }


def _system_blocks(settings: "AgentConfig") -> list[dict[str, Any]]:
    # One cache breakpoint on the only system block; render order tools -> system -> messages
    # means it covers the tool definitions too.
    return [
        {
            "type": "text",
            "text": settings.system_text(),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _tool_result_block(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": result.serialized}],
        "is_error": not result.ok,
    }


def _parse_submission(response: ProviderResponse) -> tuple[dict[str, Any] | None, str | None]:
    # No text at all is a scored failure to submit; text that should have parsed but did
    # not is the structured-output contract failing, which is run_invalid.
    candidates = [text for text in response.text_blocks() if text.strip()]
    if not candidates:
        return None, "no_receiver_submitted"
    for text in candidates:
        try:
            value = json.loads(text.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("receiver_source"), str):
            return value, None
    return None, "structured_output_parse_failure"


def _new_run_dir(root: Path, capture_id: str) -> tuple[str, Path]:
    # A directory name is a published value; refuse one that carries sealed identity.
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe = "".join(c for c in capture_id if c.isalnum() or c in "-_")
    if not safe or safe != capture_id:
        raise ValueError("capture_id contains unsafe path characters")
    run_id = f"agent-{timestamp}-{secrets.token_hex(6)}-{safe}"
    run_dir = root / run_id
    refuse_sealed_identity(
        {},
        path=run_dir,
        because="a run directory is minted in the repository before any record is written",
    )
    os.mkdir(run_dir)
    return run_id, run_dir


def _total(usages: Sequence[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for usage in usages:
        total = total + usage
    return total


def cache_effect_warnings(usages: Sequence[TokenUsage]) -> list[str]:
    # The first call legitimately reads nothing; a zero read after that means the stable
    # prefix moved.
    warnings: list[str] = []
    for index, usage in enumerate(usages[1:], start=2):
        if usage.cache_read_tokens == 0 and usage.prompt_tokens_total > 0:
            warnings.append(
                f"call {index} read zero cached tokens: the stable prefix is not stable"
            )
    return warnings


# What an agent run publishes, declared as data; run_agent assembles honest documents and
# write_record applies these policies.

AGENT_PUBLIC_POLICY = RecordPolicy(
    name="agent-run-public",
    operations=(
        Substitute(key="capture_id", template="sealed:{capture_ref}", when=SEALED),
        # Idempotent with the feedback wall's 1/64 quantization; kept so an off-grid BER
        # arriving through any future path cannot survive.
        Redact(key="feedback", redactor="evaluator_feedback", when=SEALED),
    ),
)

TRACE_PUBLIC_POLICY = RecordPolicy(
    name="agent-trace-public",
    operations=(
        Substitute(key="capture_id", template="sealed:{capture_ref}", when=SEALED),
        Redact(key="feedback", redactor="evaluator_feedback", when=SEALED),
        # receiver_sha256 is derived from the sealed capture; it must not appear in the repo.
        Null(keys=("receiver_sha256",), when=SEALED),
    ),
)

# Identity a sealed run's trace may not state in the repository. RecordPolicy refuses to
# exist unless every key here is nulled for a sealed run.
TRACE_INTERNAL_IDENTITY = (
    "capture_id",
    "receiver_sha256",
    "receiver_source",
    "approach",
    "aligned_ber_unrounded",
)

TRACE_INTERNAL_POLICY = RecordPolicy(
    name="agent-trace-internal",
    identity=TRACE_INTERNAL_IDENTITY,
    operations=(
        Null(keys=TRACE_INTERNAL_IDENTITY, when=SEALED),
        Redact(key="sandbox", redactor="sandbox", when=SEALED),
        Redact(key="evaluator_internal", redactor="evaluator_internal", when=SEALED),
    ),
)

# Identity a sealed agent run may not state in the repository; same guarantee as above.
AGENT_INTERNAL_IDENTITY = (
    "capture_id",
    "capture_dir",
    "receiver_sha256",
    "approach",
    "aligned_ber_unrounded",
)

AGENT_INTERNAL_POLICY = RecordPolicy(
    name="agent-run-internal",
    identity=AGENT_INTERNAL_IDENTITY,
    operations=(
        # Both halves of the sandbox result are recorded elsewhere; the envelope keeps neither.
        Drop(keys=("feedback", "internal"), at=("sandbox",), when=ALWAYS),
        Null(keys=AGENT_INTERNAL_IDENTITY, when=SEALED),
        Redact(key="sandbox", redactor="sandbox", when=SEALED),
        # Blanks the private framing truth while keeping packet_success.
        Redact(key="evaluator_internal", redactor="evaluator_internal", when=SEALED),
        Constant(key="transcript_location", value="sealed_run_artifacts", when=SEALED),
    ),
)


def run_agent(
    capture_dir: str | os.PathLike[str],
    provider: Provider,
    *,
    config: AgentConfig | None = None,
    run_root: str | os.PathLike[str] = RUNS_ROOT,
    executor: Callable[..., dict[str, Any]] = run_receiver,
    sealed_roots: Iterable[str | os.PathLike[str]] = (),
    permitted_capture_parents: Iterable[str | os.PathLike[str]] = (),
    preflight_sandbox: bool = True,
    sealed_token: Any = None,
    abort_on_sandbox_unavailable: bool = True,
    run_date: date | None = None,
) -> dict[str, Any]:
    """Run one capture end to end and return the orchestrator record.

    Returns a ``public`` half and an ``internal`` half; only ``public.feedback`` came from
    the evaluator, via forward_feedback. ``run_date`` defaults to today (UTC) so each run is
    priced against the rate table on its own date.
    """
    settings = config or AgentConfig()
    priced_on = run_date or datetime.now(timezone.utc).date()
    # Fail before the first request if the model cannot be priced.
    rates_for(settings.model, priced_on)
    capture = Path(capture_dir)
    meta = _public_metadata(capture)
    capture_id = str(meta.get("capture_id") or "")
    loaded: CaptureSignal = load_capture(capture)
    # Not authorize_read: that logs a sealed read, and run_receiver already makes it.
    sealing = sealing_of_capture(capture)
    sealed = sealing.sealed
    capture_ref = sealing.capture_ref

    transcript = _Transcript(system=_system_blocks(settings))
    task = build_task_message(meta)
    transcript.messages.append({"role": "user", "content": [{"type": "text", "text": task}]})

    budget = ExecutionBudget(settings.max_executions)
    execute_once = guarded_executor(executor, budget)
    usages: list[TokenUsage] = []
    tool_calls: list[dict[str, Any]] = []
    provider_invalid: str | None = None
    harness_reason: str | None = None
    submission: dict[str, Any] | None = None
    served_models: list[str] = []
    stop_reasons: list[str | None] = []

    while True:
        # output_format rides on every turn; the harness cannot know which turn submits.
        request = ProviderRequest(
            model=settings.model,
            system=tuple(transcript.system),
            messages=tuple(transcript.messages),
            tools=settings.offered_tool_schemas(),
            max_tokens=settings.max_output_tokens,
            effort=settings.effort,
            thinking=settings.thinking,
            output_format=RECEIVER_SUBMISSION_SCHEMA,
        )
        response = provider.complete(request)
        usages.append(response.usage)
        served_models.append(response.model)
        stop_reasons.append(response.stop_reason)
        if response.invalid_reason is not None:
            provider_invalid = response.invalid_reason
            break
        if settings.max_run_usd is not None:
            spent = cost_usd(_total(usages), model=settings.model, on=priced_on)
            if spent > settings.max_run_usd:
                harness_reason = "cost_cap_exceeded"
                break
        transcript.messages.append({"role": "assistant", "content": list(response.content)})
        uses = response.tool_uses()
        if not uses:
            submission, harness_reason = _parse_submission(response)
            break
        if len(tool_calls) + len(uses) > settings.max_tool_calls:
            harness_reason = "tool_budget_exhausted"
            break
        blocks: list[dict[str, Any]] = []
        for use in uses:
            name = str(use.get("name", ""))
            arguments = use.get("input") or {}
            if name in settings.withheld_tools:
                # Executing a withheld tool on a hallucinated call would leak the withheld
                # measurement; treat it as a tool that does not exist.
                result = _withheld_tool_result(name)
            else:
                result = call_tool(name, arguments, loaded)
            tool_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "ok": result.ok,
                    "error": result.error,
                    "result_bytes": result.size_bytes,
                }
            )
            transcript.tool_results.append({"name": name, "content": result.serialized})
            blocks.append(_tool_result_block(str(use.get("id", "")), result))
        transcript.messages.append({"role": "user", "content": blocks})

    if submission is None and harness_reason is None and provider_invalid is None:
        harness_reason = "no_receiver_submitted"

    total_usage = _total(usages)
    # A sealed capture's id must not name a directory under runs/; use the opaque reference.
    run_id, run_dir = _new_run_dir(
        Path(run_root).resolve() / AGENT_RUNS_SUBDIR,
        (capture_ref if sealed else capture_id) or "unknown",
    )
    artifact_dir = sealed_run_artifact_dir(run_id, sealed_token) if sealed else run_dir
    sandbox: dict[str, Any] = {}
    feedback: dict[str, Any] = {}
    internal_evaluator: dict[str, Any] = {}
    evaluator_error: str | None = None
    packet_success = False
    receiver_sha256: str | None = None
    aligned_ber_unrounded: float | None = None

    if submission is not None:
        source = submission["receiver_source"].encode("utf-8")
        receiver_sha256 = sha256(source).hexdigest()
        (artifact_dir / f"receiver-{receiver_sha256}.py").write_bytes(source)
        with tempfile.TemporaryDirectory(prefix="modembench-agent-") as scratch:
            receiver_path = Path(scratch) / "receiver.py"
            receiver_path.write_bytes(source)
            sandbox = execute_once(
                capture,
                receiver_path,
                run_root=Path(run_root),
                sealed_roots=tuple(sealed_roots),
                permitted_capture_parents=tuple(permitted_capture_parents),
                preflight=preflight_sandbox,
                sealed_token=sealed_token,
            )
        status = str(sandbox.get("status"))
        if status == "sandbox_unavailable" and abort_on_sandbox_unavailable:
            raise SandboxUnavailableAbort(
                "sandbox_unavailable during an agent run: on a machine where this fires it "
                "fires for every run, and the sweep would report a confident 0%. "
                f"detail: {sandbox.get('error')!r}"
            )
        internal_evaluator = dict(sandbox.get("internal") or {})
        packet_success = internal_evaluator.get("packet_success") is True
        raw_feedback = dict(sandbox.get("feedback") or {})
        # Preserve the unrounded BER for analysis; the agent-visible value is quantized.
        raw_ber = raw_feedback.get("aligned_ber")
        if isinstance(raw_ber, (int, float)) and not isinstance(raw_ber, bool):
            aligned_ber_unrounded = float(raw_ber)
        try:
            feedback = forward_feedback({"feedback": raw_feedback})
        except FeedbackWallError as exc:
            harness_reason = "feedback_wall_violation"
            feedback = {}
            internal_evaluator["feedback_wall_error"] = str(exc)
        if feedback.get("error") == "evaluator_invalid":
            evaluator_error = "evaluator_invalid"

    outcome = classify_outcome(
        sandbox_status=str(sandbox.get("status")) if sandbox else None,
        packet_success=packet_success,
        sandbox_error=(sandbox.get("error") if sandbox else None),
        provider_invalid_reason=provider_invalid,
        harness_reason=harness_reason,
        evaluator_error=evaluator_error,
    )

    # Nothing below is redacted by hand; write_record applies the two policies above.
    public_document = {
        "run_id": run_id,
        "capture_id": capture_id,
        "arm": settings.arm,
        "model_requested": settings.model,
        "tool_calls": tool_calls,
        "feedback": feedback,
        "outcome": outcome.as_dict(),
    }
    cost = cost_breakdown(total_usage, model=settings.model, on=priced_on)
    # Built through the policy so an unclassified new field fails here, not ships unredacted.
    identity = AGENT_INTERNAL_POLICY.identity_fields(
        capture_id=capture_id,
        capture_dir=str(capture.resolve()),
        receiver_sha256=receiver_sha256,
        approach=(submission or {}).get("approach"),
        aligned_ber_unrounded=aligned_ber_unrounded,
    )
    internal_document = {
        "run_id": run_id,
        **identity,
        "capture_ref": capture_ref,
        "sealed": sealed,
        # Relative: an absolute form would put the sealed root's path into the repository.
        "sealed_run_record": (
            str(sealed_run_record_path(run_id, ".").relative_to(".")) if sealed else None
        ),
        "config": settings.as_dict(),
        "frozen_config_sha256": settings.frozen_hash(),
        "tools_sha256": tools_sha256(),
        "provider": getattr(provider, "name", type(provider).__name__),
        "models_served": served_models,
        "stop_reasons": stop_reasons,
        "executions": budget.spent,
        "execution_cap": budget.limit,
        "sandbox": sandbox,
        "evaluator_internal": internal_evaluator,
        "usage": total_usage.as_dict(),
        "cost": cost,
        "priced_on": priced_on.isoformat(),
        "cache_warnings": cache_effect_warnings(usages),
        "feedback_wall": feedback_config(),
        "outcome": outcome.as_dict(),
        "transcript_location": "orchestrator",
    }
    orchestrator_dir = run_dir / ".orchestrator"
    orchestrator_dir.mkdir(mode=0o700, exist_ok=True)
    if sealed:
        # Written before the repository records so a crash between them leaves a resolvable
        # run. Unredacted: it lives inside the sealed root.
        write_sealed_run_record(
            run_id,
            {
                "run_id": run_id,
                **identity,
                "capture_ref": capture_ref,
                "sandbox": sandbox,
                "evaluator_internal": internal_evaluator,
                "transcript": transcript.as_dict(),
                "artifact_dir": str(artifact_dir),
            },
            sealed_token,
        )
        write_json_once(
            artifact_dir / "transcript.json", transcript.as_dict(), description=_ARTIFACT
        )
    else:
        write_json_once(
            orchestrator_dir / "transcript.json", transcript.as_dict(), description=_ARTIFACT
        )
    # write_record returns what it wrote, so callers only ever see the redacted halves.
    public = write_record(
        run_dir / "agent-run.json",
        public_document,
        AGENT_PUBLIC_POLICY,
        sealing=sealing,
        description=_ARTIFACT,
    )
    internal = write_record(
        orchestrator_dir / "agent-run-internal.json",
        internal_document,
        AGENT_INTERNAL_POLICY,
        sealing=sealing,
        description=_ARTIFACT,
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "public": public,
        "internal": internal,
        "transcript": transcript.as_dict(),
        "outcome": outcome,
    }
