"""Iterative arm: author, execute, read Tier-2 feedback, repair."""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

from ..records import (
    append_record,
    sealing_of_capture,
    trace_head_sha256,
    write_json_once,
    write_record,
)
from ..sandbox.runner import RUNS_ROOT, run_receiver
from ..sealed import (
    sealed_run_artifact_dir,
    sealed_run_record_path,
    write_sealed_run_record,
)
from .accounting import TokenUsage, cost_breakdown, cost_usd, rates_for
from .feedback import FeedbackWallError, feedback_config, forward_feedback
from .harness import (
    AGENT_INTERNAL_POLICY,
    AGENT_PUBLIC_POLICY,
    AGENT_RUNS_SUBDIR,
    MAX_ROUND_RERUNS,
    ROUND_COMPOSITION_RULE,
    TRACE_INTERNAL_POLICY,
    TRACE_POLICY_VERSION,
    TRACE_PUBLIC_POLICY,
    AgentConfig,
    ExecutionBudget,
    RECEIVER_SUBMISSION_SCHEMA,
    Outcome,
    RoundRerunCapExceeded,
    SandboxUnavailableAbort,
    _new_run_dir,
    _parse_submission,
    _public_metadata,
    _system_blocks,
    _Transcript,
    _tool_result_block,
    _total,
    build_task_message,
    cache_effect_warnings,
    classify_outcome,
    compose_rounds,
    guarded_executor,
)
from .provider import Provider, ProviderRequest
from .tools import TOOL_SCHEMAS, CaptureSignal, call_tool, load_capture, tools_sha256

_ARTIFACT = "agent run artifact"
# The construct this arm measures, carried as data in every record it writes.
MEASURED_CONSTRUCT: dict[str, Any] = {
    "name": "fresh-context, feedback-conditioned revision of a prior same-model artifact",
    "short_name": "single-repair",
    "may_claim": (
        "that supplying the repair packet — the prior receiver, its four-field score and its "
        "one-line approach note — changes the packet-success rate of the next sampled "
        "solution by Delta."
    ),
    "must_not_claim": (
        "conversational iteration; self-reflection; learning from one's own reasoning; "
        "convergence; multi-step revision beyond a single repair; anything about what "
        "preserving the prior reasoning would or would not have done; or any behaviour at a "
        "round count the budget did not fund."
    ),
    "why": (
        "round 2 is a fresh context. It carries the code and the result, not the assistant "
        "turn that produced them, because carrying that turn costs ~2.76x the available "
        "budget headroom and would make R = 1 at the frozen N. The model therefore has repair "
        "evidence and no experiential continuity."
    ),
    "decided_by": (
        "independent review of the arm's construct; resolves the open "
        "escalated 'iterative or single-repair' question toward single-repair"
    ),
}

PUBLIC_TRACE_NAME = "trace.jsonl"
INTERNAL_TRACE_NAME = "trace-internal.jsonl"


def _repair_message(
    *,
    round_index: int,
    max_rounds: int,
    previous_source: str,
    previous_approach: str | None,
    feedback: Mapping[str, Any],
) -> str:
    # The whole of what a repair round inherits.
    approach = (previous_approach or "").strip() or "(not recorded)"
    lines = [
        f"=== ROUND {round_index} OF {max_rounds} — REPAIR ===",
        "",
        f"Your round-{round_index - 1} receiver was executed. It did not recover the packet.",
        "",
        "What it scored — this is the complete result, there is nothing else:",
        "",
        f"  acquisition_success : {feedback.get('acquisition_success')!r}",
        f"  crc_pass            : {feedback.get('crc_pass')!r}",
        f"  aligned_ber         : {feedback.get('aligned_ber')!r}"
        "   (quantized to a multiple of 1/64; null when acquisition failed)",
        f"  error               : {feedback.get('error')!r}",
        "",
        f"Your stated approach was: {approach}",
        "",
        "The receiver you submitted, verbatim:",
        "",
        "```python",
        previous_source.rstrip("\n"),
        "```",
        "",
        "The measurements above this message are the same recording as before; the tools are "
        "not available again and there is nothing further to measure. Diagnose from the result "
        "and the code, then submit a complete replacement receiver.py — not a patch, not a "
        "diff, the whole module.",
    ]
    if round_index >= max_rounds:
        lines.append("")
        lines.append(
            f"This is round {max_rounds}, the last one. There is no further attempt after this."
        )
    return "\n".join(lines)


def _tool_results_block(results: Iterable[Mapping[str, Any]]) -> str:
    # Round 1's measurements, re-rendered for the fresh round-2 context.
    parts = ["=== YOUR ROUND-1 MEASUREMENTS (verbatim) ==="]
    for entry in results:
        parts.append(f"$ {entry['name']}\n{entry['content']}")
    return "\n\n".join(parts)


def run_agent_iterative(
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
    max_round_reruns: int = MAX_ROUND_RERUNS,
) -> dict[str, Any]:
    """Run one capture through the R-round feedback loop. Same return shape as run_agent,
    plus ``rounds``."""
    settings = config or AgentConfig()
    if not settings.iterative:
        raise ValueError(
            f"run_agent_iterative needs max_rounds > 1; got {settings.max_rounds}. At R = 1 "
            "the arm under test does not exist — use run_agent, whose system block correctly "
            "states the one-shot condition."
        )
    priced_on = run_date or datetime.now(timezone.utc).date()
    # Fail before the first request if the model cannot be priced.
    rates_for(settings.model, priced_on)
    capture = Path(capture_dir)
    meta = _public_metadata(capture)
    capture_id = str(meta.get("capture_id") or "")
    loaded: CaptureSignal = load_capture(capture)
    sealing = sealing_of_capture(capture)
    sealed = sealing.sealed
    capture_ref = sealing.capture_ref

    task = build_task_message(meta)
    system_blocks = _system_blocks(settings)

    # The structural budget is R plus the re-run allowance, because an invalid round may
    # have spent an execution before the runner attributed the failure. R caps valid rounds.
    budget = ExecutionBudget(settings.max_rounds + max_round_reruns)
    execute = guarded_executor(executor, budget)
    valid_rounds: list[Outcome] = []
    rounds_issued = 0
    round_reruns = 0

    run_id, run_dir = _new_run_dir(
        Path(run_root).resolve() / AGENT_RUNS_SUBDIR,
        (capture_ref if sealed else capture_id) or "unknown",
    )
    artifact_dir = sealed_run_artifact_dir(run_id, sealed_token) if sealed else run_dir
    orchestrator_dir = run_dir / ".orchestrator"
    orchestrator_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    public_trace = run_dir / PUBLIC_TRACE_NAME
    internal_trace = artifact_dir / INTERNAL_TRACE_NAME

    transcript = _Transcript(system=list(system_blocks))
    usages: list[TokenUsage] = []
    round_usages: list[TokenUsage] = []
    all_tool_calls: list[dict[str, Any]] = []
    round_one_tool_results: list[dict[str, Any]] = []
    served_models: list[str] = []
    stop_reasons: list[str | None] = []

    rounds_detail: list[dict[str, Any]] = []
    matched_usd = 0.0
    gross_usd = 0.0

    # The complete inter-round carry; nothing else crosses rounds.
    previous_source: str | None = None
    previous_approach: str | None = None
    previous_feedback: dict[str, Any] = {}

    final_sandbox: dict[str, Any] = {}
    final_feedback: dict[str, Any] = {}
    final_internal_evaluator: dict[str, Any] = {}
    final_receiver_sha256: str | None = None
    final_approach: str | None = None
    final_ber_unrounded: float | None = None
    round_one_status: str | None = None

    while len(valid_rounds) < settings.max_rounds:
        round_index = len(valid_rounds) + 1
        rounds_issued += 1
        first_round = round_index == 1

        # Round 1 is a tool-using conversation. Round >= 2 is a fresh-context turn carrying
        # the code, the result and the round-1 measurements; tools=() makes that structural.
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": task}]}
        ]
        if not first_round:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _tool_results_block(round_one_tool_results)}
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _repair_message(
                                round_index=round_index,
                                max_rounds=settings.max_rounds,
                                previous_source=previous_source or "",
                                previous_approach=previous_approach,
                                feedback=previous_feedback,
                            ),
                        }
                    ],
                }
            )

        offered_tools = settings.offered_tool_schemas() if first_round else ()
        round_tool_calls: list[dict[str, Any]] = []
        round_usage_list: list[TokenUsage] = []
        provider_invalid: str | None = None
        harness_reason: str | None = None
        submission: dict[str, Any] | None = None

        while True:
            request = ProviderRequest(
                model=settings.model,
                system=tuple(system_blocks),
                messages=tuple(messages),
                tools=offered_tools,
                max_tokens=settings.max_output_tokens,
                effort=settings.effort,
                thinking=settings.thinking,
                output_format=RECEIVER_SUBMISSION_SCHEMA,
            )
            response = provider.complete(request)
            usages.append(response.usage)
            round_usage_list.append(response.usage)
            served_models.append(response.model)
            stop_reasons.append(response.stop_reason)
            if response.invalid_reason is not None:
                provider_invalid = response.invalid_reason
                break
            if settings.max_run_usd is not None:
                spent = cost_usd(_total(usages), model=settings.model, on=priced_on)
                if spent > settings.max_run_usd:
                    # Undisclosed backstop; stays run_invalid.
                    harness_reason = "cost_cap_exceeded"
                    break
            messages.append({"role": "assistant", "content": list(response.content)})
            uses = response.tool_uses()
            if not uses:
                submission, harness_reason = _parse_submission(response)
                break
            if not first_round:
                # Unreachable while tools=() holds; a call to an un-offered tool is a
                # transport fault, not a scored failure.
                harness_reason = "transport_error"
                break
            if len(all_tool_calls) + len(round_tool_calls) + len(uses) > settings.max_tool_calls:
                harness_reason = "tool_budget_exhausted"
                break
            blocks: list[dict[str, Any]] = []
            for use in uses:
                name = str(use.get("name", ""))
                arguments = use.get("input") or {}
                if name in settings.withheld_tools:
                    # Same guard as run_agent: a withheld tool must not execute.
                    from .harness import _withheld_tool_result

                    result = _withheld_tool_result(name)
                else:
                    result = call_tool(name, arguments, loaded)
                round_tool_calls.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "ok": result.ok,
                        "error": result.error,
                        "result_bytes": result.size_bytes,
                    }
                )
                round_one_tool_results.append({"name": name, "content": result.serialized})
                transcript.tool_results.append({"name": name, "content": result.serialized})
                blocks.append(_tool_result_block(str(use.get("id", "")), result))
            messages.append({"role": "user", "content": blocks})

        if submission is None and harness_reason is None and provider_invalid is None:
            harness_reason = "no_receiver_submitted"

        # --- execute this round ------------------------------------------------------------
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
            (artifact_dir / f"receiver-r{round_index}-{receiver_sha256}.py").write_bytes(source)
            with tempfile.TemporaryDirectory(prefix="modembench-agent-") as scratch:
                receiver_path = Path(scratch) / "receiver.py"
                receiver_path.write_bytes(source)
                sandbox = execute(
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
        round_usage = _total(round_usage_list)
        round_cost = cost_usd(round_usage, model=settings.model, on=priced_on)
        gross_usd += round_cost
        invalid = outcome.kind == "run_invalid"
        if not invalid:
            matched_usd += round_cost

        # --- trace both halves, per round --------------------------------------------------
        common = {
            "run_id": run_id,
            "round": round_index,
            "attempt_of_round": round_reruns if invalid else 0,
            "issued": rounds_issued,
            "outcome": outcome.as_dict(),
            "trace_policy_version": TRACE_POLICY_VERSION,
        }
        append_record(
            public_trace,
            {
                **common,
                "capture_id": capture_id,
                "tool_calls": round_tool_calls,
                # Already through the wall; the trace never touches the evaluator's object.
                "feedback": feedback,
                "receiver_sha256": receiver_sha256,
                "usage": round_usage.as_dict(),
                "cost_usd": round_cost,
                "counted_toward_matched_spend": not invalid,
            },
            TRACE_PUBLIC_POLICY,
            sealing=sealing,
        )
        append_record(
            internal_trace,
            {
                **common,
                **TRACE_INTERNAL_POLICY.identity_fields(
                    capture_id=capture_id,
                    receiver_sha256=receiver_sha256,
                    receiver_source=(submission or {}).get("receiver_source"),
                    approach=(submission or {}).get("approach"),
                    aligned_ber_unrounded=aligned_ber_unrounded,
                ),
                "sandbox": sandbox,
                "evaluator_internal": internal_evaluator,
                "usage": round_usage.as_dict(),
                "cost_usd": round_cost,
            },
            TRACE_INTERNAL_POLICY,
            sealing=sealing,
        )

        if invalid:
            # Re-issued in place; does not consume a round of R, spend is gross not matched.
            round_reruns += 1
            if round_reruns > max_round_reruns:
                raise RoundRerunCapExceeded(
                    f"round {round_index} of run {run_id} returned run_invalid "
                    f"{round_reruns} times (cap {max_round_reruns}); last reason "
                    f"{outcome.reason!r}. The whole run is abandoned rather than scored: an "
                    "unattributable failure booked against the model measures the harness."
                )
            continue

        valid_rounds.append(outcome)
        rounds_detail.append(
            {
                "round": round_index,
                "outcome": outcome.as_dict(),
                "cost_usd": round_cost,
                "usage": round_usage.as_dict(),
                "tool_calls": len(round_tool_calls),
            }
        )
        round_usages.append(round_usage)
        all_tool_calls.extend(round_tool_calls)
        if round_index == 1:
            round_one_status = outcome.status
        final_sandbox = sandbox
        final_feedback = feedback
        final_internal_evaluator = internal_evaluator
        final_receiver_sha256 = receiver_sha256
        final_approach = (submission or {}).get("approach")
        final_ber_unrounded = aligned_ber_unrounded
        if outcome.kind == "scored_success":
            break
        previous_source = (submission or {}).get("receiver_source")
        previous_approach = final_approach
        previous_feedback = dict(feedback)
        if previous_source is None:
            # No module to repair from; stop with the failure this round earned.
            break

    outcome = compose_rounds(valid_rounds, round_cap=settings.max_rounds)
    total_usage = _total(usages)

    transcript.messages = list(messages)
    public_head = trace_head_sha256(public_trace)
    internal_head = trace_head_sha256(internal_trace)

    rounds_summary = {
        "round_cap": settings.max_rounds,
        "rounds_issued": rounds_issued,
        "rounds_valid": len(valid_rounds),
        "round_reruns": round_reruns,
        "reached_round_cap": len(valid_rounds) >= settings.max_rounds
        and outcome.kind != "scored_success",
        "round_one_status": round_one_status,
        # The analysis decomposes Delta by this without re-reading R traces.
        "round_one_failure_status": (
            round_one_status if round_one_status not in (None, "ok") or len(valid_rounds) > 1
            else None
        ),
        # TH-19 clause 2 / pre-registration section 6 step 1: three-way decomposition.
        "decomposition": {
            "succeeded_in_round_one": len(valid_rounds) == 1
            and outcome.kind == "scored_success",
            "terminated_inside_round_one": len(valid_rounds) == 1
            and outcome.kind != "scored_success",
            "executed_two_or_more_rounds": len(valid_rounds) >= 2,
        },
        "composition_rule": list(ROUND_COMPOSITION_RULE),
        "measured_construct": dict(MEASURED_CONSTRUCT),
    }
    spend = {
        # Matched sums the valid rounds; gross adds the re-issued ones.
        "matched_spend_usd": matched_usd,
        "gross_spend_usd": gross_usd,
        "rerun_spend_usd": gross_usd - matched_usd,
        "budget_per_signal_usd": None,
        "reporting_rule": (
            "reported in both directions, triggers in one. The only quantities this arm may "
            "make a TRIGGER out of are rounds_used > R and matched_spend > B x (1 + tol) — "
            "both over-consumption, both ex-ante cap violations. Nothing emitted here fires "
            "on low. Every diagnostic defined on how much of its budget the loop CONSUMED "
            "moves toward firing as the loop gets BETTER, because a working loop stops early "
            "on success: at R = N = 2 the expected matched-spend ratio is 1 - q1/2, i.e. "
            "0.888-0.933 at the per-attempt rate consistent with the calibration band. A "
            "ratio of 0.9 is the loop working, not the instrument failing."
        ),
    }

    public_document = {
        "run_id": run_id,
        "capture_id": capture_id,
        "arm": settings.arm,
        "model_requested": settings.model,
        "tool_calls": all_tool_calls,
        "feedback": final_feedback,
        "outcome": outcome.as_dict(),
        "rounds": rounds_summary,
        "spend": spend,
        "trace": {
            "path": PUBLIC_TRACE_NAME,
            "head_sha256": public_head,
            "trace_policy_version": TRACE_POLICY_VERSION,
        },
    }
    cost = cost_breakdown(total_usage, model=settings.model, on=priced_on)
    identity = AGENT_INTERNAL_POLICY.identity_fields(
        capture_id=capture_id,
        capture_dir=str(capture.resolve()),
        receiver_sha256=final_receiver_sha256,
        approach=final_approach,
        aligned_ber_unrounded=final_ber_unrounded,
    )
    internal_document = {
        "run_id": run_id,
        **identity,
        "capture_ref": capture_ref,
        "sealed": sealed,
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
        "execution_cap_note": (
            "R + max_round_reruns, not R. The structural guard has to cover a round that "
            "entered the sandbox before the runner attributed its failure to the machine; R "
            "caps VALID rounds, which is rounds.rounds_valid."
        ),
        "sandbox": final_sandbox,
        "evaluator_internal": final_internal_evaluator,
        "usage": total_usage.as_dict(),
        "per_round_usage": [entry["usage"] for entry in rounds_detail],
        "cost": cost,
        "priced_on": priced_on.isoformat(),
        "cache_warnings": cache_effect_warnings(usages),
        "feedback_wall": feedback_config(),
        "outcome": outcome.as_dict(),
        "rounds": rounds_summary,
        "rounds_detail": rounds_detail,
        "spend": spend,
        "trace": {
            "public_path": PUBLIC_TRACE_NAME,
            "public_head_sha256": public_head,
            "internal_head_sha256": internal_head,
            "trace_policy_version": TRACE_POLICY_VERSION,
        },
        "transcript_location": "orchestrator",
    }

    if sealed:
        write_sealed_run_record(
            run_id,
            {
                "run_id": run_id,
                **identity,
                "capture_ref": capture_ref,
                "sandbox": final_sandbox,
                "evaluator_internal": final_internal_evaluator,
                "transcript": transcript.as_dict(),
                "artifact_dir": str(artifact_dir),
                "internal_trace": str(internal_trace),
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
        "rounds": rounds_summary,
    }
