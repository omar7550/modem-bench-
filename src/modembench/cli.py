"""Top-level ModemBench command-line interface."""

from __future__ import annotations

import argparse
from hashlib import sha256
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable

import numpy as np

from .agent.characterize import default_report, write_characterization
from .agent.harness import AgentConfig, preflight, run_agent
from .agent.provider import AnthropicProvider, HEADLINE_MODEL, ProviderError, ReplayProvider
from .diagnostics import diagnose
from .evaluate import encode_truth
from .framing import build_frame
from .gate import run_gate
from .generator import CaptureConflictError, generate_capture
from .records import (
    SEALED,
    Constant,
    Null,
    RecordPolicy,
    sealing_of_location,
    write_record,
)
from .reference_rx import decode
from .sandbox.oracle_source import materialize_oracle_source
from .sandbox.runner import replay_run, run_receiver
from .sealed import (
    MANIFEST_ARTIFACT,
    PAYLOAD_ARTIFACT,
    capture_reference,
    open_sealed,
    read_private_artifact,
    sealed_root_containing,
)
from .splits import (
    VERDICT_VERIFIED,
    dev_split_root,
    load_commitment,
    materialize_split,
    sealed_split_spec,
    verify_split,
)


def _spine(seed: int, outdir: Path, profile: str) -> int:
    capture = generate_capture(seed, outdir / "captures", profile=profile)
    # Truth is read once, through the chokepoint, and handed to the evaluator on stdin;
    # never as a private path in the child's argv.
    manifest_bytes = read_private_artifact(capture.capture_dir, MANIFEST_ARTIFACT)
    payload = read_private_artifact(capture.capture_dir, PAYLOAD_ARTIFACT)
    manifest = json.loads(manifest_bytes)
    waveform = manifest["waveform"]
    framing = manifest["framing"]
    impairments = manifest["impairments"]
    iq = np.load(capture.capture_dir / "iq.npy", allow_pickle=False)
    bits = decode(
        iq,
        waveform["sample_rate_hz"],
        sps=waveform["sps"],
        beta=waveform["rrc_beta"],
        offset=waveform["packet_offset_samples"],
        sync_bits=np.asarray(framing["sync_bits"], dtype=np.uint8),
        payload_len=framing["payload_length_bytes"],
        cfo_hz=impairments["cfo"]["applied_value"],
        phase_rad=impairments["phase"]["applied_value"],
        amplitude=impairments["amplitude"]["applied_value"],
        timing_mu=impairments["fractional_timing"]["applied_value"],
        fd_group_delay=impairments["fd_group_delay_samples"],
    )

    run_dir = outdir / "runs" / capture.capture_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bits_path = run_dir / "oracle_bits.npy"
    np.save(bits_path, bits, allow_pickle=False)
    completed = subprocess.run(
        [sys.executable, "-m", "modembench.evaluate", str(bits_path), "--truth-stdin"],
        check=False,
        capture_output=True,
        text=True,
        input=encode_truth(manifest_bytes, payload),
    )
    try:
        evaluated = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"evaluator returned invalid JSON: {completed.stderr.strip()}") from exc
    frame = build_frame(np.asarray(framing["sync_bits"], dtype=np.uint8), payload)
    diagnostics = diagnose(
        iq,
        waveform["sample_rate_hz"],
        sps=waveform["sps"],
        beta=waveform["rrc_beta"],
        offset=waveform["packet_offset_samples"],
        frame_bits=frame,
        sync_len=len(framing["sync_bits"]),
        payload_len=framing["payload_length_bytes"],
        decoded_bits=bits,
        evaluation=evaluated,
        cfo_hz=impairments["cfo"]["applied_value"],
        phase_rad=impairments["phase"]["applied_value"],
        amplitude=impairments["amplitude"]["applied_value"],
        timing_mu=impairments["fractional_timing"]["applied_value"],
        fd_group_delay=impairments["fd_group_delay_samples"],
    )
    (run_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    summary = {
        "capture_id": capture.capture_id,
        "reused": capture.reused,
        "profile": profile,
        "feedback": evaluated["feedback"],
        "internal": evaluated["internal"],
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    expected_code = 0 if evaluated["internal"]["packet_success"] else 1
    if completed.returncode != expected_code:
        return 1
    return expected_code


def _default_captures_root(
    split_id: str, commitments_dir: Path | None, sealed_root: Path | None
) -> Path:
    """Sealed captures live under the sealed root; published splits under captures/<name>."""
    if sealed_root is not None:
        return sealed_root / split_id
    document = load_commitment(split_id, commitments_dir=commitments_dir)
    name = (document or {}).get("split_spec", {}).get("name")
    if isinstance(name, str) and name:
        return dev_split_root().parent / name
    return dev_split_root()


def _verify_split(args: argparse.Namespace) -> int:
    """Report an aggregate verdict only; exit 0 exclusively on `verified`.

    Takes no sealed session and reads no salt; --sealed-root only locates the captures.
    """
    captures_root = args.captures_root
    if captures_root is None:
        captures_root = _default_captures_root(args.split_id, args.commitments_dir, args.sealed_root)
    result = verify_split(
        args.split_id,
        captures_root=captures_root,
        commitments_dir=args.commitments_dir,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["verdict"] == VERDICT_VERIFIED else 1


def _materialize_split(args: argparse.Namespace) -> int:
    """Rebuild a published split's captures so `verify-split` reproduces on a fresh clone."""
    result = materialize_split(
        args.split_id,
        captures_root=args.captures_root,
        commitments_dir=args.commitments_dir,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["matches_commitment"] else 1


def _agent_run(args: argparse.Namespace) -> int:
    """One agent run against one capture. Pre-flight runs before any provider is constructed."""
    report = preflight()
    if not report.ok:
        print(json.dumps({"error": report.error, "preflight": report.as_dict()}, sort_keys=True))
        return 1
    provider = (
        ReplayProvider.from_path(args.replay)
        if args.replay is not None
        else AnthropicProvider(model=args.model)
    )
    result = run_agent(
        args.capture,
        provider,
        config=AgentConfig(model=args.model),
        run_root=args.run_root,
    )
    outcome = result["outcome"]
    print(
        json.dumps(
            {
                "orchestrator_only": True,
                "run_id": result["run_id"],
                "run_dir": result["run_dir"],
                "outcome": outcome.as_dict(),
                "frozen_config_sha256": result["internal"]["frozen_config_sha256"],
                "tools_sha256": result["internal"]["tools_sha256"],
                "total_usd": result["internal"]["cost"]["total_usd"],
                "feedback": result["public"]["feedback"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if outcome.packet_success else 1


# --- the budget-matched arms -----------------------------------------------------------------
# The arms package writes nothing; the driver lives here and persists one document through
# records.write_record.

_ARM_ARTIFACT = "arm run record"

# Identity is the captures root: for a sealed campaign that path is inside the sealed store.
# Everything else is already redacted upstream (sealed:<ref> labels, nulled unrounded BER).
ARM_RUN_POLICY = RecordPolicy(
    name="arm-run",
    identity=("captures_root",),
    operations=(
        Null(keys=("captures_root",), when=SEALED),
        # Sealed attempt artifacts live in the sealed store; the record states the relocation.
        Constant(key="attempt_artifacts_location", value="sealed_run_artifacts", when=SEALED),
    ),
)

SEALED_REFUSAL = (
    "arm-run refuses --split sealed unless the sealed-access path is engaged explicitly. "
    "Reading the sealed split means opening a counted session: `sealed.open_sealed` consumes "
    "one of the authorizations in data/sealed_access.json, the count is not refundable, and "
    "the campaign has two accesses in total. A command that spends one because a flag "
    "defaulted is a defect. Pass --sealed-run-name <an authorized run name> to spend one "
    "deliberately; --split-id defaults to the published sealed split."
)

DEGENERATE_N_REFUSAL = (
    "arm-run refuses to run best-of-N at N = 1. At N = 1 this arm IS the naive single-call "
    "arm: the best-of-n curve has one point, selection has nothing to select between, and "
    "the compute confound the ticket exists to remove is still in the comparison. Running it "
    "anyway would produce a record that says 'best-of-n' and means 'best-of-1'. N is derived "
    "from the ledger residual (modembench.arms.budget.derive_budget -> "
    "modembench.arms.ledger.allocate) and is a budget decision, not a flag: raise "
    "TOTAL_AVAILABLE_USD, set BATCH_SEALED_CAMPAIGN, or cut a ledger line. To exercise the "
    "mechanism at an N the ledger does not fund, pass --unfunded-n K explicitly; that record "
    "is marked funded=false and is not campaign data."
)


#: Sentinel: round cap is the frozen N. An object, not 0, so an unset spec cannot mean "no rounds".
N_ROUNDS_FROM_N = object()


@dataclass(frozen=True)
class _ArmSpec:
    """One arm arm-run can drive, and the AgentConfig.arm label its attempts carry."""

    name: str
    agent_arm: str
    summary: str
    run: Callable[["_ArmContext"], dict[str, Any]]
    #: Rounds per run; above 1 selects the iterative system block via AgentConfig.iterative.
    rounds: int = 1


@dataclass(frozen=True)
class _ArmContext:
    """Everything an arm runner is given."""

    spec: "_ArmSpec"
    captures: tuple[Path, ...]
    n: int
    funded: bool
    config: AgentConfig
    run_root: Path
    provider_for: Callable[[int], Any]
    sealed_token: Any
    sealed_roots: tuple[Path, ...]


def _signal_label(capture_dir: Path) -> str:
    """A sealed capture is named by its opaque reference, never its directory."""
    if sealed_root_containing(capture_dir) is not None:
        return f"sealed:{capture_reference(capture_dir)}"
    return capture_dir.name


def _split_captures(root: Path) -> tuple[Path, ...]:
    """Every capture under a split root, in one deterministic order."""
    if not root.is_dir():
        raise RuntimeError(
            f"no split captures under {root}. Captures are gitignored; rebuild the split with "
            "`materialize-split <split_id>` first."
        )
    found = tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir() and (path / "meta.json").is_file()),
            key=lambda path: path.name,
        )
    )
    if not found:
        raise RuntimeError(f"{root} holds no captures (a capture is a directory with meta.json)")
    return found


def _replay_recordings(path: Path) -> tuple[Path, ...]:
    """One recording per attempt from a directory, or one recording shared by every attempt."""
    if path.is_dir():
        found = tuple(sorted(path.glob("*.json")))
        if not found:
            raise RuntimeError(f"--replay {path} is a directory holding no *.json recordings")
        return found
    if not path.is_file():
        raise RuntimeError(f"--replay {path} is neither a recording nor a directory of them")
    return (path,)


def _provider_factory(
    recordings: tuple[Path, ...] | None, model: str, transport: str = "subscription"
) -> Callable[[int], Any]:
    """A fresh provider per attempt: a shared one would leak state between attempts."""

    def provider_for(index: int) -> Any:
        if recordings is None:
            if transport == "subscription":
                # Live model calls run on subscription accounts by project policy.
                from .agent.subscription import SubscriptionProvider

                return SubscriptionProvider(model)
            return AnthropicProvider(model=model)
        if len(recordings) == 1:
            return ReplayProvider.from_path(recordings[0])
        if index >= len(recordings):
            raise RuntimeError(
                f"attempt {index} has no recording: --replay supplied {len(recordings)}. "
                "Best-of-N issues N attempts per signal plus its re-run allowance, so a "
                "directory replay needs one recording per issued attempt."
            )
        return ReplayProvider.from_path(recordings[index])

    return provider_for


def _run_iterative_arm(context: _ArmContext) -> dict[str, Any]:
    """One R-round feedback loop per signal. No selection: the final receiver is the outcome."""
    from .agent.iterative import run_agent_iterative

    results = []
    for capture_dir in context.captures:
        record = run_agent_iterative(
            capture_dir,
            # Index 0: one run per signal, not N; a directory replay needs one recording per run.
            context.provider_for(0),
            config=context.config,
            run_root=context.run_root,
            sealed_roots=context.sealed_roots,
            permitted_capture_parents=(
                (capture_dir.parent,) if context.sealed_token is not None else ()
            ),
            sealed_token=context.sealed_token,
        )
        results.append(
            {
                "signal": _signal_label(capture_dir),
                "run_id": record["run_id"],
                "outcome": record["outcome"].as_dict(),
                "rounds": record["rounds"],
                "spend": record["public"]["spend"],
                "trace": record["public"]["trace"],
            }
        )
    return {"results": results, "analysis": _iterative_analysis(results)}


def _iterative_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Round decomposition and spend, aggregated over signals. Reported, never triggered on."""
    from .agent.harness import status_histogram

    total = len(results) or 1
    decomposition = {
        key: sum(1 for row in results if row["rounds"]["decomposition"][key])
        for key in (
            "succeeded_in_round_one",
            "terminated_inside_round_one",
            "executed_two_or_more_rounds",
        )
    }
    matched = sum(row["spend"]["matched_spend_usd"] for row in results)
    gross = sum(row["spend"]["gross_spend_usd"] for row in results)
    from .agent.iterative import MEASURED_CONSTRUCT

    return {
        "measured_construct": dict(MEASURED_CONSTRUCT),
        "status_histogram": status_histogram(row["outcome"] for row in results),
        "round_decomposition": decomposition,
        # TH-19 clause 2: round-one successes appear in neither half, so this is not q1 restated.
        "terminated_early_share": (
            decomposition["terminated_inside_round_one"]
            / max(
                1,
                decomposition["terminated_inside_round_one"]
                + decomposition["executed_two_or_more_rounds"],
            )
        ),
        "terminated_early_share_evaluable": (
            decomposition["terminated_inside_round_one"]
            + decomposition["executed_two_or_more_rounds"]
        )
        >= 20,
        "round_one_success_share": decomposition["succeeded_in_round_one"] / total,
        "round_one_failure_statuses": sorted(
            {
                row["rounds"]["round_one_status"]
                for row in results
                if row["rounds"]["round_one_status"]
            }
        ),
        "matched_spend_usd": matched,
        "gross_spend_usd": gross,
        "rerun_spend_usd": gross - matched,
        "round_reruns": sum(row["rounds"]["round_reruns"] for row in results),
    }


def _run_best_of_n_arm(context: _ArmContext) -> dict[str, Any]:
    """Issue N mutually blind attempts per signal, then analyse what was already paid for."""
    from .arms import best_of_n_curve, mean_selected_ber, run_best_of_n

    results = []
    for capture_dir in context.captures:

        def run_attempt(index: int, capture_dir: Path = capture_dir) -> dict[str, Any]:
            # index is all the attempt is told; it returns a run_agent record.
            return run_agent(
                capture_dir,
                context.provider_for(index),
                config=context.config,
                run_root=context.run_root,
                sealed_roots=context.sealed_roots,
                permitted_capture_parents=(
                    (capture_dir.parent,) if context.sealed_token is not None else ()
                ),
                sealed_token=context.sealed_token,
            )

        results.append(
            run_best_of_n(
                run_attempt=run_attempt,
                n=context.n,
                signal_label=_signal_label(capture_dir),
                enforce_campaign_ceiling=context.funded,
            )
        )
    return {
        "results": results,
        "analysis": {
            "best_of_n_curve": best_of_n_curve(results),
            "mean_selected_ber": mean_selected_ber(results),
        },
    }


ARM_SPECS: dict[str, _ArmSpec] = {
    "best-of-n": _ArmSpec(
        name="best-of-n",
        agent_arm="best-of-n",
        summary=(
            "N mutually blind one-shot attempts per signal; harness-side selection on "
            "crc_pass after all N complete"
        ),
        run=_run_best_of_n_arm,
    ),
    # The iterative arm carries its own system block and agent_arm label, which is what makes
    # the arm-specific digests differ while the arm-invariant one stays shared.
    "iterative": _ArmSpec(
        name="iterative",
        agent_arm="iterative",
        summary=(
            "one R-round loop per signal: author, execute, read the four-field Tier-2 result, "
            "repair. R = N at the frozen budget. Each repair round is a FRESH context carrying "
            "the code and the result but not the prior reasoning, so what this measures is "
            "single-repair, not conversational iteration -- see iterative.MEASURED_CONSTRUCT"
        ),
        run=_run_iterative_arm,
        rounds=N_ROUNDS_FROM_N,
    ),
}


def _resolve_n(args: argparse.Namespace) -> tuple[int, bool, str]:
    """(N, funded, provenance). The ledger's ceiling is bypassed only by asking for it."""
    from .arms import N_ATTEMPTS, validate_campaign_n

    if args.unfunded_n is not None:
        if args.n is not None:
            raise ValueError("--n and --unfunded-n are alternatives; pass one")
        n = int(args.unfunded_n)
        if n < 1:
            raise ValueError(f"--unfunded-n must be at least 1; got {n}")
        return n, False, "--unfunded-n: NOT campaign data, the ledger does not fund this N"
    if args.n is None:
        return (
            validate_campaign_n(N_ATTEMPTS),
            True,
            "the frozen N, derived from the ledger residual",
        )
    return (
        validate_campaign_n(int(args.n)),
        True,
        "--n, validated against the ledger-derived ceiling",
    )


def _arm_merge(args: argparse.Namespace) -> int:
    """Reassemble sharded arm records. Refuses disagreeing shards and overlapping signals."""
    from .records import write_json_once

    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.records]
    if len(shards) < 2:
        raise ValueError("arm-merge needs at least two shard records")
    for key in ("arm", "agent_arm_label", "frozen_budget_sha256", "n", "funded", "sealed"):
        values = {json.dumps(shard.get(key), sort_keys=True) for shard in shards}
        if len(values) != 1:
            raise ValueError(f"shards disagree on {key}: {sorted(values)} — not one experiment")

    merged = dict(shards[0])
    detail: list = []
    seen: set[str] = set()
    for shard in shards:
        for entry in shard.get("signals_detail") or shard.get("results") or ():
            label = str(entry.get("signal_label") or entry.get("signal"))
            if label in seen:
                raise ValueError(
                    f"signal {label!r} appears in more than one shard: two lanes ran it, and "
                    "merging would double-count a cell the pairing assumes unique"
                )
            seen.add(label)
            detail.append(entry)
    detail.sort(key=lambda entry: str(entry.get("signal_label") or entry.get("signal")))
    key_name = "signals_detail" if "signals_detail" in shards[0] else "results"
    merged[key_name] = detail
    merged["signals"] = len(detail)
    for total_key in (
        "attempts_issued", "attempts_valid", "reruns",
        "matched_spend_usd", "gross_spend_usd",
        "selected_packet_successes", "pass_at_n_upper_bound_successes",
    ):
        if all(total_key in shard for shard in shards):
            merged[total_key] = sum(shard[total_key] for shard in shards)
    merged["merged_from_shards"] = [str(path) for path in args.records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(args.out, merged, description="merged arm record")
    print(json.dumps({"signals": len(detail), "shards": len(shards), "out": str(args.out)},
                     sort_keys=True, separators=(",", ":")))
    return 0


def _gate_analysis(args: argparse.Namespace) -> int:
    """The gate, from arm records on disk to one analysis record.

    --selftest asserts the pipeline end to end on synthetic paired records.
    """
    from .gate_analysis import analyze

    if args.selftest:
        import random as _random

        rng = _random.Random(2026)
        one_recs, it_recs = [], []
        for _ in range(3):
            ones = {f"s{i:02d}": rng.random() < 0.30 for i in range(40)}
            its = {f"s{i:02d}": rng.random() < 0.65 for i in range(40)}
            one_recs.append({
                "frozen_budget_sha256": "selftest",
                "signals_detail": [
                    {"signal_label": key, "selection": {"packet_success": value}}
                    for key, value in ones.items()
                ],
            })
            it_recs.append({
                "frozen_budget_sha256": "selftest",
                "results": [
                    {
                        "signal": key,
                        "outcome": {
                            "kind": "scored_success" if value else "scored_agent_failure",
                            "packet_success": value,
                        },
                    }
                    for key, value in its.items()
                ],
            })
        result = analyze(one_recs, it_recs, draws=2_000)
        ok = result["verdict"] == "PASS" and result["delta_hat"] > 0.2
        print(json.dumps({
            "selftest": "ok" if ok else "FAILED",
            "delta_hat": result["delta_hat"],
            "bca_lower": result["interval"]["bca"]["lower"],
            "verdict": result["verdict"],
            "conclusion": result["conclusion"],
        }, sort_keys=True, separators=(",", ":")))
        return 0 if ok else 1

    if not args.one_shot or not args.iterative:
        raise ValueError("gate-analysis needs --one-shot and --iterative records, or --selftest")
    one_recs = [json.loads(p.read_text(encoding="utf-8")) for p in args.one_shot]
    it_recs = [json.loads(p.read_text(encoding="utf-8")) for p in args.iterative]
    calibration = (
        json.loads(args.coverage_calibration.read_text(encoding="utf-8"))
        if args.coverage_calibration
        else None
    )
    result = analyze(one_recs, it_recs, coverage_calibration=calibration)
    if args.out:
        # Through the shared writer so the record gets the same destination-keyed backstop.
        from .records import write_json_once

        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_once(args.out, result, description="gate analysis record")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _agent_replay(args: argparse.Namespace) -> int:
    """Re-verify one retained iterative run from its record (not from the model; no seed exists).

    Checks the trace chain, receiver hashes, sandbox re-execution per round, and the
    recomposed outcome. Sealed runs are refused: a replay would consume a counted access.
    """
    from .agent.harness import AGENT_RUNS_SUBDIR, classify_outcome, compose_rounds
    from .records import verify_trace_chain
    from .sandbox.runner import RUNS_ROOT, run_receiver

    root = Path(args.run_root) if args.run_root is not None else Path(RUNS_ROOT)
    run_dir = root.resolve() / AGENT_RUNS_SUBDIR / args.run_id
    if not run_dir.is_dir():
        raise RuntimeError(f"no retained agent run at {run_dir}")
    record = json.loads((run_dir / "agent-run.json").read_text(encoding="utf-8"))
    internal_path = run_dir / ".orchestrator" / "agent-run-internal.json"
    internal = json.loads(internal_path.read_text(encoding="utf-8"))

    if internal.get("sealed"):
        print(
            json.dumps(
                {
                    "status": "refused",
                    "run_id": args.run_id,
                    "error": (
                        "sealed replay would consume one of the two counted sealed accesses: "
                        "open_sealed counts every session, and a verification is not what "
                        "those two authorizations are for. Refused until the one-use, "
                        "write-scoped verification capability exists."
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1

    mismatches: list[str] = []
    trace_path = run_dir / record["trace"]["path"]
    chain = verify_trace_chain(trace_path, expected_head=record["trace"]["head_sha256"])
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    capture = Path(internal["capture_dir"])
    if not capture.is_dir():
        raise RuntimeError(
            f"the capture this run was made against is gone: {capture}. Captures are "
            "gitignored; rebuild the split with `materialize-split` first."
        )

    replayed: list = []
    for event in events:
        digest = event.get("receiver_sha256")
        if digest is None:
            continue
        source = run_dir / f"receiver-r{event['round']}-{digest}.py"
        if not source.is_file():
            mismatches.append(f"round {event['round']}: retained receiver {source.name} is gone")
            continue
        actual = sha256(source.read_bytes()).hexdigest()
        if actual != digest:
            mismatches.append(
                f"round {event['round']}: retained receiver hashes to {actual[:16]}…, the "
                f"trace recorded {digest[:16]}…"
            )
            continue
        outcome = run_receiver(capture, source, run_root=run_dir / "replay")
        again = classify_outcome(
            sandbox_status=str(outcome.get("status")),
            packet_success=(outcome.get("internal") or {}).get("packet_success") is True,
            sandbox_error=outcome.get("error"),
        )
        if again.as_dict() != event["outcome"]:
            mismatches.append(
                f"round {event['round']}: re-execution gives {again.as_dict()}, the trace "
                f"recorded {event['outcome']}"
            )
        if again.kind != "run_invalid":
            replayed.append(again)

    if not mismatches:
        recomposed = compose_rounds(replayed, round_cap=record["rounds"]["round_cap"])
        if recomposed.as_dict() != record["outcome"]:
            mismatches.append(
                f"the run outcome recomposes to {recomposed.as_dict()}, the record says "
                f"{record['outcome']}"
            )
        if len(replayed) != record["rounds"]["rounds_valid"]:
            mismatches.append(
                f"{len(replayed)} valid rounds on replay, the record says "
                f"{record['rounds']['rounds_valid']}"
            )

    print(
        json.dumps(
            {
                "status": "reproduced" if not mismatches else "mismatch",
                "run_id": args.run_id,
                "trace": chain,
                "rounds_checked": len(events),
                "mismatches": mismatches,
                "reproduces": (
                    "the run FROM ITS RECORD: the trace chain, the retained receivers and the "
                    "sandbox outcome of each round. NOT from the model — there is no seed and "
                    "no temperature control, and re-issuing the prompt to the live API is not "
                    "expected to reproduce anything."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if not mismatches else 1


def _arm_run(args: argparse.Namespace) -> int:
    """Run one budget-matched arm over one split and write one arm record.

    Refusal order is deliberate: sealed first, degenerate N next, then spending becomes possible.
    """
    # Imported here: importing `arms` raises when the ledger funds nothing, and that must
    # fail arm-run only.
    from .arms import (
        BESTOFN_POLICY_VERSION,
        BUDGET,
        BUDGET_POLICY_VERSION,
        SELECTOR,
        budget_summary,
        frozen_budget_hash,
        selector_preregistration,
    )

    spec = ARM_SPECS[args.arm]
    if args.split == "sealed" and not args.sealed_run_name:
        print(json.dumps({"error": SEALED_REFUSAL}, sort_keys=True, separators=(",", ":")))
        return 1
    if args.sealed_run_name and args.split != "sealed":
        raise ValueError("--sealed-run-name applies only to --split sealed")
    if args.split == "sealed" and args.captures_root is not None:
        # Sealed captures live where the token says they live, and nowhere else.
        raise ValueError(
            "--captures-root does not apply to --split sealed: a sealed split's captures are "
            "located from the open session's own sealed root"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be at least 1; got {args.limit}")

    shard: tuple[int, int] | None = None
    if getattr(args, "shard", None):
        try:
            index_text, of_text = str(args.shard).split("/", 1)
            shard = (int(index_text), int(of_text))
        except ValueError:
            raise ValueError(f"--shard must be 'k/m'; got {args.shard!r}") from None
        if not (0 <= shard[0] < shard[1]):
            raise ValueError(f"--shard index must be in [0, m); got {args.shard!r}")

    n, funded, n_provenance = _resolve_n(args)
    if n <= 1:
        print(
            json.dumps(
                {
                    "error": DEGENERATE_N_REFUSAL,
                    "n": n,
                    "n_source": n_provenance,
                    "degenerate": True,
                    "budget_degenerate_note": BUDGET.degenerate_note,
                    "budget": budget_summary(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1

    report = preflight()
    if not report.ok:
        print(json.dumps({"error": report.error, "preflight": report.as_dict()}, sort_keys=True))
        return 1

    recordings = _replay_recordings(args.replay) if args.replay is not None else None
    # R is N: the budget that buys N blind attempts buys R rounds.
    rounds = n if spec.rounds is N_ROUNDS_FROM_N else spec.rounds
    config = AgentConfig(
        model=args.model,
        arm=spec.agent_arm,
        max_rounds=rounds,
        max_executions=rounds,
        withheld_tools=("symbol_rate_candidates",),
    )
    # Funded runs must match the frozen record before spending; --unfunded-n runs are exempt.
    if funded:
        from .frozen import verify_frozen

        verify_frozen(config)
    # Not created here: run_agent makes it on the first attempt, so a refused command leaves
    # no directory behind.
    out = Path(args.out)

    if args.split == "sealed":
        split_id = args.split_id or sealed_split_spec().split_id
        with open_sealed(split_id, args.sealed_run_name) as token:
            captures_root = Path(token.sealed_root) / split_id
            produced = _drive(
                spec,
                captures_root,
                n=n,
                funded=funded,
                config=config,
                out=out,
                recordings=recordings,
                model=args.model,
                limit=args.limit,
                shard=shard,
                transport=getattr(args, "transport", "subscription"),
                sealed_token=token,
                sealed_roots=(Path(token.sealed_root),),
            )
    else:
        split_id = None
        captures_root = args.captures_root or dev_split_root()
        produced = _drive(
            spec,
            captures_root,
            n=n,
            funded=funded,
            config=config,
            out=out,
            recordings=recordings,
            model=args.model,
            limit=args.limit,
                shard=shard,
                transport=getattr(args, "transport", "subscription"),
            sealed_token=None,
            sealed_roots=(),
        )

    # Sealedness is derived from the captures root itself, not from the --split flag; the
    # flag is cross-checked against the derivation below. No token needed: the session is closed.
    sealing = sealing_of_location(captures_root, describes="the arm's captures root")
    sealed = sealing.sealed
    if (args.split == "sealed") and not sealed:
        raise RuntimeError(
            f"--split sealed resolved its captures to {captures_root}, which is not inside any "
            "sealed root: refusing to write a record that would claim a sealed campaign and "
            "redact nothing"
        )
    results = produced["results"]
    selections = [result.selection() for result in results]
    signals = len(results)
    document = {
        "arm": spec.name,
        "arm_summary": spec.summary,
        "agent_arm_label": spec.agent_arm,
        "bestofn_policy_version": BESTOFN_POLICY_VERSION,
        "budget_policy_version": BUDGET_POLICY_VERSION,
        "split": args.split,
        "split_id": split_id,
        **ARM_RUN_POLICY.identity_fields(captures_root=str(captures_root)),
        "sealed": sealed,
        "attempt_artifacts_location": "run_root",
        "run_root": str(out.resolve()),
        "n": n,
        "funded": funded,
        "n_source": n_provenance,
        "signals": signals,
        # The record states the transport that actually served the runs. "anthropic" was
        # hardcoded here and survived one live subscription smoke as a provenance lie.
        "provider": (
            "replay"
            if recordings is not None
            else ("subscription-cli" if args.transport == "subscription" else "anthropic")
        ),
        "replay": _replay_provenance(recordings, n),
        "selector": SELECTOR,
        "selector_preregistration": selector_preregistration(),
        "frozen_budget_sha256": frozen_budget_hash(config=config),
        "budget": budget_summary(),
        "attempts_issued": sum(result.attempts_issued for result in results),
        "attempts_valid": sum(result.attempts_valid for result in results),
        "reruns": sum(result.reruns for result in results),
        "matched_spend_usd": sum(result.matched_spend_usd for result in results),
        "gross_spend_usd": sum(result.gross_spend_usd for result in results),
        "selected_packet_successes": sum(1 for s in selections if s.packet_success),
        "pass_at_n_upper_bound_successes": sum(1 for s in selections if s.pass_at_n_upper_bound),
        "signals_detail": [result.as_dict() for result in results],
        **produced["analysis"],
    }
    out.mkdir(parents=True, exist_ok=True)
    record = write_record(
        out / "arm-run.json",
        document,
        ARM_RUN_POLICY,
        sealing=sealing,
        description=_ARM_ARTIFACT,
    )
    print(
        json.dumps(
            {
                "orchestrator_only": True,
                "arm": spec.name,
                "split": args.split,
                "n": n,
                "funded": funded,
                "signals": signals,
                "attempts_issued": record["attempts_issued"],
                "attempts_valid": record["attempts_valid"],
                "reruns": record["reruns"],
                "selector": SELECTOR,
                "selected_packet_successes": record["selected_packet_successes"],
                "success_rate": (record["selected_packet_successes"] / signals) if signals else None,
                "pass_at_n_upper_bound_successes": record["pass_at_n_upper_bound_successes"],
                "matched_spend_usd": record["matched_spend_usd"],
                "gross_spend_usd": record["gross_spend_usd"],
                "record": str(out / "arm-run.json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    # Zero on a completed sweep, not a successful one: the arm's result is the record.
    return 0


def _replay_provenance(recordings: tuple[Path, ...] | None, n: int) -> dict[str, Any]:
    """Replay provenance: one shared recording means zero spread by construction, so say which."""
    if recordings is None:
        return {"replayed": False}
    shared = len(recordings) == 1
    return {
        "replayed": True,
        "no_api_calls": True,
        "recordings": [path.name for path in recordings],
        "shared_recording": shared,
        "note": (
            "one recording replayed for every attempt: all N attempts serve identical "
            "recorded responses, so this run proves the path and measures nothing about "
            "attempt-to-attempt spread"
            if shared
            else f"one recording per attempt index, for up to {len(recordings)} issued attempts"
        ),
        "attempts_per_signal_requested": n,
    }


def _drive(
    spec: _ArmSpec,
    captures_root: Path,
    *,
    n: int,
    funded: bool,
    config: AgentConfig,
    out: Path,
    recordings: tuple[Path, ...] | None,
    model: str,
    limit: int | None,
    sealed_token: Any,
    sealed_roots: tuple[Path, ...],
    shard: tuple[int, int] | None = None,
    transport: str = "subscription",
) -> dict[str, Any]:
    captures = _split_captures(Path(captures_root))
    if limit is not None:
        captures = captures[:limit]
    if shard is not None:
        index, of = shard
        # Stride sharding over the sorted list: shard k of m takes captures[k::m], disjoint
        # by construction; arm-merge refuses overlaps.
        captures = captures[index::of]
    return spec.run(
        _ArmContext(
            spec=spec,
            captures=captures,
            n=n,
            funded=funded,
            config=config,
            run_root=out,
            provider_for=_provider_factory(recordings, model, transport),
            sealed_token=sealed_token,
            sealed_roots=sealed_roots,
        )
    )


def _characterize_tools(args: argparse.Namespace) -> int:
    report = default_report(args.captures_root)
    json_path, markdown_path = write_characterization(report, args.out)
    print(
        json.dumps(
            {
                "capture_count": report["capture_count"],
                "tools_sha256": report["tools_sha256"],
                "symbol_rate_top1_rate": report["symbol_rate_ranking"]["top1_rate"],
                "ast_rejected_rate": report["ast_gate"]["ast_rejected_rate"],
                "paths": [str(json_path), str(markdown_path)],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modembench")
    commands = parser.add_subparsers(dest="command", required=True)
    spine = commands.add_parser("spine", help="generate, oracle-decode, and evaluate one capture")
    spine.add_argument("--seed", type=int, required=True)
    spine.add_argument("--outdir", type=Path, default=Path("."))
    spine.add_argument("--profile", choices=("clean", "impaired"), default="clean")
    gate = commands.add_parser("gate", help="run the immutable hr-30 reference-receiver gate")
    gate.add_argument("--n", type=int, required=True)
    gate.add_argument("--snr-db", type=float, required=True)
    gate.add_argument("--seed-base", type=int, required=True)
    gate.add_argument("--profile", choices=("clean", "impaired"), required=True)
    gate.add_argument("--out", type=Path, required=True)
    oracle = commands.add_parser(
        "make-oracle-receiver", help="materialize a protected content-addressed oracle receiver"
    )
    oracle.add_argument("capture_dir", type=Path)
    sandbox_run = commands.add_parser("sandbox-run", help="run and evaluate a sandboxed receiver")
    sandbox_run.add_argument("capture_dir", type=Path)
    sandbox_run.add_argument("receiver", type=Path)
    sandbox_run.add_argument("--timeout", type=float, default=10.0)
    sandbox_run.add_argument("--mem-mib", type=float, default=512.0)
    replay = commands.add_parser("sandbox-replay", help="replay a retained sandbox run")
    replay.add_argument("run_id")
    verify = commands.add_parser(
        "verify-split", help="recompute a split and compare it with its published commitment"
    )
    verify.add_argument("split_id")
    verify.add_argument("--captures-root", type=Path, default=None)
    verify.add_argument("--commitments-dir", type=Path, default=None)
    verify.add_argument(
        "--sealed-root",
        type=Path,
        default=None,
        help="locate a sealed split's captures under this root (no salt is read)",
    )
    materialize = commands.add_parser(
        "materialize-split",
        help="rebuild a published split's captures from its committed specification",
    )
    materialize.add_argument("split_id")
    materialize.add_argument("--captures-root", type=Path, default=None)
    materialize.add_argument("--commitments-dir", type=Path, default=None)
    gate_cmd = commands.add_parser(
        "gate-analysis",
        help="the comparison gate: paired delta, BCa interval, McNemar, verdict, routed conclusion",
    )
    gate_cmd.add_argument("--one-shot", type=Path, nargs="*", default=None,
                          help="one best-of-n arm-run.json per replicate, in replicate order")
    gate_cmd.add_argument("--iterative", type=Path, nargs="*", default=None,
                          help="one iterative arm-run.json per replicate, in replicate order")
    gate_cmd.add_argument("--coverage-calibration", type=Path, default=None,
                          help="the TH-22 calibration record; required for a sealed analysis")
    gate_cmd.add_argument("--out", type=Path, default=None,
                          help="write the analysis record here")
    gate_cmd.add_argument("--selftest", action="store_true",
                          help="run the module against synthetic paired records and exit")
    agent_replay = commands.add_parser(
        "agent-replay",
        help="re-verify a retained iterative agent run from its run id",
    )
    agent_replay.add_argument("--run-id", required=True)
    agent_replay.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="where runs/ lives, if not the default",
    )
    merge = commands.add_parser(
        "arm-merge",
        help="merge sharded arm-run records (parallel lanes) into one arm record",
    )
    merge.add_argument("--out", required=True, type=Path)
    merge.add_argument("records", nargs="+", type=Path)
    agent = commands.add_parser(
        "agent-run", help="run one no-evaluator-feedback agent attempt against a capture"
    )
    agent.add_argument("--capture", type=Path, required=True)
    agent.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="serve recorded responses instead of calling the API (no network)",
    )
    # No --effort or --thinking: both are frozen in agent/provider.py.
    agent.add_argument("--model", default=HEADLINE_MODEL)
    agent.add_argument("--run-root", type=Path, default=Path("runs"))
    arm = commands.add_parser(
        "arm-run", help="run one budget-matched arm over a split and write one arm record"
    )
    arm.add_argument(
        "--arm",
        choices=sorted(ARM_SPECS),
        required=True,
        help="; ".join(f"{name}: {spec.summary}" for name, spec in sorted(ARM_SPECS.items())),
    )
    arm.add_argument(
        "--split",
        choices=("dev", "sealed"),
        default="dev",
        help="sealed is refused unless --sealed-run-name engages the counted sealed session",
    )
    arm.add_argument(
        "--replay",
        type=Path,
        default=None,
        help=(
            "a recording, or a directory of recordings served one per attempt index, instead "
            "of calling the API (no network)"
        ),
    )
    arm.add_argument(
        "--n",
        type=int,
        default=None,
        help="campaign N; defaults to the frozen ledger-derived N and is validated against its ceiling",
    )
    arm.add_argument(
        "--unfunded-n",
        type=int,
        default=None,
        help=(
            "run the mechanism at an N the ledger does not fund; the record is marked "
            "funded=false and is not campaign data"
        ),
    )
    arm.add_argument("--limit", type=int, default=None, help="run only the first K signals")
    arm.add_argument(
        "--shard",
        default=None,
        help="'k/m': this process runs captures[k::m] of the sorted split — the parallel "
        "lanes of the campaign. Shards are disjoint by construction; reassemble with "
        "arm-merge, which refuses overlaps.",
    )
    arm.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "arms",
        help=(
            "the only location this command writes: the arm record and every attempt's run "
            "records. Name a fresh directory per arm run; the record is written once and a "
            "second run into the same directory is refused rather than overwritten."
        ),
    )
    arm.add_argument("--captures-root", type=Path, default=None)
    arm.add_argument(
        "--sealed-run-name",
        default=None,
        help="spend one counted sealed access under this authorized run name (--split sealed)",
    )
    arm.add_argument("--split-id", default=None, help="sealed split id; defaults to the published one")
    arm.add_argument("--model", default=HEADLINE_MODEL)
    arm.add_argument(
        "--transport",
        choices=("subscription", "api"),
        default="subscription",
        help="how live model calls travel. 'subscription' (default) serves the full harness "
        "over `claude -p` (the project default); 'api' uses AnthropicProvider and "
        "requires a key. Replay runs ignore this.",
    )
    characterize = commands.add_parser(
        "characterize-tools",
        help="measure every diagnostic tool against manifest truth and write the error table",
    )
    characterize.add_argument("--captures-root", type=Path, default=None)
    characterize.add_argument("--out", type=Path, default=Path("docs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "spine":
            return _spine(args.seed, args.outdir, args.profile)
        if args.command == "gate":
            command_line = shlex.join(
                [sys.executable, "-m", "modembench", *(argv if argv is not None else sys.argv[1:])]
            )
            code, report, paths = run_gate(
                n=args.n,
                snr_db=args.snr_db,
                seed_base=args.seed_base,
                profile=args.profile,
                outdir=args.out,
                command_line=command_line,
            )
            print(
                json.dumps(
                    {
                        "attempt": report["attempt"],
                        "gate_passed": report["gate_passed"],
                        "n_valid": report["n_valid"],
                        "n_success": report["n_success"],
                        "observed_sample_rate": report["observed_sample_rate"],
                        "wilson_95_lower_bound": report["wilson_95_lower_bound"],
                        "reports": [str(path) for path in paths],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return code
        if args.command == "make-oracle-receiver":
            path = materialize_oracle_source(args.capture_dir)
            print(
                json.dumps(
                    {"receiver_path": str(path), "receiver_sha256": path.stem.removeprefix("receiver-")},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "sandbox-run":
            result = run_receiver(
                args.capture_dir,
                args.receiver,
                timeout_s=args.timeout,
                memory_mib=args.mem_mib,
            )
            output = {
                "orchestrator_only": True,
                "run_id": result["run_id"],
                "sandbox": {
                    key: value
                    for key, value in result.items()
                    if key not in {"feedback", "internal"}
                },
                "feedback": result["feedback"],
                "internal": result["internal"],
            }
            print(json.dumps(output, sort_keys=True, separators=(",", ":")))
            return int(
                result["status"] != "ok" or result["internal"].get("packet_success") is not True
            )
        if args.command == "sandbox-replay":
            result = replay_run(args.run_id)
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0 if result["status"] == "reproduced" else 1
        if args.command == "gate-analysis":
            return _gate_analysis(args)
        if args.command == "arm-merge":
            return _arm_merge(args)
        if args.command == "agent-replay":
            return _agent_replay(args)
        if args.command == "verify-split":
            return _verify_split(args)
        if args.command == "materialize-split":
            return _materialize_split(args)
        if args.command == "agent-run":
            return _agent_run(args)
        if args.command == "arm-run":
            return _arm_run(args)
        if args.command == "characterize-tools":
            return _characterize_tools(args)
    except (CaptureConflictError, OSError, ValueError, RuntimeError, KeyError, ProviderError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True, separators=(",", ":")))
        return 1
    return 2
