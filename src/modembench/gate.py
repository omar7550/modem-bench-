"""Hr-30 randomized gate runner with immutable reports and strict validity."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tomllib
from typing import Any
import uuid

import numpy as np
import scipy

from .diagnostics import diagnose, flattened_metrics
from .evaluate import encode_truth
from .framing import build_frame
from .generator import canonical_json, generate_capture
from .impairments import ImpairmentConfig, ImpairmentControl, ImpairmentRanges
from .reference_rx import decode
from .sealed import (
    MANIFEST_ARTIFACT,
    PAYLOAD_ARTIFACT,
    read_private_artifact,
)


GATE_THRESHOLD = 0.95
MIN_GATE_DRAWS = 100
WILSON_Z_95 = 1.959963984540054
LOGGER = logging.getLogger(__name__)
ATTEMPT_JSON_PATTERN = re.compile(r"attempt-(\d+)\.json")
ATTEMPT_ORPHAN_PATTERN = re.compile(r"attempt-(\d+)(?:\.md|-runs.*)")


def wilson_lower_bound(successes: int, total: int) -> float:
    """Return the two-sided 95% Wilson score interval's lower endpoint."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    z2 = WILSON_Z_95**2
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    margin = WILSON_Z_95 * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total**2)
    )
    return max(0.0, (center - margin) / denominator)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_tree_provenance(root: Path | None = None) -> dict[str, Any]:
    """Hash locked source inputs as SHA-256(sorted relative path + per-file hash)."""
    project = (root or _project_root()).resolve()
    source_root = project / "src/modembench"
    candidates = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    candidates.extend((project / "pyproject.toml", project / "data/seed_registry.json"))
    if any(not path.is_file() for path in candidates):
        raise RuntimeError("source-tree provenance input is missing")
    # Policy and commitments bind too (presence-optional: deleting them cannot widen
    # anything). The access log and its anchor are excluded: they change on every open,
    # so including them would make routine reads look like policy edits.
    candidates.extend(sorted(project.glob("data/sealed_access.json")))
    candidates.extend(sorted(project.glob("data/commitments/*.json")))
    per_file = {
        path.relative_to(project).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(candidates)
    }
    digest_input = b"".join(
        f"{relative}\0{digest}\n".encode("utf-8") for relative, digest in sorted(per_file.items())
    )
    return {"source_tree_sha256": sha256(digest_input).hexdigest(), "files": per_file}


def _git_provenance(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, check=False, capture_output=True, text=True
    )
    if head.returncode != 0 or status.returncode != 0:
        raise RuntimeError("git provenance query failed")
    return {"head_sha": head.stdout.strip(), "dirty": bool(status.stdout)}


def _environment(root: Path) -> dict[str, Any]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "declared_pins": project["project"]["dependencies"],
    }


def _load_registry(root: Path) -> dict[str, Any]:
    registry = json.loads((root / "data/seed_registry.json").read_text(encoding="utf-8"))
    if registry.get("schema_version") != "1.0" or not isinstance(registry.get("reservations"), list):
        raise RuntimeError("seed registry is malformed")
    return registry


def _validate_seed_block(registry: dict[str, Any], seed_base: int, n: int) -> None:
    match = next(
        (
            entry
            for entry in registry["reservations"]
            if entry.get("name") == "gate-diagnostics" and entry.get("sealed") is False
        ),
        None,
    )
    if match is None:
        raise RuntimeError("non-sealed gate-diagnostics reservation is missing")
    if seed_base < int(match["start"]) or seed_base + n - 1 > int(match["stop"]):
        raise ValueError("requested gate seeds fall outside the gate-diagnostics reservation")
    # sealed reservations publish no start/stop; check the blocks that declare one
    blocks = [entry for entry in registry["reservations"] if "start" in entry and "stop" in entry]
    ordered = sorted(blocks, key=lambda item: int(item["start"]))
    if any(int(left["stop"]) >= int(right["start"]) for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError("seed registry reservations overlap")


def _next_attempt(outdir: Path) -> tuple[int, str | None, str]:
    published_attempts = {
        int(match.group(1))
        for path in outdir.iterdir()
        if (match := ATTEMPT_JSON_PATTERN.fullmatch(path.name)) is not None
    }
    for path in sorted(outdir.iterdir()):
        match = ATTEMPT_ORPHAN_PATTERN.fullmatch(path.name)
        if match is not None and int(match.group(1)) not in published_attempts:
            LOGGER.warning(
                "ignoring orphan gate artifact without a published JSON report: %s", path
            )
    previous_number = max(published_attempts, default=0)
    attempt = previous_number + 1
    previous = f"attempt-{previous_number:03d}" if previous_number else None
    rationale = (
        "Initial execution of the locked feasibility gate."
        if previous is None
        else f"Rerun after {previous}; the previous attempt remains immutable."
    )
    return attempt, previous, rationale


def _create_attempt_runs_dir(outdir: Path, attempt: int) -> Path:
    stem = f"attempt-{attempt:03d}"
    published_json = outdir / f"{stem}.json"
    runs_dir = outdir / f"{stem}-runs"
    if runs_dir.exists():
        if published_json.exists():
            raise FileExistsError(
                f"published attempt run directory is immutable: {runs_dir}"
            )
        orphan_dir = outdir / f"{stem}-runs-orphan-{uuid.uuid4().hex}"
        LOGGER.warning(
            "moving aside orphan gate run directory without a published JSON report: %s -> %s",
            runs_dir,
            orphan_dir,
        )
        runs_dir.rename(orphan_dir)
    runs_dir.mkdir(parents=False, exist_ok=False)
    return runs_dir


def _decode_from_manifest(iq: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    waveform = manifest["waveform"]
    framing = manifest["framing"]
    impairments = manifest["impairments"]
    return decode(
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


def _validate_evaluator_result(result: Any, returncode: int) -> None:
    if not isinstance(result, dict) or set(result) != {"feedback", "internal"}:
        raise RuntimeError("evaluator returned malformed JSON")
    feedback = result["feedback"]
    internal = result["internal"]
    if not isinstance(feedback, dict) or set(feedback) != {
        "acquisition_success",
        "crc_pass",
        "aligned_ber",
        "error",
    }:
        raise RuntimeError("evaluator feedback schema is malformed")
    if not isinstance(internal, dict) or set(internal) != {
        "packet_success",
        "n_payload_bits",
        "alignment_offset",
        "sync_hamming",
    }:
        raise RuntimeError("evaluator internal schema is malformed")
    if type(feedback["acquisition_success"]) is not bool or type(feedback["crc_pass"]) is not bool:
        raise RuntimeError("evaluator feedback boolean fields are malformed")
    aligned_ber = feedback["aligned_ber"]
    if aligned_ber is not None and (
        isinstance(aligned_ber, bool)
        or not isinstance(aligned_ber, (int, float))
        or not math.isfinite(float(aligned_ber))
        or not 0.0 <= float(aligned_ber) <= 1.0
    ):
        raise RuntimeError("evaluator aligned_ber is malformed")
    if feedback["error"] is not None and not isinstance(feedback["error"], str):
        raise RuntimeError("evaluator error field is malformed")
    if type(internal["packet_success"]) is not bool:
        raise RuntimeError("evaluator packet_success is malformed")
    if type(internal["n_payload_bits"]) is not int:
        raise RuntimeError("evaluator n_payload_bits is malformed")
    for field in ("alignment_offset", "sync_hamming"):
        if internal[field] is not None and type(internal[field]) is not int:
            raise RuntimeError(f"evaluator {field} is malformed")
    expected_code = 0 if internal["packet_success"] is True else 1
    if returncode != expected_code:
        raise RuntimeError("evaluator exit code is inconsistent with evaluator JSON")


def _attempt_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# ModemBench feasibility gate — attempt {report['attempt']:03d}",
        "",
        report["rationale"],
        "",
        f"- Valid runs: {report['n_valid']} / {report['n_requested']}",
        f"- Packet successes: {report['n_success']}",
        f"- Observed sample rate: {report['observed_sample_rate']:.6f}",
        f"- Gate threshold: {report['gate_threshold']:.6f}",
        f"- Wilson 95% lower bound (reported, not gated): {report['wilson_95_lower_bound']:.6f}",
        f"- Gate passed: {str(report['gate_passed']).lower()}",
        "",
        "Failure attribution is heuristic first-crossing only; it is not causal.",
        "",
    ]
    if report["failures"]:
        lines.extend(("| Seed | Attributed stage | Unattributed |", "|---:|---|:---:|"))
        for failure in report["failures"]:
            lines.append(
                f"| {failure['seed']} | {failure['attributed_stage'] or ''} | "
                f"{str(failure['unattributed']).lower()} |"
            )
        lines.append("")
    if report["invalid_runs"]:
        lines.extend(("## Invalid runs", ""))
        lines.extend(
            f"- Seed {invalid['seed']}: {invalid['error_type']}: {invalid['error']}"
            for invalid in report["invalid_runs"]
        )
        lines.append("")
    lines.extend(("## Provenance", "", f"- Command: `{report['command_line']}`"))
    lines.append(f"- HEAD: `{report['provenance']['git']['head_sha']}`")
    lines.append(f"- Dirty: {str(report['provenance']['git']['dirty']).lower()}")
    lines.append(
        f"- Source-tree SHA-256: `{report['provenance']['source_tree']['source_tree_sha256']}`"
    )
    lines.append("")
    return "\n".join(lines)


def _write_and_fsync(path: Path, payload: bytes, *, exclusive: bool) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_attempt(report: dict[str, Any], outdir: Path) -> tuple[Path, Path]:
    stem = f"attempt-{report['attempt']:03d}"
    json_path = outdir / f"{stem}.json"
    md_path = outdir / f"{stem}.md"
    if json_path.exists():
        raise FileExistsError(f"published attempt already exists: {json_path}")
    json_payload = canonical_json(report) + b"\n"
    markdown_payload = _attempt_markdown(report).encode("utf-8")
    _write_and_fsync(md_path, markdown_payload, exclusive=False)
    if not md_path.read_text(encoding="utf-8").startswith("# ModemBench feasibility gate"):
        raise RuntimeError("attempt Markdown verification failed")
    staging_path = outdir / f".{stem}.{uuid.uuid4().hex}.json.staging"
    try:
        _write_and_fsync(staging_path, json_payload, exclusive=True)
        if json.loads(staging_path.read_text(encoding="utf-8"))["attempt"] != report["attempt"]:
            raise RuntimeError("attempt JSON verification failed")
        if json_path.exists():
            raise FileExistsError(f"published attempt already exists: {json_path}")
        os.replace(staging_path, json_path)
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise
    return json_path, md_path


def run_gate(
    *,
    n: int,
    snr_db: float,
    seed_base: int,
    profile: str,
    outdir: Path,
    command_line: str | None = None,
) -> tuple[int, dict[str, Any], tuple[Path, Path]]:
    """Execute the gate and return ``(exit_code, report, report_paths)``."""
    if isinstance(n, bool) or n <= 0 or isinstance(seed_base, bool) or seed_base < 0:
        raise ValueError("n must be positive and seed-base must be non-negative")
    if not np.isfinite(snr_db):
        raise ValueError("snr-db must be finite")
    root = _project_root()
    registry = _load_registry(root)
    _validate_seed_block(registry, seed_base, n)
    before_source = source_tree_provenance(root)
    outdir.mkdir(parents=True, exist_ok=True)
    attempt, previous_attempt, rationale = _next_attempt(outdir)
    runs_dir = _create_attempt_runs_dir(outdir, attempt)
    captures_dir = outdir / "captures"

    config = ImpairmentConfig.from_profile(profile)
    if profile == "impaired":
        config = replace(config, awgn=ImpairmentControl(True, float(snr_db)))
    ranges = ImpairmentRanges()
    valid_runs: list[dict[str, Any]] = []
    invalid_runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for seed in range(seed_base, seed_base + n):
        try:
            capture = generate_capture(seed, captures_dir, config=config, ranges=ranges)
            # read once through the chokepoint and hand truth to the evaluator on stdin
            manifest_bytes = read_private_artifact(capture.capture_dir, MANIFEST_ARTIFACT)
            payload = read_private_artifact(capture.capture_dir, PAYLOAD_ARTIFACT)
            manifest = json.loads(manifest_bytes)
            iq = np.load(capture.capture_dir / "iq.npy", allow_pickle=False)
            bits = _decode_from_manifest(iq, manifest)
            run_dir = runs_dir / f"seed-{seed}"
            run_dir.mkdir(parents=False, exist_ok=False)
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
                raise RuntimeError("evaluator returned malformed JSON") from exc
            _validate_evaluator_result(evaluated, completed.returncode)

            framing = manifest["framing"]
            waveform = manifest["waveform"]
            impairments = manifest["impairments"]
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
            diagnostics_path = run_dir / "diagnostics.json"
            diagnostics_path.write_bytes(canonical_json(diagnostics) + b"\n")
            if not diagnostics_path.is_file():
                raise RuntimeError("diagnostics attribution artifact is missing")
            reloaded_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if reloaded_diagnostics.get("internal") is not True:
                raise RuntimeError("diagnostics attribution artifact is malformed")

            draws = {
                name: {
                    key: impairments[name][key]
                    for key in ("enabled", "drawn_value", "applied_value", "overridden")
                }
                for name in ("fractional_timing", "cfo", "phase", "amplitude", "awgn")
            }
            success = evaluated["internal"]["packet_success"]
            metric_values = flattened_metrics(diagnostics)
            row = {
                "seed": seed,
                "capture_id": capture.capture_id,
                "packet_success": success,
                "draws": draws,
                "diagnostics_artifact": diagnostics_path.relative_to(outdir).as_posix(),
                "attributed_stage": diagnostics["attributed_stage"],
                "unattributed": diagnostics["unattributed"],
                "metric_values": metric_values,
            }
            failure_row = None
            if success is False:
                failure_row = {
                    **row,
                    "attribution_method": diagnostics["attribution_method"],
                }
            valid_runs.append(row)
            if failure_row is not None:
                failures.append(failure_row)
        except Exception as exc:
            invalid_runs.append(
                {"seed": seed, "error_type": type(exc).__name__, "error": str(exc)}
            )

    after_source = source_tree_provenance(root)
    if after_source["source_tree_sha256"] != before_source["source_tree_sha256"]:
        invalid_runs.append(
            {
                "seed": None,
                "error_type": "ProvenanceHashError",
                "error": "source tree changed while the gate was running",
            }
        )
    n_valid = len(valid_runs)
    n_success = sum(int(row["packet_success"]) for row in valid_runs)
    observed_rate = n_success / n_valid if n_valid else 0.0
    gate_passed = bool(
        not invalid_runs
        and n_valid == n
        and n_valid >= MIN_GATE_DRAWS
        and observed_rate >= GATE_THRESHOLD
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "attempt": attempt,
        "previous_attempt": previous_attempt,
        "rationale": rationale,
        "command_line": command_line
        or shlex.join([sys.executable, "-m", "modembench", *sys.argv[1:]]),
        "profile": profile,
        "snr_db": float(snr_db),
        "seed_base": seed_base,
        "n_requested": n,
        "n_valid": n_valid,
        "n_success": n_success,
        "observed_sample_rate": observed_rate,
        "gate_threshold": GATE_THRESHOLD,
        "wilson_95_lower_bound": wilson_lower_bound(n_success, n_valid),
        "wilson_is_gate_criterion": False,
        "gate_passed": gate_passed,
        "gate_rule": "exit 0 iff n_valid=n>=100 and observed sample rate>=0.95",
        "attribution_semantics": "heuristic first-threshold crossing only; not causal",
        "config": config.to_dict(),
        "ranges": ranges.to_dict(),
        "ranges_hash": ranges.ranges_hash,
        "config_hash": config.config_hash,
        "seed_registry": registry,
        "runs": valid_runs,
        "failures": failures,
        "invalid_runs": invalid_runs,
        "provenance": {
            "git": _git_provenance(root),
            "source_tree": after_source,
            "environment": _environment(root),
        },
    }
    paths = _write_attempt(report, outdir)
    return (0 if gate_passed else 1), report, paths
