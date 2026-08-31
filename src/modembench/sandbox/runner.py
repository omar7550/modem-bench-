"""Sandboxed receiver execution, retained artifacts, and replay."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import scipy

from ..evaluate import encode_truth
from ..evaluator import load_receiver_output

# Some of these are re-exports: agent/feedback.py and the sealed-sandbox tests still
# resolve them through this module. Do not delete as unused.
from ..records import (
    ALWAYS,
    SEALED,
    SEALED_BER_GRID_DENOMINATOR,
    SEALED_EVALUATOR_REDACTIONS,
    SEALED_SANDBOX_REDACTIONS,
    Drop,
    Null,
    RecordPolicy,
    Redact,
    Substitute,
    quantize_ber_to_grid,
    refuse_sealed_identity,
    sealed_safe_evaluator,
    sealed_safe_sandbox,
    sealing_of_capture,
    write_record,
)
from ..sealed import (
    IQ_ARTIFACT,
    MANIFEST_ARTIFACT,
    META_ARTIFACT,
    PAYLOAD_ARTIFACT,
    SealedAccessError,
    authorize_read,
    capture_reference,
    private_artifact_path,
    private_dir,
    read_private_artifact,
    read_sealed_run_record,
    sealed_root_containing,
    sealed_run_artifact_dir,
    sealed_run_record_path,
    write_sealed_run_record,
)
from .ast_gate import AST_POLICY_VERSION, check_source
from .oracle_source import oracle_artifact_name
from .profile import (
    POLICY_VERSION,
    REPO_ROOT,
    SANDBOX_EXEC,
    SandboxPolicy,
    build_policy,
    ensure_sandbox,
    profile_hashes,
    render_profile,
)
from .shim_template import SHIM_VERSION, make_shim_source, shim_sha256

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEMORY_MIB = 512.0
MAX_OUTPUT_BITS = 1 << 20
# per-file limit; headroom covers np.save's header, since CPython ignores SIGXFSZ and
# an exact cap would truncate a maximum-size output silently
DEFAULT_FSIZE_BYTES = MAX_OUTPUT_BITS + (1 << 16)
TAIL_BYTES = 4096
POLL_INTERVAL_S = 0.25
TERMINATE_GRACE_S = 0.75
RUNS_ROOT = REPO_ROOT / "runs"
RUN_ID_HASH_CHARS = 16
# bump when the set of hashed scoring files changes
EVALUATOR_HASH_VERSION = "evaluator-hash-v2"
RESULT_STATUSES = {
    "ok",
    "ast_rejected",
    "timeout",
    "memory_exceeded",
    "cpu_exceeded",
    "fsize_exceeded",
    "resource_monitor_unavailable",
    "crashed",
    "bad_output",
    "sandbox_unavailable",
}


def _hash_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_cap(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return converted


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _capture_inputs(capture_dir: Path, token: Any = None) -> dict[str, Any]:
    # deliberately returns no private path; a private location inside this dict would
    # taint every caller for the chokepoint scan
    iq_path = capture_dir / IQ_ARTIFACT
    meta_path = capture_dir / META_ARTIFACT
    for path in (
        iq_path,
        meta_path,
        private_artifact_path(capture_dir, MANIFEST_ARTIFACT),
        private_artifact_path(capture_dir, PAYLOAD_ARTIFACT),
    ):
        if not path.is_file():
            raise ValueError(f"capture artifact is missing: {path}")
    meta = _read_json(meta_path)
    manifest_bytes = read_private_artifact(capture_dir, MANIFEST_ARTIFACT, token)
    payload_bytes = read_private_artifact(capture_dir, PAYLOAD_ARTIFACT, token)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("capture manifest is not a JSON object")
    sample_rate = meta.get("sample_rate_hz")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise ValueError("public sample_rate_hz is invalid")
    sample_rate = float(sample_rate)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("public sample_rate_hz is invalid")
    capture_id = meta.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id or manifest.get("capture_id") != capture_id:
        raise ValueError("capture identifiers are invalid or inconsistent")
    iq_hash = _hash_file(iq_path)
    payload_hash = _hash_bytes(payload_bytes)
    expected = manifest.get("hashes")
    if not isinstance(expected, dict) or iq_hash != expected.get(IQ_ARTIFACT):
        raise ValueError("actual IQ hash does not match the capture manifest")
    if payload_hash != expected.get(PAYLOAD_ARTIFACT):
        raise ValueError("actual payload hash does not match the capture manifest")
    return {
        "capture_id": capture_id,
        "iq_path": iq_path,
        "meta_path": meta_path,
        "sample_rate": sample_rate,
        "iq_sha256": iq_hash,
        "meta_sha256": _hash_file(meta_path),
        "manifest_sha256": _hash_bytes(manifest_bytes),
        "payload_sha256": payload_hash,
    }


def _new_run_dir(
    root: Path, receiver_hash: str, capture_id: str, *, sealed: bool
) -> tuple[str, Path]:
    """Mint one run directory, refusing a name that is itself sealed identity.

    Sealed name is <ts>-<nonce>-<capture ref>: capture id and receiver digest are sealed
    identity and must not land in a runs/ listing.
    """
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    nonce = secrets.token_hex(6)
    safe_capture_id = "".join(c for c in capture_id if c.isalnum() or c in "-_")
    if not safe_capture_id or safe_capture_id != capture_id:
        raise ValueError("capture_id contains unsafe path characters")
    run_id = (
        f"{timestamp}-{nonce}-{safe_capture_id}"
        if sealed
        else f"{timestamp}-{nonce}-{receiver_hash[:RUN_ID_HASH_CHARS]}-{safe_capture_id}"
    )
    if sealed and receiver_hash[:RUN_ID_HASH_CHARS].casefold() in run_id.casefold():
        raise SealedAccessError(
            "a sealed run directory would spell the receiver digest, which is sealed "
            "identity whenever the receiver is the capture's own protected oracle"
        )
    run_dir = root / run_id
    refuse_sealed_identity(
        {},
        path=run_dir,
        because="a run directory is minted in the repository before any record is written",
    )
    os.mkdir(run_dir)
    return run_id, run_dir


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _group_rss_mib(pgid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-g", str(pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    except ValueError:
        return None
    if not values:
        return None
    return sum(values) / 1024.0


def _terminate_group(process: subprocess.Popen[bytes], pgid: int) -> bool:
    """Tear down a process group; returns True iff the group is gone afterwards."""
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # macOS killpg on an exited-but-unreaped group can report EPERM; both mean
            # nothing left to signal
            pass
    deadline = time.monotonic() + TERMINATE_GRACE_S
    while _group_alive(pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.025)
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # macOS killpg on an exited-but-unreaped group can report EPERM; both mean
            # nothing left to signal
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # escaping here would abort run_receiver before any artifact is written
            pass
    return not _group_alive(pgid)


def _supervise(
    process: subprocess.Popen[bytes], *, timeout_s: float, memory_mib: float
) -> tuple[str | None, float]:
    """Supervise one process group; an observed breach is permanently latched."""
    pgid = process.pid
    started = time.monotonic()
    latched: str | None = None
    max_rss = 0.0
    while True:
        returncode = process.poll()
        alive = _group_alive(pgid)
        if alive:
            rss = _group_rss_mib(pgid)
            if rss is None:
                if _group_alive(pgid):
                    latched = latched or "resource_monitor_unavailable"
            else:
                max_rss = max(max_rss, rss)
                if rss > memory_mib:
                    latched = latched or "memory_exceeded"
        # a clean exit observed on this tick wins over the deadline; only a group still
        # running at the deadline is a real timeout
        if returncode is not None and not alive:
            if latched is not None:
                if not _terminate_group(process, pgid):
                    return "resource_monitor_unavailable", max_rss
                return latched, max_rss
            return None, max_rss
        if time.monotonic() - started >= timeout_s:
            latched = latched or "timeout"
        if latched is not None:
            if not _terminate_group(process, pgid):
                # a group we failed to stop is a monitor failure, not a clean breach
                return "resource_monitor_unavailable", max_rss
            return latched, max_rss
        time.sleep(POLL_INTERVAL_S)


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - TAIL_BYTES))
            return stream.read(TAIL_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _diagnostic_status(path: Path) -> dict[str, Any] | None:
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value


def _classify_exit(returncode: int | None) -> tuple[str, str | None, int | None]:
    if returncode is None:
        return "crashed", "process_not_reaped", None
    if returncode < 0:
        signum = -returncode
        if signum == signal.SIGXCPU:
            return "cpu_exceeded", "cpu_limit", signum
        if signum == signal.SIGXFSZ:
            return "fsize_exceeded", "file_size_limit", signum
        return "crashed", f"signal_{signum}", signum
    mapping = {
        10: ("crashed", "missing_receive"),
        11: ("crashed", "receiver_raised"),
        12: ("bad_output", "bad_output"),
        13: ("crashed", "iq_load_failed"),
        14: ("crashed", "shim_internal_error"),
    }
    status, error = mapping.get(returncode, ("crashed", f"unexpected_exit_{returncode}"))
    return status, error, None


def _retain_file(source: Path, destination: Path) -> None:
    # the sealed store may be on another filesystem; bare os.replace raises EXDEV
    try:
        os.replace(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
        os.unlink(source)


def _base_result(*, receiver_hash: str, iq_hash: str) -> dict[str, Any]:
    return {
        "status": "crashed",
        "receiver_sha256": receiver_hash,
        "iq_sha256": iq_hash,
        "stdout_tail": "",
        "stderr_tail": "",
        "wall_time_s": 0.0,
        "max_rss_mib": 0.0,
        "exit_code": None,
        "exit_signal": None,
        "error": None,
    }


def _execute(
    *,
    policy: SandboxPolicy,
    profile: str,
    shim_source: str,
    source: bytes,
    iq_path: Path,
    run_dir: Path,
    bits_dir: Path,
    receiver_hash: str,
    iq_hash: str,
    timeout_s: float,
    memory_mib: float,
    preflight: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = _base_result(receiver_hash=receiver_hash, iq_hash=iq_hash)
    scratch = policy.scratch_dir
    stdout_path = scratch / "stdout.txt"
    stderr_path = scratch / "stderr.txt"
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    diagnostic: dict[str, Any] | None = None
    try:
        shutil.copyfile(iq_path, scratch / "iq.npy")
        (scratch / "receiver.py").write_bytes(source)
        (scratch / "_shim.py").write_text(shim_source, encoding="utf-8")
        if preflight:
            preflight_result = ensure_sandbox(policy, profile)
            if not preflight_result.ok:
                result["status"] = "sandbox_unavailable"
                result["error"] = preflight_result.error
                return result, None
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [
                    str(SANDBOX_EXEC),
                    "-p",
                    profile,
                    str(policy.interpreter_literals[0]),
                    "-I",
                    "-B",
                    "_shim.py",
                ],
                cwd=scratch,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            breach, max_rss = _supervise(process, timeout_s=timeout_s, memory_mib=memory_mib)
        result["max_rss_mib"] = round(max_rss, 6)
        result["exit_code"] = process.returncode if process.returncode is not None and process.returncode >= 0 else None
        result["exit_signal"] = -process.returncode if process.returncode is not None and process.returncode < 0 else None
        result["stdout_tail"] = _tail(stdout_path)
        result["stderr_tail"] = _tail(stderr_path)
        if breach is not None:
            result["status"] = breach
            result["error"] = breach
            return result, _diagnostic_status(scratch / "status.json")
        if _group_alive(process.pid):
            _terminate_group(process, process.pid)
            result["status"] = "crashed"
            result["error"] = "process_group_survived"
            return result, None
        if process.returncode != 0:
            status, error, exit_signal = _classify_exit(process.returncode)
            result["status"] = status
            result["error"] = error
            result["exit_signal"] = exit_signal
            return result, _diagnostic_status(scratch / "status.json")
        bits, validation_error = load_receiver_output(scratch / "bits.npy")
        if validation_error is not None or bits is None:
            result["status"] = "bad_output"
            result["error"] = validation_error or "output_invalid"
            return result, _diagnostic_status(scratch / "status.json")
        retained_bits = bits_dir / "bits.npy"
        _retain_file(scratch / "bits.npy", retained_bits)
        result["status"] = "ok"
        result["error"] = None
        result["bits_sha256"] = _hash_file(retained_bits)
        result["bits_path"] = str(retained_bits)
        diagnostic = _diagnostic_status(scratch / "status.json")
        return result, diagnostic
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None:
            _terminate_group(process, process.pid)
        result["status"] = "sandbox_unavailable"
        # Never interpolate exc: SubprocessError.__str__ embeds the sandbox-exec argv,
        # which carries the rendered profile and every private/sealed path.
        result["error"] = f"sandbox launch failed: {type(exc).__name__}"
        return result, diagnostic
    finally:
        result["wall_time_s"] = round(time.monotonic() - started, 6)


def _environment_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _evaluator_module_hash() -> str:
    # hash the whole scoring surface, not just the CLI wrapper, or replay reports
    # `reproduced` after a scorer rewrite
    base = REPO_ROOT / "src" / "modembench"
    digest = sha256()
    digest.update(EVALUATOR_HASH_VERSION.encode("utf-8"))
    for name in ("evaluate.py", "evaluator.py", "framing.py"):
        digest.update(name.encode("utf-8"))
        digest.update(_hash_file(base / name).encode("ascii"))
    return digest.hexdigest()


def _run_evaluator(
    bits_path: Path, capture_dir: Path, token: Any = None
) -> tuple[dict[str, Any], int, list[str]]:
    # Truth goes to the child on stdin, not as a path: the evaluator CLI can refuse
    # sealed captures outright, and evaluator_command carries no sealed path.
    manifest_bytes = read_private_artifact(capture_dir, MANIFEST_ARTIFACT, token)
    payload = read_private_artifact(capture_dir, PAYLOAD_ARTIFACT, token)
    command = [sys.executable, "-m", "modembench.evaluate", str(bits_path), "--truth-stdin"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        input=encode_truth(manifest_bytes, payload),
    )
    try:
        evaluated = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("evaluator returned invalid JSON") from exc
    if not isinstance(evaluated, dict) or set(evaluated) != {"feedback", "internal"}:
        raise RuntimeError("evaluator returned an invalid object")
    internal = evaluated.get("internal")
    if not isinstance(internal, dict) or type(internal.get("packet_success")) is not bool:
        raise RuntimeError("evaluator omitted packet_success")
    expected_code = 0 if internal["packet_success"] else 1
    if completed.returncode != expected_code:
        raise RuntimeError("evaluator return code disagrees with packet_success")
    return evaluated, completed.returncode, command


def retained_receiver_name(receiver_hash: str, *, sealed: bool) -> str:
    """Retained-copy file name; sealed runs use a plain name because the digest is sealed identity."""
    return "receiver.py" if sealed else f"receiver-{receiver_hash}.py"


def _oracle_locator(
    capture_dir: Path, receiver_hash: str, source: bytes, token: Any = None
) -> Path | None:
    """Recognise a receiver that is already the capture's protected oracle."""
    name = oracle_artifact_name(receiver_hash)
    try:
        stored = read_private_artifact(capture_dir, name, token)
    except OSError:
        return None
    if stored != source:
        return None
    return private_artifact_path(capture_dir, name).resolve()


def _receiver_source(
    capture: Path, receiver: Path, token: Any = None
) -> tuple[bytes, str | None]:
    # A receiver inside a sealed root is only accepted when it is this capture's own
    # protected oracle, read through the chokepoint; anything else would exfiltrate
    # sealed truth into runs/.
    resolved = Path(os.path.realpath(os.fspath(receiver)))
    if sealed_root_containing(resolved) is None:
        if not receiver.is_file():
            raise ValueError(f"receiver source is missing: {receiver}")
        return receiver.read_bytes(), None
    protected = Path(os.path.realpath(str(private_dir(capture))))
    try:
        name = resolved.relative_to(protected).as_posix()
    except ValueError:
        raise SealedAccessError(
            "a receiver source inside a sealed root may only be that capture's own "
            "protected oracle"
        ) from None
    return read_private_artifact(capture, name, token), name


# What a sandbox run publishes, declared once as data; write_record applies it.

SANDBOX_PUBLIC_POLICY = RecordPolicy(
    name="sandbox-run-public",
    operations=(
        # run_dir is noise and bits_path names the sealed store
        Drop(keys=("run_dir", "bits_path"), at=("sandbox",), when=ALWAYS),
        Redact(key="sandbox", redactor="sandbox", when=SEALED),
        Substitute(key="capture_id", template="sealed:{capture_ref}", when=SEALED),
        # the receiver digest is sealed identity; null every address that spells it,
        # including the content-addressed artifact name
        Null(keys=("source_sha256", "artifact"), at=("receiver",), when=SEALED),
        Null(keys=("source_sha256",), at=("gate",), when=SEALED),
        # BER snaps to the dyadic grid: its raw denominator is the payload length
        Redact(key="feedback", redactor="evaluator_feedback", when=SEALED),
    ),
)

# Identity a sealed run may not state in the repository. RecordPolicy refuses to exist
# unless every key here is nulled for a sealed run.
SANDBOX_INTERNAL_IDENTITY = (
    "capture_id",
    "capture_dir",
    # receiver_sha256 is identity: the receiver may be the capture's own protected
    # oracle, making the digest an off-budget correctness oracle
    "receiver_sha256",
    "oracle_locator",
    "iq_sha256",
    "meta_sha256",
    "manifest_sha256",
    "payload_sha256",
    "rendered_profile",
    "sealed_roots",
    "permitted_capture_parents",
)

SANDBOX_INTERNAL_POLICY = RecordPolicy(
    name="sandbox-run-internal",
    identity=SANDBOX_INTERNAL_IDENTITY,
    operations=(
        Null(keys=SANDBOX_INTERNAL_IDENTITY, when=SEALED),
        # the command names the bits, which live in the sealed store
        Null(keys=("evaluator_command",), when=SEALED),
        Redact(key="sandbox", redactor="sandbox", when=SEALED),
        Redact(key="evaluator", redactor="evaluator", when=SEALED),
    ),
)


def run_receiver(
    capture_dir: str | os.PathLike[str],
    receiver_path: str | os.PathLike[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mib: float = DEFAULT_MEMORY_MIB,
    run_root: str | os.PathLike[str] = RUNS_ROOT,
    sealed_roots: Iterable[str | os.PathLike[str]] = (),
    permitted_capture_parents: Iterable[str | os.PathLike[str]] = (),
    enforce_ast: bool = True,
    preflight: bool = True,
    evaluate: bool = True,
    scratch_parent: str | os.PathLike[str] | None = None,
    sealed_token: Any = None,
) -> dict[str, Any]:
    """Gate, execute, evaluate, and retain one receiver run.

    Sealed captures require a live run token; redaction is applied only by write_record
    under the two policies above, never by hand here.
    """
    timeout = _validate_cap("timeout_s", timeout_s)
    memory = _validate_cap("memory_mib", memory_mib)
    sealed_roots = tuple(sealed_roots)
    permitted_capture_parents = tuple(permitted_capture_parents)
    capture = Path(os.path.realpath(os.fspath(capture_dir)))
    session = authorize_read(capture, sealed_token)
    # cross-check the gate's answer against the record writer's: a divergence means a
    # sealed read was about to be written as a dev run
    sealing = sealing_of_capture(capture, sealed_token)
    if sealing.sealed != (session is not None):
        raise SealedAccessError(
            "the read gate and the record writer disagree about whether this capture is "
            f"sealed (gate={session is not None}, record={sealing.sealed}): refusing to write "
            "a record whose redaction cannot be shown to match the access that produced it"
        )
    sealed = sealing.sealed
    capture_ref = capture_reference(capture)
    receiver = Path(receiver_path)
    source, protected_name = _receiver_source(capture, receiver, sealed_token)
    receiver_hash = _hash_bytes(source)
    inputs = _capture_inputs(capture, sealed_token)
    gate = check_source(source)
    run_id, retained_dir = _new_run_dir(
        Path(run_root).resolve(),
        receiver_hash,
        capture_ref if sealed else inputs["capture_id"],
        sealed=sealed,
    )
    # the recovered frame IS the sealed payload, so sealed bits stay in the sealed store
    bits_dir = sealed_run_artifact_dir(run_id, sealed_token) if sealed else retained_dir
    oracle_path = (
        private_artifact_path(capture, protected_name).resolve()
        if protected_name is not None
        else _oracle_locator(capture, receiver_hash, source, sealed_token)
    )
    receiver_artifact: Path | None = None
    if oracle_path is None:
        receiver_artifact = retained_dir / retained_receiver_name(receiver_hash, sealed=sealed)
        receiver_artifact.write_bytes(source)
    result = _base_result(receiver_hash=receiver_hash, iq_hash=inputs["iq_sha256"])
    result["run_id"] = run_id
    result["run_dir"] = str(retained_dir)
    scratch_path: Path | None = None
    profile = ""
    policy_template_sha = None
    rendered_profile_sha = None
    shim_source = ""
    diagnostic = None
    evaluator: dict[str, Any] = {
        "feedback": {
            "acquisition_success": False,
            "crc_pass": False,
            "aligned_ber": None,
            "error": None,
        },
        "internal": {"packet_success": False},
    }
    evaluator_returncode: int | None = None
    evaluator_command: list[str] | None = None
    try:
        if enforce_ast and not gate["ok"]:
            result["status"] = "ast_rejected"
            result["error"] = "ast_rejected"
            evaluator["feedback"]["error"] = "ast_rejected"
        elif sys.platform != "darwin" or not SANDBOX_EXEC.is_file():
            result["status"] = "sandbox_unavailable"
            result["error"] = "sandbox-exec is unavailable on this platform"
            evaluator["feedback"]["error"] = "sandbox_unavailable"
        else:
            scratch_path = Path(
                tempfile.mkdtemp(prefix="modembench-sandbox-", dir=scratch_parent)
            )
            policy = build_policy(
                capture,
                scratch_path,
                sealed_roots=sealed_roots,
                permitted_capture_parents=permitted_capture_parents,
                python=sys.executable,
            )
            profile = render_profile(policy)
            policy_template_sha, rendered_profile_sha = profile_hashes(policy, profile)
            shim_source = make_shim_source(
                sample_rate=inputs["sample_rate"],
                cpu_seconds=max(1, math.ceil(timeout)),
                fsize_bytes=DEFAULT_FSIZE_BYTES,
                project_src=str(REPO_ROOT / "src"),
            )
            result, diagnostic = _execute(
                policy=policy,
                profile=profile,
                shim_source=shim_source,
                source=source,
                iq_path=inputs["iq_path"],
                run_dir=retained_dir,
                bits_dir=bits_dir,
                receiver_hash=receiver_hash,
                iq_hash=inputs["iq_sha256"],
                timeout_s=timeout,
                memory_mib=memory,
                preflight=preflight,
            )
            result["run_id"] = run_id
            result["run_dir"] = str(retained_dir)
            if result["status"] != "ok":
                evaluator["feedback"]["error"] = result["status"]
            if result["status"] == "ok" and evaluate:
                try:
                    evaluator, evaluator_returncode, evaluator_command = _run_evaluator(
                        Path(result["bits_path"]), capture, sealed_token
                    )
                except RuntimeError as exc:
                    evaluator["feedback"]["error"] = "evaluator_invalid"
                    result["error"] = str(exc)
        # nothing below is redacted by hand; write_record applies the policies
        public_document = {
            "capture_id": inputs["capture_id"],
            "receiver": {
                # same key name the AST gate uses, so SEALED_RECEIVER_DIGEST_KEYS covers both
                "source_sha256": receiver_hash,
                "artifact": receiver_artifact.name if receiver_artifact is not None else None,
                "retention": "run_copy" if receiver_artifact is not None else "protected_reference",
            },
            "gate": {
                **gate,
                "policy_version": AST_POLICY_VERSION,
            },
            "sandbox": result,
            "feedback": evaluator["feedback"],
        }
        internal_dir = retained_dir / ".orchestrator"
        internal_dir.mkdir(mode=0o700)
        identity = SANDBOX_INTERNAL_POLICY.identity_fields(
            capture_id=inputs["capture_id"],
            capture_dir=str(capture),
            receiver_sha256=receiver_hash,
            oracle_locator=str(oracle_path) if oracle_path is not None else None,
            iq_sha256=inputs["iq_sha256"],
            meta_sha256=inputs["meta_sha256"],
            manifest_sha256=inputs["manifest_sha256"],
            payload_sha256=inputs["payload_sha256"],
            rendered_profile=profile or None,
            sealed_roots=[str(Path(path).resolve()) for path in sealed_roots],
            permitted_capture_parents=[
                str(Path(path).resolve()) for path in permitted_capture_parents
            ],
        )
        internal_document = {
            **identity,
            "capture_ref": capture_ref if sealed else None,
            "sealed": sealed,
            # relative on purpose: the absolute form would put the sealed root's path in
            # the repository
            "sealed_run_record": (
                str(sealed_run_record_path(run_id, ".").relative_to(".")) if sealed else None
            ),
            "gate_policy_version": AST_POLICY_VERSION,
            "kernel_policy_version": POLICY_VERSION,
            "policy_template_sha256": policy_template_sha,
            "rendered_profile_sha256": rendered_profile_sha,
            "shim_version": SHIM_VERSION,
            "shim_sha256": shim_sha256(shim_source) if shim_source else None,
            "sandbox": result,
            "diagnostic_status": diagnostic,
            "evaluator": evaluator,
            "evaluator_returncode": evaluator_returncode,
            "evaluator_module_sha256": _evaluator_module_hash(),
            "evaluator_command": evaluator_command,
            "environment": _environment_record(),
            "caps": {
                "timeout_s": timeout,
                "memory_mib": memory,
                "memory_semantics": "sampled_process_group_rss_guardrail",
                "fsize_bytes": DEFAULT_FSIZE_BYTES,
            },
        }
        if sealed:
            # written before the repository records, so a crash between them leaves a
            # replayable run rather than an orphaned opaque reference
            write_sealed_run_record(
                run_id,
                {
                    "run_id": run_id,
                    **identity,
                    **{
                        key: result.get(key)
                        for key in SEALED_SANDBOX_REDACTIONS
                        if key not in SANDBOX_INTERNAL_IDENTITY
                    },
                    "bits_path": result.get("bits_path"),
                    "evaluator_command": evaluator_command,
                },
                sealed_token,
            )
        write_record(
            retained_dir / "run.json",
            public_document,
            SANDBOX_PUBLIC_POLICY,
            sealing=sealing,
        )
        write_record(
            internal_dir / "run-internal.json",
            internal_document,
            SANDBOX_INTERNAL_POLICY,
            sealing=sealing,
        )
    finally:
        if scratch_path is not None:
            shutil.rmtree(scratch_path, ignore_errors=False)
    result["feedback"] = evaluator["feedback"]
    result["internal"] = evaluator["internal"]
    return result


# Bound, not redefined: sealed-sandbox tests import these private names from here.
_sealed_safe_sandbox = sealed_safe_sandbox
_sealed_safe_evaluator = sealed_safe_evaluator


def _environment_changed(
    internal: dict[str, Any], run_dir: Path, token: Any = None
) -> str | None:
    if internal.get("environment") != _environment_record():
        return "runtime version or platform changed"
    if internal.get("evaluator_module_sha256") != _evaluator_module_hash():
        return "evaluator module changed"
    capture = Path(internal["capture_dir"])
    inputs = _capture_inputs(capture, token)
    for key in ("iq_sha256", "meta_sha256", "manifest_sha256", "payload_sha256"):
        if internal.get(key) != inputs[key]:
            return f"capture provenance changed: {key}"
    receiver_hash = internal["receiver_sha256"]
    oracle = internal.get("oracle_locator")
    source_path = Path(oracle) if oracle else run_dir / retained_receiver_name(
        receiver_hash, sealed=internal.get("sealed") is True
    )
    if not source_path.is_file() or _hash_file(source_path) != receiver_hash:
        return "receiver source changed"
    with tempfile.TemporaryDirectory(prefix="modembench-replay-policy-") as scratch:
        policy = build_policy(
            capture,
            scratch,
            sealed_roots=internal.get("sealed_roots", ()),
            permitted_capture_parents=internal.get("permitted_capture_parents", ()),
            python=sys.executable,
        )
        template_hash, _rendered_hash = profile_hashes(policy)
    if internal.get("policy_template_sha256") != template_hash:
        return "sandbox policy changed"
    if internal.get("gate_policy_version") != AST_POLICY_VERSION:
        return "AST policy changed"
    if internal.get("shim_version") != SHIM_VERSION:
        return "shim version changed"
    caps = internal.get("caps", {})
    current_shim = make_shim_source(
        sample_rate=inputs["sample_rate"],
        cpu_seconds=max(1, math.ceil(float(caps.get("timeout_s", 0)))),
        fsize_bytes=DEFAULT_FSIZE_BYTES,
        project_src=str(REPO_ROOT / "src"),
    )
    if internal.get("shim_sha256") != shim_sha256(current_shim):
        return "shim source changed"
    return None


def replay_run(
    run_id: str,
    *,
    run_root: str | os.PathLike[str] = RUNS_ROOT,
    sealed_token: Any = None,
) -> dict[str, Any]:
    """Replay a retained run and classify reproducibility.

    Replay is itself a sealed read, so a sealed capture requires a live run token, which
    is also forwarded to the replayed run.
    """
    root = Path(run_root).resolve()
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be a single path component")
    original_dir = root / run_id
    public = _read_json(original_dir / "run.json")
    internal = _read_json(original_dir / ".orchestrator" / "run-internal.json")
    if internal.get("sealed") is True:
        # the repository record holds no sealed identity; reconstitute it from the
        # sealed store, which needs the token
        internal = {**internal, **read_sealed_run_record(run_id, sealed_token)}
    # before anything touches the capture: _environment_changed re-hashes private truth
    authorize_read(Path(os.path.realpath(str(internal["capture_dir"]))), sealed_token)
    changed = _environment_changed(internal, original_dir, sealed_token)
    if changed is not None:
        return {"status": "changed_environment", "run_id": run_id, "detail": changed}
    receiver_hash = internal["receiver_sha256"]
    receiver_path = (
        Path(internal["oracle_locator"])
        if internal.get("oracle_locator")
        else original_dir / retained_receiver_name(
            receiver_hash, sealed=internal.get("sealed") is True
        )
    )
    caps = internal["caps"]
    replayed = run_receiver(
        internal["capture_dir"],
        receiver_path,
        timeout_s=caps["timeout_s"],
        memory_mib=caps["memory_mib"],
        run_root=root,
        sealed_roots=internal.get("sealed_roots", ()),
        permitted_capture_parents=internal.get("permitted_capture_parents", ()),
        sealed_token=sealed_token,
    )
    original_sandbox = public["sandbox"]
    # a sealed run's bits digest lives only in the sealed record; run.json would compare
    # None with None and always report `reproduced`
    original_bits_sha256 = (
        internal.get("bits_sha256") if internal.get("sealed") is True
        else original_sandbox.get("bits_sha256")
    )
    same = (
        original_sandbox.get("status") == replayed.get("status")
        and original_bits_sha256 == replayed.get("bits_sha256")
        and public.get("feedback") == replayed.get("feedback")
    )
    return {
        "status": "reproduced" if same else "nondeterministic",
        "run_id": run_id,
        "replay_run_id": replayed["run_id"],
        "original_bits_sha256": original_bits_sha256,
        "replay_bits_sha256": replayed.get("bits_sha256"),
    }
