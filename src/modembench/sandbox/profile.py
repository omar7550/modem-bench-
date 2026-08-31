"""macOS SBPL rendering and fail-closed sandbox preflight."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from typing import Iterable

from ..sealed import PRIVATE_ARTIFACTS, private_dir, sealed_denial_roots

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
POLICY_VERSION = "modembench-sbpl-v7-sealed0"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMP_ROOTS = (Path("/private/tmp"), Path("/private/var/folders"))
DEFAULT_VOLUME_ROOT = Path("/Volumes")


class SandboxUnavailable(RuntimeError):
    """Raised when the kernel boundary cannot be proved for a run."""


@dataclass(frozen=True)
class SandboxPolicy:
    capture_dir: Path
    capture_private_dir: Path
    scratch_dir: Path
    sealed_roots: tuple[Path, ...]
    permitted_capture_parents: tuple[Path, ...]
    repo_root: Path
    home: Path
    temp_roots: tuple[Path, ...]
    volume_root: Path
    interpreter_literals: tuple[Path, ...]

    @property
    def canonical_key(self) -> tuple[str, ...]:
        return (
            str(self.capture_dir),
            str(self.capture_private_dir),
            *(str(path) for path in self.sealed_roots),
            *(str(path) for path in self.interpreter_literals),
            str(self.repo_root),
            str(self.home),
            *(str(path) for path in self.temp_roots),
            str(self.volume_root),
            POLICY_VERSION,
        )


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    error: str | None = None


_PREFLIGHT_CACHE: set[tuple[str, ...]] = set()
_PREFLIGHT_LOCK = threading.Lock()


def _real(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def interpreter_literals(python: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    """Every path the interpreter may exec. Framework builds re-exec into Python.app;
    omitting that target kills every child at posix_spawn."""
    venv_python = Path(python or sys.executable).absolute()
    resolved_python = _real(venv_python)
    base_executable = _real(getattr(sys, "_base_executable", None) or resolved_python)
    literals = [venv_python, resolved_python, base_executable]
    for anchor in (base_executable, resolved_python):
        app_binary = _real(
            anchor.parent.parent / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
        )
        if app_binary.exists():
            literals.append(app_binary)
    seen: dict[str, Path] = {}
    for path in literals:
        seen.setdefault(str(path), path)
    return tuple(seen.values())


def build_policy(
    capture_dir: str | os.PathLike[str],
    scratch_dir: str | os.PathLike[str],
    *,
    sealed_roots: Iterable[str | os.PathLike[str]] = (),
    permitted_capture_parents: Iterable[str | os.PathLike[str]] = (),
    python: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> SandboxPolicy:
    capture = _real(capture_dir)
    scratch = _real(scratch_dir)
    repo = _real(repo_root)
    default_parent = _real(repo / "captures")
    permitted = (default_parent, *(_real(path) for path in permitted_capture_parents))
    if not any(_is_within(capture, parent) for parent in permitted):
        raise ValueError("capture directory is outside configured permitted-capture parents")
    protected = _real(private_dir(capture))
    if not capture.is_dir() or not protected.is_dir():
        raise ValueError("capture directory must contain a private directory")
    # sealed_denial_roots() is always included: callers may only add roots, never
    # replace them, so an override naming a decoy cannot leave the real store undenied
    sealed = _dedupe((*sealed_denial_roots(), *(_real(path) for path in sealed_roots)))
    return SandboxPolicy(
        capture_dir=capture,
        capture_private_dir=protected,
        scratch_dir=scratch,
        sealed_roots=sealed,
        permitted_capture_parents=tuple(permitted),
        repo_root=repo,
        home=_real(Path.home()),
        # deny scratch.parent too: TMPDIR can sit outside DEFAULT_TEMP_ROOTS, and a
        # sibling scratch holds another run's oracle constants
        temp_roots=_dedupe((*(_real(path) for path in DEFAULT_TEMP_ROOTS), scratch.parent)),
        volume_root=_real(DEFAULT_VOLUME_ROOT),
        interpreter_literals=interpreter_literals(python),
    )


def _dedupe(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: dict[str, Path] = {}
    for path in paths:
        seen.setdefault(str(path), path)
    return tuple(seen.values())


def _literal(value: str | os.PathLike[str]) -> str:
    return json.dumps(os.fspath(value), ensure_ascii=False)


def render_profile(policy: SandboxPolicy) -> str:
    denied = (
        policy.repo_root,
        policy.home,
        *policy.temp_roots,
        policy.volume_root,
        policy.capture_dir,
        *policy.sealed_roots,
    )
    deny_filters = " ".join(f"(subpath {_literal(path)})" for path in denied)
    # deny metadata on truth roots: st_size of payload.bin is the payload length
    truth_roots = (policy.capture_dir, *policy.sealed_roots, *policy.temp_roots)
    truth_filters = " ".join(f"(subpath {_literal(path)})" for path in truth_roots)
    exec_filters = " ".join(f"(literal {_literal(path)})" for path in policy.interpreter_literals)
    return "\n".join(
        (
            "(version 1)",
            "(deny default)",
            "(allow sysctl-read)",
            "(allow file-read-metadata)",
            "(allow file-read*)",
            f"(deny file-read* {deny_filters})",
            f"(deny file-read-metadata {truth_filters})",
            f"(allow file-read-metadata (subpath {_literal(policy.scratch_dir)}))",
            f"(allow file-read* (subpath {_literal(policy.repo_root / '.venv')}))",
            f"(allow file-read* file-write* (subpath {_literal(policy.scratch_dir)}))",
            '(allow file-write-data (literal "/dev/null"))',
            f"(allow process-exec {exec_filters})",
            "",
        )
    )


def profile_hashes(policy: SandboxPolicy, rendered: str | None = None) -> tuple[str, str]:
    profile = rendered if rendered is not None else render_profile(policy)
    rendered_hash = sha256(profile.encode("utf-8")).hexdigest()
    normalized = profile.replace(_literal(policy.scratch_dir), '"<SCRATCH>"')
    normalized = normalized.replace(_literal(policy.scratch_dir.parent), '"<SCRATCH_PARENT>"')
    template_material = f"{POLICY_VERSION}\n{normalized}".encode("utf-8")
    return sha256(template_material).hexdigest(), rendered_hash


_PREFLIGHT_CODE = r'''
import json, os, signal, socket, subprocess, sys
capture_file, outside_write, port_text, helper_text, scratch = sys.argv[1:]
out = {}
try:
    open(capture_file, "rb").read(1)
except PermissionError:
    out["capture_read_denied"] = True
else:
    out["capture_read_denied"] = False
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    s.connect(("127.0.0.1", int(port_text)))
except PermissionError:
    out["network_denied"] = True
except OSError as exc:
    out["network_denied"] = False
    out["network_error"] = repr(exc)
finally:
    try: s.close()
    except Exception: pass
try:
    open(outside_write, "wb").write(b"x")
except PermissionError:
    out["outside_write_denied"] = True
else:
    out["outside_write_denied"] = False
try:
    subprocess.run(["/usr/bin/true"], check=False)
except PermissionError:
    out["exec_denied"] = True
else:
    out["exec_denied"] = False
try:
    child = os.fork()
except PermissionError:
    out["fork_denied"] = True
else:
    out["fork_denied"] = False
    if child == 0: os._exit(0)
    os.waitpid(child, 0)
try:
    os.kill(int(helper_text), signal.SIGTERM)
except PermissionError:
    out["signal_denied"] = True
else:
    out["signal_denied"] = False
probe = os.path.join(scratch, ".preflight-probe")
with open(probe, "wb") as stream:
    stream.write(b"ok")
out["scratch_write"] = os.path.exists(probe)
os.unlink(probe)
out["scratch_write"] = out["scratch_write"] and not os.path.exists(probe)
print(json.dumps(out, sort_keys=True))
'''


def _sentinel(policy: SandboxPolicy) -> Path:
    for name in PRIVATE_ARTIFACTS:
        candidate = policy.capture_private_dir / name
        if candidate.is_file():
            return candidate
    raise SandboxUnavailable("capture private root has no preflight sentinel")


def _run_preflight(policy: SandboxPolicy, rendered_profile: str) -> PreflightResult:
    if sys.platform != "darwin" or not SANDBOX_EXEC.is_file():
        return PreflightResult(False, "sandbox-exec is unavailable on this platform")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    helper: subprocess.Popen[bytes] | None = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        helper = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        outside_write = policy.capture_private_dir / ".modembench-preflight-write"
        command = [
            str(SANDBOX_EXEC),
            "-p",
            rendered_profile,
            str(policy.interpreter_literals[0]),
            "-I",
            "-B",
            "-c",
            _PREFLIGHT_CODE,
            str(_sentinel(policy)),
            str(outside_write),
            str(port),
            str(helper.pid),
            str(policy.scratch_dir),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        helper_survived = helper.poll() is None
        if outside_write.exists():
            outside_write.unlink()
            return PreflightResult(False, "sandbox allowed an outside-scratch write")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return PreflightResult(False, f"preflight returned invalid JSON (exit {completed.returncode})")
        required = {
            "capture_read_denied",
            "network_denied",
            "outside_write_denied",
            "exec_denied",
            "fork_denied",
            "signal_denied",
            "scratch_write",
        }
        if completed.returncode != 0 or any(result.get(name) is not True for name in required):
            unmet = sorted(name for name in required if result.get(name) is not True)
            return PreflightResult(
                False, f"sandbox preflight failed (exit {completed.returncode}; unmet: {','.join(unmet) or 'none'})"
            )
        if not helper_survived:
            return PreflightResult(False, "sandbox signal probe killed its sacrificial helper")
        return PreflightResult(True)
    except (OSError, subprocess.SubprocessError, SandboxUnavailable) as exc:
        # never interpolate exc: TimeoutExpired.__str__ embeds argv (profile, sealed
        # paths) and this string reaches the agent-visible run.json
        return PreflightResult(False, f"sandbox preflight error: {type(exc).__name__}")
    finally:
        listener.close()
        if helper is not None:
            try:
                os.killpg(helper.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            helper.wait(timeout=2)


def ensure_sandbox(policy: SandboxPolicy, rendered_profile: str | None = None) -> PreflightResult:
    """Prove the rendered policy once, caching by canonical policy inputs."""
    profile = rendered_profile if rendered_profile is not None else render_profile(policy)
    key = policy.canonical_key
    with _PREFLIGHT_LOCK:
        if key in _PREFLIGHT_CACHE:
            return PreflightResult(True)
        result = _run_preflight(policy, profile)
        if result.ok:
            _PREFLIGHT_CACHE.add(key)
        return result


def clear_preflight_cache() -> None:
    """Clear the successful-preflight cache (primarily for deterministic tests)."""
    with _PREFLIGHT_LOCK:
        _PREFLIGHT_CACHE.clear()
