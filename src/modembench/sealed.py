"""Sealed-split access control: policy, authorizations, hash-chained log, and tokens.

Content-exposing operations (generation, sealed capture reads, the salt) require a
counted run token and are logged; there is deliberately no uncounted authorization
class. Read-only integrity verification is unauthorized and unlogged by design.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Any, Iterator

from .merkle import CaptureLeaf, MerkleError, order_leaves, split_root

SEALED_SCHEMA_VERSION = "1.0"
ANCHOR_SCHEMA_VERSION = "1.0"
POLICY_ENV_VAR = "MODEMBENCH_SEALED_POLICY"
MAX_OPENS_ENV_VAR = "MODEMBENCH_SEALED_MAX_OPENS"
DEFAULT_POLICY_RELATIVE = "data/sealed_access.json"
LOG_ANCHOR_RELATIVE = "data/sealed_log_anchor.json"
COMMITMENTS_RELATIVE = "data/commitments"
LEAVES_FILENAME = "leaves.json"
SALT_FILENAME = "salt.bin"
SALT_BYTES = 32
# the private-artifact vocabulary lives only here; the chokepoint test bans the
# literals in every other module
PRIVATE_DIRNAME = "private"
MANIFEST_ARTIFACT = "manifest.json"
PAYLOAD_ARTIFACT = "payload.bin"
SEED_ARTIFACT = "seed.json"
ORACLE_SUBDIR = "oracle"
PRIVATE_ARTIFACTS = (MANIFEST_ARTIFACT, PAYLOAD_ARTIFACT, SEED_ARTIFACT)
IQ_ARTIFACT = "iq.npy"
META_ARTIFACT = "meta.json"
RUN_PURPOSE = "run"
# re-entry window for a session another process left open; recovery is close_sealed
SESSION_LEASE_SECONDS = 24 * 60 * 60
# O_NOFOLLOW pins the checked path to the opened path; 0 where unsupported degrades to
# the (st_dev, st_ino) re-verification, which still refuses a swapped target
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# openat pins the final component to the authorized directory inode
_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
# exactly one authorization class, counted; an uncounted one would make max_opens
# bound nothing
PURPOSES = (RUN_PURPOSE,)
COUNTED_PURPOSES = (RUN_PURPOSE,)


class SealedAccessError(RuntimeError):
    """A sealed-resource access was refused, or its policy could not be proved."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _running_source_root() -> Path:
    # separate from _project_root so relocating the ledger cannot relocate the
    # provenance measurement
    return Path(__file__).resolve().parents[2]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _real(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def _is_within(path: Path, parent: Path) -> bool:
    if path == parent:
        return True
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- policy -------------------------------------------------------------------------


@dataclass(frozen=True)
class RunAuthorization:
    """One named entry of the repository policy's authorized_runs list."""

    run_name: str
    split_id: str | None = None
    opened_at: str | None = None
    session_id: str | None = None
    closed_at: str | None = None

    @property
    def consumed(self) -> bool:
        return self.opened_at is not None

    @property
    def live(self) -> bool:
        """Consumed and not yet closed; the state a restart may re-enter."""
        return self.consumed and self.closed_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "split_id": self.split_id,
            "opened_at": self.opened_at,
            "session_id": self.session_id,
            "closed_at": self.closed_at,
        }


@dataclass(frozen=True)
class SealedPolicy:
    """The resolved sealed-access policy for one process. ledger_path is always the
    repository policy file; policy_path is what this process was pointed at."""

    policy_path: Path
    ledger_path: Path
    anchor_path: Path
    sealed_root: Path
    repo_sealed_root: Path
    log_path: Path
    max_opens: int
    narrowed_by_env: bool
    policy_override: bool
    authorized_runs: tuple[RunAuthorization, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_path": str(self.policy_path),
            "ledger_path": str(self.ledger_path),
            "anchor_path": str(self.anchor_path),
            "sealed_root": str(self.sealed_root),
            "repo_sealed_root": str(self.repo_sealed_root),
            "log_path": str(self.log_path),
            "max_opens": self.max_opens,
            "narrowed_by_env": self.narrowed_by_env,
            "policy_override": self.policy_override,
            "authorized_runs": [entry.to_dict() for entry in self.authorized_runs],
        }


def policy_path(root: Path | None = None) -> Path:
    """Return the policy file this process was pointed at (override or repository)."""
    override = os.environ.get(POLICY_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return ledger_path(root)


def ledger_path(root: Path | None = None) -> Path:
    """Return the repository policy file, the only consumption ledger."""
    return (root or _project_root()) / DEFAULT_POLICY_RELATIVE


def log_anchor_path(root: Path | None = None) -> Path:
    """Return the committed chain-head anchor (not a provenance input)."""
    return (root or _project_root()) / LOG_ANCHOR_RELATIVE


def _env_max_opens() -> int | None:
    raw = os.environ.get(MAX_OPENS_ENV_VAR)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise SealedAccessError(f"{MAX_OPENS_ENV_VAR} must be a non-negative integer") from exc
    if value < 0:
        raise SealedAccessError(f"{MAX_OPENS_ENV_VAR} must be a non-negative integer")
    return value


def _read_policy_document(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SealedAccessError(f"sealed access policy is unreadable: {path}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SealedAccessError(f"sealed access policy is malformed: {path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SEALED_SCHEMA_VERSION:
        raise SealedAccessError(f"sealed access policy is malformed: {path}")
    return document


def _write_policy_document(path: Path, document: dict[str, Any]) -> None:
    # key order and non-ASCII preserved: this tracked file's hash lands in downstream
    # reports, so a rewrite must change only the values it means to change
    payload = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _replace_durably(path, payload)


def _replace_durably(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validated_max_opens(document: dict[str, Any], source: Path) -> int:
    value = document.get("max_opens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SealedAccessError(f"sealed access policy has an invalid max_opens: {source}")
    return value


def _validated_sealed_root(document: dict[str, Any], source: Path) -> Path:
    value = document.get("sealed_root")
    if not isinstance(value, str) or not value:
        raise SealedAccessError(f"sealed access policy has no sealed_root: {source}")
    # refuse ~ roots: they move with $HOME, relocating the boundary itself; a relative
    # root would likewise resolve against CWD
    if "~" in value:
        raise SealedAccessError(
            "sealed access policy sealed_root must be a literal absolute path: a '~' resolves "
            "through $HOME, which would let the environment relocate the boundary itself"
        )
    path = Path(value)
    if not path.is_absolute():
        raise SealedAccessError("sealed access policy sealed_root must be an absolute path")
    return path


def _assert_outside_repository(candidate: Path, project: Path, *, origin: str) -> None:
    # a sealed root inside the repo is a sealed root inside the agent's working tree;
    # enforced for the repository policy's own root, not just overrides
    for repository_root in {project.resolve(), _running_source_root().resolve()}:
        if _is_within(_real(candidate), _real(repository_root)):
            raise SealedAccessError(
                f"{origin} may not place the sealed root inside the repository"
            )


def _parse_authorizations(value: Any, source: Path) -> tuple[RunAuthorization, ...]:
    """Parse authorized_runs; absent or malformed refuses."""
    if not isinstance(value, list):
        raise SealedAccessError(f"sealed access policy has no authorized_runs list: {source}")
    entries: list[RunAuthorization] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise SealedAccessError(f"sealed access policy has a malformed authorized_runs entry: {source}")
        name = item.get("run_name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise SealedAccessError(f"an authorized_runs entry has no usable run_name: {source}")
        if name in seen:
            raise SealedAccessError(f"authorized_runs names a run twice: {name!r}")
        seen.add(name)
        split_id = item.get("split_id")
        opened_at = item.get("opened_at")
        session_id = item.get("session_id")
        closed_at = item.get("closed_at")
        for field, field_value in (
            ("split_id", split_id), ("opened_at", opened_at), ("session_id", session_id),
            ("closed_at", closed_at),
        ):
            if field_value is not None and not isinstance(field_value, str):
                raise SealedAccessError(f"authorized_runs entry {name!r} has a malformed {field}")
        if (opened_at is None) != (split_id is None):
            raise SealedAccessError(
                f"authorized_runs entry {name!r} records consumption without binding a split_id"
            )
        if closed_at is not None and opened_at is None:
            raise SealedAccessError(
                f"authorized_runs entry {name!r} records a close without an open"
            )
        entries.append(RunAuthorization(name, split_id, opened_at, session_id, closed_at))
    return tuple(entries)


def load_policy(path: str | os.PathLike[str] | None = None, *, root: Path | None = None) -> SealedPolicy:
    """Load the sealed-access policy, refusing when it is absent or malformed.

    An override may only relocate sealed_root and narrow max_opens; the ledger always
    comes from the repository policy, or a fresh override file would be a fresh budget.
    """
    project = (root or _project_root()).resolve()
    ledger = project / DEFAULT_POLICY_RELATIVE
    repository = _read_policy_document(ledger)

    max_opens = _validated_max_opens(repository, ledger)
    authorized = _parse_authorizations(repository.get("authorized_runs"), ledger)
    log_relative = repository.get("log_path")
    if not isinstance(log_relative, str) or not log_relative:
        raise SealedAccessError(f"sealed access policy has no log_path: {ledger}")
    repo_sealed_root = _validated_sealed_root(repository, ledger)
    _assert_outside_repository(
        repo_sealed_root, project, origin="the repository sealed access policy"
    )
    sealed_path = repo_sealed_root

    resolved = Path(path).expanduser() if path is not None else policy_path(project)
    override = _real(resolved) != _real(ledger)
    if override:
        document = _read_policy_document(resolved)
        for field in ("authorized_runs", "log_path"):
            if field in document:
                raise SealedAccessError(
                    "an overriding sealed access policy may not supply "
                    f"{field!r}: the consumption ledger is always the repository policy"
                )
        sealed_path = _validated_sealed_root(document, resolved)
        max_opens = min(max_opens, _validated_max_opens(document, resolved))
        _assert_outside_repository(
            sealed_path, project, origin="an overriding sealed access policy"
        )

    narrowed = False
    env_limit = _env_max_opens()
    if env_limit is not None and env_limit < max_opens:
        # environment may narrow, never widen
        max_opens = env_limit
        narrowed = True
    log = Path(log_relative).expanduser()
    if not log.is_absolute():
        log = ledger.parent / log
    return SealedPolicy(
        policy_path=resolved,
        ledger_path=ledger,
        anchor_path=project / LOG_ANCHOR_RELATIVE,
        sealed_root=sealed_path,
        repo_sealed_root=repo_sealed_root,
        log_path=log,
        max_opens=max_opens,
        narrowed_by_env=narrowed,
        policy_override=override,
        authorized_runs=authorized,
    )


def configured_sealed_root(root: Path | None = None) -> Path:
    """Resolve the sealed root this process is pointed at."""
    return _real(load_policy(root=root).sealed_root)


def sealed_denial_roots(root: Path | None = None) -> tuple[Path, ...]:
    """Every root the sandbox must deny: the configured one and the repository's, so an
    override naming a decoy cannot leave the real store undenied."""
    active = load_policy(root=root)
    roots = (_real(active.sealed_root), _real(active.repo_sealed_root))
    return tuple(dict.fromkeys(roots))


def sealed_salt_path(sealed_root: str | os.PathLike[str]) -> Path:
    return Path(sealed_root) / SALT_FILENAME


def _gated_roots(token: Any = None) -> tuple[Path, ...]:
    # THE gate decision: derived from process state only, no relocating parameter.
    # Union of configured, repository, and token roots so a decoy override cannot
    # leave the real store ungated.
    active = load_policy()
    roots = [_real(active.sealed_root), _real(active.repo_sealed_root)]
    if isinstance(token, SealedToken):
        roots.append(_real(token.sealed_root))
    return tuple(dict.fromkeys(roots))


def gated_sealed_roots(token: Any = None) -> tuple[Path, ...]:
    """The gate's root union as data, so records.py derives sealedness from the same
    answer the read gate gave."""
    return _gated_roots(token)


def _assert_sealed_access_permitted() -> None:
    """max_opens == 0 disables sealed access outright, re-entry included."""
    if load_policy().max_opens == 0:
        raise SealedAccessError(
            "sealed access is disabled: max_opens is 0, which refuses opening a session, "
            "re-entering one another process left live, and every sealed read, write, "
            "generation and salt extraction"
        )


def _sealed_seed_gate(master_seed: int) -> bool:
    # enforcement point: takes no registry, or the caller would decide which seeds
    # need permission
    return is_sealed_seed(master_seed)


def sealed_root_containing(path: str | os.PathLike[str]) -> Path | None:
    """The sealed root a path resolves inside, if any."""
    target = _real(path)
    for root in _gated_roots():
        if _is_within(target, root):
            return root
    return None


def read_sealed_salt(sealed_root: str | os.PathLike[str], token: Any = None) -> bytes:
    """Read the sealed salt. The salt regenerates the whole sealed set, so this needs a
    live counted run token and is logged."""
    _assert_sealed_access_permitted()
    live = _assert_run_token(token, "reading the sealed salt")
    # resolved once; the file itself is checked, not just its directory
    resolved = _real(sealed_salt_path(sealed_root))
    if not _is_within(_real(sealed_root), _real(live.sealed_root)) or not _is_within(
        resolved, _real(live.sealed_root)
    ):
        raise SealedAccessError("the sealed salt may only be read inside the authorized sealed root")
    active = _live_policy(live)
    # logged before the bytes are handed over, so a crash cannot leave a recordless read
    with _policy_lock(active):
        _append_and_anchor(
            active,
            _session_record(
                event="salt_read",
                split_id=live.split_id,
                run_name=live.run_name,
                purpose=live.purpose,
                session_id=live.session_id,
                timestamp=_utc_now(),
                policy=active,
                merkle_root=None,
                detail="sealed salt extracted",
                provenance=_provenance(),
            ),
        )
    try:
        salt = _read_authorized_object(_decide_artifact(resolved))
    except OSError as exc:
        raise SealedAccessError(f"sealed salt is unreadable: {resolved}") from exc
    if len(salt) != SALT_BYTES:
        raise SealedAccessError(f"sealed salt must be exactly {SALT_BYTES} bytes: {resolved}")
    return salt


def write_sealed_salt(sealed_root: str | os.PathLike[str], salt: bytes | None = None) -> bytes:
    """Create the sealed salt once; an existing differing salt is a hard error.

    Deliberately unauthorized: writing cannot exfiltrate, overwrite is refused, and the
    published commitment binds salt_sha256. Still obeys the max_opens kill switch.
    """
    _assert_sealed_access_permitted()
    material = secrets.token_bytes(SALT_BYTES) if salt is None else bytes(salt)
    if len(material) != SALT_BYTES:
        raise SealedAccessError(f"sealed salt must be exactly {SALT_BYTES} bytes")
    path = sealed_salt_path(sealed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != material:
            raise SealedAccessError(f"refusing to replace an existing sealed salt: {path}")
        return material
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with temporary.open("xb") as stream:
        stream.write(material)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return material


def salt_sha256(salt: bytes) -> str:
    """What the repo commits to; publish this, never the salt."""
    return sha256(bytes(salt)).hexdigest()


# --- seed registry ------------------------------------------------------------------


def load_seed_registry(root: Path | None = None) -> dict[str, Any]:
    """Load the seed registry, refusing when it is absent or malformed."""
    path = (root or _project_root()) / "data/seed_registry.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedAccessError(f"seed registry is unreadable: {path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "1.0":
        raise SealedAccessError("seed registry is malformed")
    reservations = document.get("reservations")
    plain_max = document.get("plain_seed_max")
    if not isinstance(reservations, list) or not all(isinstance(item, dict) for item in reservations):
        raise SealedAccessError("seed registry is malformed")
    if isinstance(plain_max, bool) or not isinstance(plain_max, int) or plain_max < 0:
        raise SealedAccessError("seed registry has no usable plain_seed_max")
    return document


def plain_seed_max(registry: dict[str, Any] | None = None, *, root: Path | None = None) -> int:
    """The inclusive upper bound of the plain (non-derived) seed space."""
    return int((registry if registry is not None else load_seed_registry(root))["plain_seed_max"])


def sealed_reservations(registry: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(entry for entry in registry["reservations"] if entry.get("sealed") is True)


def is_sealed_seed(
    master_seed: int, registry: dict[str, Any] | None = None, *, root: Path | None = None
) -> bool:
    """Decide whether a seed belongs to a sealed reservation.

    Everything above plain_seed_max is sealed by construction; explicit start/stop
    blocks are honoured as well.
    """
    document = registry if registry is not None else load_seed_registry(root)
    if master_seed > plain_seed_max(document):
        return True
    for entry in sealed_reservations(document):
        start = entry.get("start")
        stop = entry.get("stop")
        if isinstance(start, int) and isinstance(stop, int) and start <= master_seed <= stop:
            return True
    return False


# --- sessions and tokens ------------------------------------------------------------


@dataclass(frozen=True)
class SealedToken:
    """Proof that a live :func:`open_sealed` session authorizes sealed access."""

    session_id: str
    split_id: str
    run_name: str
    purpose: str
    sealed_root: Path
    opened_at: str
    counted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "split_id": self.split_id,
            "run_name": self.run_name,
            "purpose": self.purpose,
            "sealed_root": str(self.sealed_root),
            "opened_at": self.opened_at,
            "counted": self.counted,
        }


# module-private, no accessor: a public reader would hand a live capability token to
# any in-process caller
_LIVE_SESSIONS: dict[str, SealedToken] = {}
_LIVE_POLICIES: dict[str, SealedPolicy] = {}
_LIVE_LOCK = threading.Lock()


def assert_live(token: Any) -> SealedToken:
    """Refuse anything that is not the exact token object of a live session."""
    if not isinstance(token, SealedToken):
        raise SealedAccessError("sealed access requires an open_sealed token")
    with _LIVE_LOCK:
        current = _LIVE_SESSIONS.get(token.session_id)
    if current is not token:
        raise SealedAccessError("sealed access token does not belong to a live session")
    return token


def _live_policy(token: SealedToken) -> SealedPolicy:
    with _LIVE_LOCK:
        active = _LIVE_POLICIES.get(token.session_id)
    if active is None:  # pragma: no cover - a live token always has a recorded policy
        raise SealedAccessError("sealed access token has no recorded policy")
    return active


def _assert_run_token(token: Any, action: str) -> SealedToken:
    # only a run token carries sealed authority; a reintroduced uncounted class must
    # not silently inherit it
    live = assert_live(token)
    if live.purpose != RUN_PURPOSE:
        raise SealedAccessError(
            f"{action} requires a purpose={RUN_PURPOSE!r} session token, not {live.purpose!r}"
        )
    return live


def authorize_generation(
    master_seed: int,
    captures_dir: str | os.PathLike[str] | None = None,
    token: Any = None,
) -> None:
    """Guard the generation boundary: sealed truth can be recreated from its seed alone.

    captures_dir is optional because the artifact engine has no destination yet; takes
    no registry or root, which would let the caller relocate the gate.
    """
    if not _sealed_seed_gate(master_seed):
        return
    _assert_sealed_access_permitted()
    if token is None:
        raise SealedAccessError(
            "generating a sealed-reservation seed requires an open_sealed token"
        )
    live = _assert_run_token(token, "sealed generation")
    if captures_dir is None:
        return
    target = _real(captures_dir)
    sealed_root = _real(live.sealed_root)
    if not _is_within(target, sealed_root):
        raise SealedAccessError(
            "sealed captures must be written inside the authorized sealed root"
        )


def authorize_read(
    capture_dir: str | os.PathLike[str],
    token: Any = None,
) -> SealedToken | None:
    """Guard the sealed read boundary; a no-op for anything outside every sealed root.

    Every sealed read needs a live run token and is logged; there is no unlogged
    variant and no parameter that moves the boundary.
    """
    return _authorize_private_read((_real(capture_dir),), token, capture_dir)


def _authorize_private_read(
    targets: tuple[Path, ...],
    token: Any,
    capture_dir: str | os.PathLike[str],
) -> SealedToken | None:
    # Plural targets on purpose: the capture dir and the resolved artifact differ when a
    # symlink sits between them, and checking only the first waves a symlinked private/
    # into the sealed store straight through.
    if not any(
        _is_within(target, root) for target in targets for root in _gated_roots(token)
    ):
        return None
    _assert_sealed_access_permitted()
    if token is None:
        raise SealedAccessError("reading a sealed capture requires an open_sealed token")
    live = _assert_run_token(token, "a sealed capture read")
    authorized = _real(live.sealed_root)
    if not all(_is_within(target, authorized) for target in targets):
        raise SealedAccessError("sealed reads must stay inside the authorized sealed root")
    recorded = _live_policy(live)
    with _policy_lock(recorded):
        _append_and_anchor(
            recorded,
            _session_record(
                event="read",
                split_id=live.split_id,
                run_name=live.run_name,
                purpose=live.purpose,
                session_id=live.session_id,
                timestamp=_utc_now(),
                policy=recorded,
                merkle_root=None,
                detail="sealed capture read",
                provenance=_provenance(),
                capture_ref=capture_reference(capture_dir),
            ),
        )
    return live


def capture_reference(capture_dir: str | os.PathLike[str]) -> str:
    """A stable, non-identifying handle: sealed paths must not land in repository files."""
    return sha256(str(_real(capture_dir)).encode("utf-8")).hexdigest()[:16]


# --- the private-artifact chokepoint --------------------------------------------------


def private_dir(capture_dir: str | os.PathLike[str]) -> Path:
    """The protected root of one capture. Naming it is all any other module may do."""
    return Path(capture_dir) / PRIVATE_DIRNAME


def private_artifact_path(capture_dir: str | os.PathLike[str], name: str) -> Path:
    """Resolve <capture>/private/<name>, refusing anything that escapes it. Returns a
    path, never bytes; read_private_artifact is the only reader."""
    if not isinstance(name, str) or not name:
        raise SealedAccessError("a private artifact name must be a non-empty string")
    if os.path.isabs(name) or name.startswith("~"):
        raise SealedAccessError("a private artifact name must be relative to the private root")
    # lexical containment: the artifact need not exist yet, so realpath proves nothing
    normalized = os.path.normpath(name)
    if normalized in (os.curdir, os.pardir) or normalized.startswith(os.pardir + os.sep):
        raise SealedAccessError("a private artifact name must stay inside the private root")
    return private_dir(capture_dir) / normalized


def _object_identity(resolved: Path) -> tuple[int, int] | None:
    """(st_dev, st_ino) of the object at resolved, or None if there is none."""
    try:
        status = os.lstat(resolved)
    except OSError:
        return None
    return (status.st_dev, status.st_ino)


@dataclass(frozen=True)
class _DecidedArtifact:
    """The exact object a gate decision was taken about: a directory pinned by inode
    plus a name inside it, re-verified at open."""

    directory: Path
    name: str
    directory_identity: tuple[int, int] | None
    identity: tuple[int, int] | None
    determined: bool

    @property
    def path(self) -> Path:
        return self.directory / self.name


def _decide_artifact(path: Path) -> _DecidedArtifact:
    # realpath leaves missing components unresolved; deciding on that let a private/
    # created after the check redirect the open into the sealed store. A parent that
    # does not exist yet is recorded determined=False and can never be opened later.
    try:
        directory = Path(os.path.realpath(path.parent, strict=True))
        determined = True
    except OSError:
        directory = _real(path.parent)
        determined = False
    identity = _object_identity(directory) if determined else None
    return _DecidedArtifact(
        directory=directory,
        name=path.name,
        directory_identity=identity,
        identity=_object_identity(directory / path.name) if determined else None,
        determined=determined,
    )


def _open_decided(decided: _DecidedArtifact, flags: int, mode: int = 0o666) -> int:
    # openat against the authorized directory's descriptor, identity-checked against
    # decision time, O_NOFOLLOW on the final component. FileNotFoundError escapes as
    # OSError on purpose: an absent artifact is not an access violation.
    handle = os.open(decided.directory, os.O_RDONLY | _O_DIRECTORY)
    try:
        opened = os.fstat(handle)
        if (
            decided.directory_identity is None
            or (opened.st_dev, opened.st_ino) != decided.directory_identity
        ):
            raise SealedAccessError(
                "the private artifact's directory changed identity between authorization "
                "and open: the path that was checked is not the directory that was opened"
            )
        if _SUPPORTS_DIR_FD:
            return os.open(decided.name, flags | _O_NOFOLLOW, mode, dir_fd=handle)
        return os.open(decided.path, flags | _O_NOFOLLOW, mode)
    finally:
        os.close(handle)


def _read_authorized_object(decided: _DecidedArtifact) -> bytes:
    # re-verifies the file's identity after the open too
    handle = _open_decided(decided, os.O_RDONLY)
    try:
        opened = os.fstat(handle)
        if decided.identity is None or (opened.st_dev, opened.st_ino) != decided.identity:
            raise SealedAccessError(
                "the private artifact changed identity between authorization and open: "
                "the path that was checked is not the object that was opened"
            )
        blocks: list[bytes] = []
        while block := os.read(handle, 1 << 20):
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(handle)


def read_private_artifact(
    capture_dir: str | os.PathLike[str],
    name: str,
    token: Any = None,
) -> bytes:
    """Open one artifact under <capture>/private/. THE chokepoint.

    The only private-truth reader in the package (test_isolation.py enforces
    this). Sealed captures need a live run token; the checked path is the opened path.
    """
    path = private_artifact_path(capture_dir, name)
    decided = _decide_artifact(path)
    _authorize_private_read(_resolved_targets(capture_dir, path, decided), token, capture_dir)
    return _read_authorized_object(decided)


def _resolved_targets(
    capture_dir: str | os.PathLike[str], path: Path, decided: _DecidedArtifact
) -> tuple[Path, ...]:
    # three objects: a symlink can make the capture dir, the realpath'd artifact, and
    # the decided path three different places; gating one leaves the others ungated
    return tuple(dict.fromkeys((_real(capture_dir), _real(path), decided.path)))


def write_private_artifact(
    capture_dir: str | os.PathLike[str],
    name: str,
    payload: bytes,
    token: Any = None,
) -> Path:
    """Create one artifact under <capture>/private/, gated like a read but not logged.

    Authorizes twice: once before anything is created so a refusal plants nothing, and
    again after the parent exists, on the strict resolution the open will use.
    """
    path = private_artifact_path(capture_dir, name)
    _authorize_private_write(
        _resolved_targets(capture_dir, path, _decide_artifact(path)), token
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    decided = _decide_artifact(path)
    _authorize_private_write(_resolved_targets(capture_dir, path, decided), token)
    # 0o666 & ~umask, matching what Path.write_bytes produced before
    handle = _open_decided(decided, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        os.write(handle, payload)
    finally:
        os.close(handle)
    return path


def _authorize_private_write(
    targets: tuple[Path, ...],
    token: Any,
) -> SealedToken | None:
    """The read gate's authority check without its log append."""
    if not any(
        _is_within(target, root) for target in targets for root in _gated_roots(token)
    ):
        return None
    _assert_sealed_access_permitted()
    if token is None:
        raise SealedAccessError("writing a sealed capture requires an open_sealed token")
    live = _assert_run_token(token, "a sealed capture write")
    authorized = _real(live.sealed_root)
    if not all(_is_within(target, authorized) for target in targets):
        raise SealedAccessError("sealed writes must stay inside the authorized sealed root")
    return live


def capture_leaf(
    capture_dir: str | os.PathLike[str],
    master_seed: int,
    token: Any = None,
) -> CaptureLeaf:
    """Fold one stored capture into its Merkle leaf, reading through the chokepoint."""
    iq = (Path(capture_dir) / IQ_ARTIFACT).read_bytes()
    manifest = read_private_artifact(capture_dir, MANIFEST_ARTIFACT, token)
    payload = read_private_artifact(capture_dir, PAYLOAD_ARTIFACT, token)
    return CaptureLeaf(
        master_seed=master_seed,
        iq_sha256=sha256(iq).hexdigest(),
        manifest_sha256=sha256(manifest).hexdigest(),
        payload_sha256=sha256(payload).hexdigest(),
    )


def verification_leaf(capture_dir: str | os.PathLike[str]) -> CaptureLeaf:
    """Return one stored capture's leaf pre-image only: a seed and three digests, what
    leaves.json publishes anyway, never content. Only splits.py may call this."""
    iq = (Path(capture_dir) / IQ_ARTIFACT).read_bytes()
    manifest = private_artifact_path(capture_dir, MANIFEST_ARTIFACT).read_bytes()
    payload = private_artifact_path(capture_dir, PAYLOAD_ARTIFACT).read_bytes()
    recorded = json.loads(private_artifact_path(capture_dir, SEED_ARTIFACT).read_text(encoding="utf-8"))
    master_seed = recorded["master_seed"]
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("master_seed is not an integer")
    leaf = CaptureLeaf(
        master_seed=master_seed,
        iq_sha256=sha256(iq).hexdigest(),
        manifest_sha256=sha256(manifest).hexdigest(),
        payload_sha256=sha256(payload).hexdigest(),
    )
    # fold now so a bad seed surfaces here, not as a MerkleError frames later
    _ = leaf.digest
    return leaf


# --- sealed run records ---------------------------------------------------------------

RUN_RECORDS_SUBDIR = "run-records"
RUN_ARTIFACTS_SUBDIR = "run-artifacts"


def sealed_run_record_path(run_id: str, sealed_root: str | os.PathLike[str]) -> Path:
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise SealedAccessError("a sealed run record id must be a single path component")
    return Path(sealed_root) / RUN_RECORDS_SUBDIR / f"{run_id}.json"


def write_sealed_run_record(run_id: str, document: dict[str, Any], token: Any) -> Path:
    """Keep a sealed run's real identity inside the sealed root, never in the repository."""
    _assert_sealed_access_permitted()
    live = _assert_run_token(token, "writing a sealed run record")
    path = sealed_run_record_path(run_id, live.sealed_root)
    # contain the record directory itself: a run-records symlinked out of the store
    # would otherwise write sealed identity wherever it points
    if not _is_within(_real(path.parent), _real(live.sealed_root)):
        raise SealedAccessError("a sealed run record must live inside the authorized sealed root")
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_durably(path, _canonical_json(document) + b"\n")
    return path


def sealed_run_artifact_dir(run_id: str, token: Any) -> Path:
    """The sealed store's home for one run's recovered content: the decoded frame IS
    sealed truth and must not land in the repository."""
    _assert_sealed_access_permitted()
    live = _assert_run_token(token, "retaining a sealed run artifact")
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise SealedAccessError("a sealed run record id must be a single path component")
    path = Path(live.sealed_root) / RUN_ARTIFACTS_SUBDIR / run_id
    if not _is_within(_real(path), _real(live.sealed_root)):
        raise SealedAccessError(
            "a sealed run artifact must live inside the authorized sealed root"
        )
    path.mkdir(parents=True, exist_ok=True)
    # re-checked after creation: _real on a missing path resolves only what exists
    if not _is_within(_real(path), _real(live.sealed_root)):
        raise SealedAccessError(
            "a sealed run artifact must live inside the authorized sealed root"
        )
    return path


def read_sealed_run_record(run_id: str, token: Any) -> dict[str, Any]:
    """Resolve a sealed run's identity; replay's only route back to the capture."""
    _assert_sealed_access_permitted()
    live = _assert_run_token(token, "reading a sealed run record")
    path = sealed_run_record_path(run_id, live.sealed_root)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedAccessError(f"sealed run record is unreadable: {run_id}") from exc
    if not isinstance(document, dict):
        raise SealedAccessError(f"sealed run record is malformed: {run_id}")
    return document


# --- published root material --------------------------------------------------------


def sealed_leaves_path(split_id: str, policy: SealedPolicy) -> Path:
    # no expanduser: it would reintroduce the $HOME-relocatable boundary
    return Path(policy.sealed_root) / split_id / LEAVES_FILENAME


def recompute_sealed_merkle_root(split_id: str, policy: SealedPolicy) -> str | None:
    """Recompute the Merkle root from the stored leaf list; None if not materialized yet."""
    path = sealed_leaves_path(split_id, policy)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedAccessError(f"sealed leaf list is unreadable: {path}") from exc
    leaves = document.get("leaves") if isinstance(document, dict) else None
    if not isinstance(leaves, list) or not leaves:
        raise SealedAccessError(f"sealed leaf list is malformed: {path}")
    try:
        ordered = order_leaves(
            CaptureLeaf(
                master_seed=int(entry["master_seed"]),
                iq_sha256=entry["iq_sha256"],
                manifest_sha256=entry["manifest_sha256"],
                payload_sha256=entry["payload_sha256"],
            )
            for entry in leaves
        )
        return split_root(ordered, split_id=split_id)
    except (KeyError, TypeError, ValueError, MerkleError) as exc:
        raise SealedAccessError(f"sealed leaf list is malformed: {path}") from exc


def published_merkle_root(split_id: str, root: Path | None = None) -> str | None:
    """Return the root of the published repo commitment, if one exists."""
    path = (root or _project_root()) / COMMITMENTS_RELATIVE / f"{split_id}.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedAccessError(f"published commitment is malformed: {path}") from exc
    value = document.get("merkle_root") if isinstance(document, dict) else None
    return value if isinstance(value, str) else None


def _authoritative_merkle_root(
    split_id: str, policy: SealedPolicy, claimed: str | None
) -> str | None:
    # never record what the caller claims: recompute, then cross-check the commitment
    recomputed = recompute_sealed_merkle_root(split_id, policy)
    published = published_merkle_root(split_id)
    if recomputed is not None and published is not None and recomputed != published:
        raise SealedAccessError(
            "the recomputed Merkle root disagrees with the published commitment for this split"
        )
    authoritative = recomputed if recomputed is not None else published
    if claimed is not None and authoritative is not None and claimed != authoritative:
        raise SealedAccessError(
            "the supplied merkle_root disagrees with the recomputed commitment for this split"
        )
    return authoritative


# --- hash-chained access log --------------------------------------------------------


def _record_digest(record: dict[str, Any]) -> str:
    return sha256(_canonical_json({k: v for k, v in record.items() if k != "record_sha256"})).hexdigest()


def read_log(log_path: str | os.PathLike[str]) -> tuple[dict[str, Any], ...]:
    """Read the append-only access log; an absent log is an empty chain."""
    path = Path(log_path)
    if not path.exists():
        return ()
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SealedAccessError(f"sealed access log line {number} is malformed") from exc
        if not isinstance(record, dict):
            raise SealedAccessError(f"sealed access log line {number} is malformed")
        records.append(record)
    return tuple(records)


def verify_log_chain(records: tuple[dict[str, Any], ...]) -> None:
    """Verify the hash chain. Genesis links to the first record's ``split_id``."""
    for index, record in enumerate(records):
        expected_prev = records[index - 1]["record_sha256"] if index else record.get("split_id")
        if record.get("prev_sha256") != expected_prev:
            raise SealedAccessError(f"sealed access log chain is broken at record {index}")
        if record.get("record_sha256") != _record_digest(record):
            raise SealedAccessError(f"sealed access log record {index} is not self-consistent")


def log_head(log_path: str | os.PathLike[str]) -> str | None:
    """Return the chain head that is anchored on disk and in the run's commit."""
    records = read_log(log_path)
    if not records:
        return None
    verify_log_chain(records)
    return str(records[-1]["record_sha256"])


def log_anchor_head(policy: SealedPolicy) -> str | None:
    """Return the chain head the committed anchor file currently records."""
    path = Path(policy.anchor_path)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedAccessError(f"sealed log anchor is malformed: {path}") from exc
    value = document.get("log_chain_head") if isinstance(document, dict) else None
    return value if isinstance(value, str) else None


def log_anchor_pending(policy: SealedPolicy) -> str | None:
    """The record the anchor named as about-to-append, if an append was interrupted.

    Naming it makes an interrupted append recognisable by name; the chain is unkeyed,
    so tolerating "one append behind" would launder an appended forgery.
    """
    path = Path(policy.anchor_path)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = document.get("pending_sha256") if isinstance(document, dict) else None
    return value if isinstance(value, str) else None


def _anchor_genesis(policy: SealedPolicy) -> dict[str, Any] | None:
    # carried across re-anchors: without it a freshly reset log looks truncated
    path = Path(policy.anchor_path)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = document.get("genesis") if isinstance(document, dict) else None
    return value if isinstance(value, dict) else None


def _write_log_anchor(policy: SealedPolicy, head: str | None, pending: str | None = None) -> None:
    path = Path(policy.anchor_path)
    if log_anchor_head(policy) == head and log_anchor_pending(policy) == pending:
        return
    genesis = _anchor_genesis(policy)
    document: dict[str, Any] = {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "log_chain_head": head,
        **({"pending_sha256": pending} if pending is not None else {}),
        **({"genesis": genesis} if genesis is not None else {}),
        "note": (
            "The head of the sealed access log, re-anchored after every append. The chain "
            "alone is self-referential, so a wholesale log rewrite is only detectable "
            "against a head recorded somewhere the rewrite does not reach. This file is "
            "committed but is deliberately NOT a source_tree_provenance input: it moves on "
            "every sealed read, and the ledger in data/sealed_access.json is the thing "
            "whose edits provenance must expose. 'pending_sha256', when present, names the "
            "record an interrupted append was about to write: the anchor is strictly "
            "authoritative, so a record it never named is a forgery even when the chain "
            "it extends is perfectly well formed."
        ),
    }
    _replace_durably(path, (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _chain_record(
    existing: tuple[dict[str, Any], ...], record: dict[str, Any]
) -> dict[str, Any]:
    """Link one record onto the chain without writing anything."""
    chained = dict(record)
    chained["prev_sha256"] = existing[-1]["record_sha256"] if existing else record["split_id"]
    chained["record_sha256"] = _record_digest(chained)
    return chained


def _write_log(log_path: Path, records: tuple[dict[str, Any], ...], appended: dict[str, Any]) -> None:
    lines = [_canonical_json(item).decode("utf-8") for item in records]
    lines.append(_canonical_json(appended).decode("utf-8"))
    _replace_durably(log_path, ("\n".join(lines) + "\n").encode("utf-8"))


def _append_and_anchor(policy: SealedPolicy, record: dict[str, Any]) -> dict[str, Any]:
    # Agreement is checked on the pre-append state, or a rewritten log gets laundered.
    # Two-phase: name the pending record, write it, clear the mark; a crash leaves a
    # state recognisable by name, an appended forgery matches nothing.
    _assert_log_agrees_with_anchor(policy)
    existing = read_log(policy.log_path)
    chained = _chain_record(existing, record)
    head = str(existing[-1]["record_sha256"]) if existing else None
    _write_log_anchor(policy, head, pending=str(chained["record_sha256"]))
    _write_log(policy.log_path, existing, chained)
    _write_log_anchor(policy, str(chained["record_sha256"]))
    return chained


def _assert_log_agrees_with_anchor(policy: SealedPolicy) -> None:
    """Refuse when the log and the anchored head tell different stories."""
    records = read_log(policy.log_path)
    verify_log_chain(records)
    recorded = log_anchor_head(policy)
    observed = str(records[-1]["record_sha256"]) if records else None
    if recorded == observed:
        return
    if not records:
        raise SealedAccessError(
            "the sealed access log is empty but the anchor records a chain head"
        )
    # an interrupted append is recognised by name; a record the anchor never named is
    # refused however well formed its chain links are
    pending = log_anchor_pending(policy)
    if pending is not None and observed == pending:
        previous = str(records[-2]["record_sha256"]) if len(records) >= 2 else None
        if recorded == previous:
            return
    raise SealedAccessError(
        "the sealed access log disagrees with the anchored chain head"
    )


def _provenance() -> dict[str, Any]:
    # local import: gate -> generator -> sealed would cycle at module scope
    from .gate import _environment, _git_provenance, source_tree_provenance

    project = _running_source_root()
    try:
        git = {"available": True, **_git_provenance(project)}
    except (OSError, RuntimeError):
        # a published tarball has no .git; record the gap rather than hide it
        git = {"available": False, "head_sha": None, "dirty": None}
    return {
        "environment": _environment(project),
        "git": git,
        "source_tree_sha256": source_tree_provenance(project)["source_tree_sha256"],
    }


def _closing_provenance(fallback: dict[str, Any]) -> dict[str, Any]:
    # re-measured at close so a source edit during a session is visible; must not raise
    # inside the closing finally, so degrade to the opening record
    try:
        return _provenance()
    except (OSError, RuntimeError, KeyError):
        return {**fallback, "closing_measurement": "unavailable"}


def _session_record(
    *,
    event: str,
    split_id: str,
    run_name: str,
    purpose: str,
    session_id: str,
    timestamp: str,
    policy: SealedPolicy,
    merkle_root: str | None,
    detail: str | None,
    provenance: dict[str, Any],
    capture_ref: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SEALED_SCHEMA_VERSION,
        "event": event,
        "split_id": split_id,
        "run_name": run_name,
        "purpose": purpose,
        "counted": purpose in COUNTED_PURPOSES,
        "session_id": session_id,
        "timestamp": timestamp,
        "sealed_root": str(_real(policy.sealed_root)),
        "max_opens": policy.max_opens,
        "policy_override": policy.policy_override,
        "merkle_root": merkle_root,
        "capture_ref": capture_ref,
        "detail": detail,
        "provenance": provenance,
    }


def count_sessions(log_path: str | os.PathLike[str], split_id: str) -> int:
    """Count opened records for one split; corroborating evidence only, the budget is
    consumed_authorizations."""
    records = read_log(log_path)
    verify_log_chain(records)
    return sum(
        1
        for record in records
        if record.get("event") == "opened"
        and record.get("split_id") == split_id
        and record.get("purpose") in COUNTED_PURPOSES
    )


def consumed_authorizations(policy: SealedPolicy) -> tuple[RunAuthorization, ...]:
    """The authoritative budget: entries the repository policy records as consumed."""
    document = _read_policy_document(policy.ledger_path)
    entries = _parse_authorizations(document.get("authorized_runs"), policy.ledger_path)
    return tuple(entry for entry in entries if entry.consumed)


def _lock_path(policy: SealedPolicy) -> Path:
    return policy.ledger_path.with_name(policy.ledger_path.name + ".lock")


@contextmanager
def _policy_lock(policy: SealedPolicy) -> Iterator[None]:
    # lock lives on a sibling, never the ledger itself: os.replace swaps the ledger's
    # inode and a lock on a replaced inode excludes nobody
    if not policy.ledger_path.exists():
        raise SealedAccessError(
            f"sealed access policy vanished before it could be locked: {policy.ledger_path}"
        )
    try:
        handle = os.open(_lock_path(policy), os.O_RDONLY | os.O_CREAT, 0o644)
    except OSError as exc:
        raise SealedAccessError(
            f"sealed access lock is unusable: {_lock_path(policy)}"
        ) from exc
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def _consume_authorization(
    policy: SealedPolicy,
    document: dict[str, Any],
    *,
    run_name: str,
    split_id: str,
    session_id: str,
    opened_at: str,
) -> None:
    # budget is global over consumed authorizations, not keyed on the caller-supplied
    # split_id: a fresh random id would otherwise mint unlimited sessions
    raw = document.get("authorized_runs")
    entries = _parse_authorizations(raw, policy.ledger_path)
    match = next((entry for entry in entries if entry.run_name == run_name), None)
    if match is None:
        raise SealedAccessError(
            f"run_name {run_name!r} is not in the sealed policy's authorized_runs"
        )
    if match.consumed:
        if match.split_id != split_id:
            raise SealedAccessError(
                f"authorization {run_name!r} is bound to a different split_id"
            )
        raise SealedAccessError(
            f"authorization {run_name!r} has already been consumed"
        )
    consumed = sum(1 for entry in entries if entry.consumed)
    if consumed >= policy.max_opens:
        raise SealedAccessError(
            f"sealed access has consumed all {policy.max_opens} authorized sessions"
        )
    for item in raw:
        if item.get("run_name") == run_name:
            item["opened_at"] = opened_at
            item["split_id"] = split_id
            item["session_id"] = session_id
            item["closed_at"] = None
    document["authorized_runs"] = raw
    # durable before the token is yielded: a crash consumes the session
    _write_policy_document(policy.ledger_path, document)


def _record_close(policy: SealedPolicy, *, run_name: str, closed_at: str) -> None:
    # caller holds the policy lock
    document = _read_policy_document(policy.ledger_path)
    raw = document.get("authorized_runs")
    _parse_authorizations(raw, policy.ledger_path)
    for item in raw:
        if item.get("run_name") == run_name and item.get("opened_at") is not None:
            item["closed_at"] = closed_at
    document["authorized_runs"] = raw
    _write_policy_document(policy.ledger_path, document)


def _ledger_authorization(policy: SealedPolicy, run_name: str) -> RunAuthorization | None:
    document = _read_policy_document(policy.ledger_path)
    entries = _parse_authorizations(document.get("authorized_runs"), policy.ledger_path)
    return next((entry for entry in entries if entry.run_name == run_name), None)


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _session_last_seen(policy: SealedPolicy, entry: RunAuthorization) -> str | None:
    # the log is consulted because re-entry deliberately does not rewrite the ledger
    stamps = [entry.opened_at] if entry.opened_at else []
    for record in read_log(policy.log_path):
        if record.get("session_id") == entry.session_id and isinstance(record.get("timestamp"), str):
            stamps.append(record["timestamp"])
    # fixed-width UTC ISO-8601 sorts lexicographically in chronological order
    return max(stamps) if stamps else None


def _assert_lease_is_fresh(policy: SealedPolicy, entry: RunAuthorization) -> None:
    # an abandoned open session must not stay re-enterable forever; recovery is
    # close_sealed
    last_seen = _session_last_seen(policy, entry)
    moment = _parse_timestamp(last_seen) if last_seen else None
    if moment is None:
        return
    idle = (datetime.now(timezone.utc) - moment).total_seconds()
    if idle <= SESSION_LEASE_SECONDS:
        return
    raise SealedAccessError(
        f"authorization {entry.run_name!r} was opened but never closed and has been idle "
        f"for {int(idle)}s, past its {SESSION_LEASE_SECONDS}s re-entry lease. Recover an "
        "abandoned campaign with close_sealed(split_id, run_name); the authorization stays "
        "consumed, because a crash spends a session by design"
    )


def _find_live(split_id: str, run_name: str, purpose: str) -> SealedToken | None:
    with _LIVE_LOCK:
        return next(
            (
                token
                for token in _LIVE_SESSIONS.values()
                if token.split_id == split_id
                and token.run_name == run_name
                and token.purpose == purpose
            ),
            None,
        )


@contextmanager
def open_sealed(
    split_id: str,
    run_name: str,
    *,
    merkle_root: str | None = None,
) -> Iterator[SealedToken]:
    """Open (or re-enter) a sealed session and yield the token that unlocks sealed work.

    Every session is counted; the budget comes from process state alone (no policy= or
    root=). Re-entry works across processes via the ledger; only the minting process
    closes on exit, and a crash consumes the authorization but not the campaign.
    """
    if not isinstance(split_id, str) or len(split_id) != 64 or any(
        character not in "0123456789abcdef" for character in split_id
    ):
        raise SealedAccessError("split_id must be a 64-character lowercase hex digest")
    if not isinstance(run_name, str) or not run_name or run_name != run_name.strip():
        raise SealedAccessError("run_name must be a non-empty trimmed string")
    purpose = RUN_PURPOSE
    active = load_policy()
    _assert_sealed_access_permitted()

    minted: SealedToken | None = None
    adopted: SealedToken | None = None
    with _policy_lock(active):
        # provenance and root recomputation run inside the lock (see
        # test_g4_the_live_session_lookup_happens_inside_the_lock)
        entry_provenance = _provenance()
        # never record what the caller claims the root is
        committed_root = _authoritative_merkle_root(split_id, active, merkle_root)
        resumed = _find_live(split_id, run_name, purpose)
        document = _read_policy_document(active.ledger_path)
        _assert_log_agrees_with_anchor(active)
        if resumed is None:
            recorded = next(
                (
                    entry
                    for entry in _parse_authorizations(
                        document.get("authorized_runs"), active.ledger_path
                    )
                    if entry.run_name == run_name
                ),
                None,
            )
            if recorded is not None and recorded.live:
                # cross-process re-entry: nothing consumed, no fresh session id
                if recorded.split_id != split_id:
                    raise SealedAccessError(
                        f"authorization {run_name!r} is bound to a different split_id"
                    )
                _assert_lease_is_fresh(active, recorded)
                assert recorded.session_id is not None and recorded.opened_at is not None
                adopted = SealedToken(
                    session_id=recorded.session_id,
                    split_id=split_id,
                    run_name=run_name,
                    purpose=purpose,
                    sealed_root=_real(active.sealed_root),
                    opened_at=recorded.opened_at,
                    counted=True,
                )
                with _LIVE_LOCK:
                    _LIVE_SESSIONS[adopted.session_id] = adopted
                    _LIVE_POLICIES[adopted.session_id] = active
                try:
                    _append_and_anchor(
                        active,
                        _session_record(
                            event="reentered",
                            split_id=split_id,
                            run_name=run_name,
                            purpose=purpose,
                            session_id=adopted.session_id,
                            timestamp=_utc_now(),
                            policy=active,
                            merkle_root=committed_root,
                            detail="resumed an open session from another process",
                            provenance=entry_provenance,
                        ),
                    )
                except BaseException:
                    with _LIVE_LOCK:
                        _LIVE_SESSIONS.pop(adopted.session_id, None)
                        _LIVE_POLICIES.pop(adopted.session_id, None)
                    raise
        if resumed is None and adopted is None:
            session_id = secrets.token_hex(16)
            opened_at = _utc_now()
            _consume_authorization(
                active,
                document,
                run_name=run_name,
                split_id=split_id,
                session_id=session_id,
                opened_at=opened_at,
            )
            minted = SealedToken(
                session_id=session_id,
                split_id=split_id,
                run_name=run_name,
                purpose=purpose,
                sealed_root=_real(active.sealed_root),
                opened_at=opened_at,
                counted=True,
            )
            with _LIVE_LOCK:
                _LIVE_SESSIONS[session_id] = minted
                _LIVE_POLICIES[session_id] = active
            try:
                _append_and_anchor(
                    active,
                    _session_record(
                        event="opened",
                        split_id=split_id,
                        run_name=run_name,
                        purpose=purpose,
                        session_id=session_id,
                        timestamp=opened_at,
                        policy=active,
                        merkle_root=committed_root,
                        detail=None,
                        provenance=entry_provenance,
                    ),
                )
            except BaseException:
                with _LIVE_LOCK:
                    _LIVE_SESSIONS.pop(session_id, None)
                    _LIVE_POLICIES.pop(session_id, None)
                raise
        elif resumed is not None:
            _append_and_anchor(
                active,
                _session_record(
                    event="reentered",
                    split_id=split_id,
                    run_name=run_name,
                    purpose=purpose,
                    session_id=resumed.session_id,
                    timestamp=_utc_now(),
                    policy=active,
                    merkle_root=committed_root,
                    detail="resumed inside a live session",
                    provenance=entry_provenance,
                ),
            )

    token = minted or adopted or resumed
    assert token is not None
    outcome = "completed"
    try:
        yield token
    except BaseException as exc:
        outcome = f"failed: {type(exc).__name__}"
        raise
    finally:
        # only the minting frame closes; an adopting frame drops its view and leaves the
        # session open, a nested in-process frame does neither
        if minted is not None:
            closing_event = "closed"
        else:
            closing_event = "reexited"
        if minted is not None or adopted is not None:
            with _LIVE_LOCK:
                _LIVE_SESSIONS.pop(token.session_id, None)
                _LIVE_POLICIES.pop(token.session_id, None)
        with _policy_lock(active):
            closed_at = _utc_now()
            if minted is not None:
                _record_close(active, run_name=run_name, closed_at=closed_at)
            _append_and_anchor(
                active,
                _session_record(
                    event=closing_event,
                    split_id=split_id,
                    run_name=run_name,
                    purpose=purpose,
                    session_id=token.session_id,
                    timestamp=closed_at,
                    policy=active,
                    merkle_root=committed_root,
                    detail=outcome,
                    provenance=_closing_provenance(entry_provenance),
                ),
            )


def close_sealed(split_id: str, run_name: str) -> bool:
    """Explicitly end a session that outlived the process which opened it.

    The documented recovery path; deliberately still works when max_opens is 0 or the
    lease expired. Idempotent (returns False when nothing was open), and refuses a
    split_id the authorization is not bound to, since closing is irreversible.
    """
    active = load_policy()
    with _policy_lock(active):
        recorded = _ledger_authorization(active, run_name)
        if recorded is None or not recorded.live:
            return False
        if recorded.split_id != split_id:
            raise SealedAccessError(
                f"authorization {run_name!r} is bound to a different split_id"
            )
        assert recorded.session_id is not None
        with _LIVE_LOCK:
            _LIVE_SESSIONS.pop(recorded.session_id, None)
            _LIVE_POLICIES.pop(recorded.session_id, None)
        closed_at = _utc_now()
        _record_close(active, run_name=run_name, closed_at=closed_at)
        _append_and_anchor(
            active,
            _session_record(
                event="closed",
                split_id=split_id,
                run_name=run_name,
                purpose=RUN_PURPOSE,
                session_id=recorded.session_id,
                timestamp=closed_at,
                policy=active,
                merkle_root=None,
                detail="closed explicitly",
                provenance=_provenance(),
            ),
        )
    return True
