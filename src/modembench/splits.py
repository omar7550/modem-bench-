"""Split specifications, seed derivation, and the repo-side split commitment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import secrets
import tomllib
from typing import Any, Iterable, Sequence

from .generator import (
    GENERATOR_VERSION,
    SCHEMA_VERSION as CAPTURE_SCHEMA_VERSION,
    BurstPlacement,
    canonical_json,
    generate_capture,
)
from .impairments import ImpairmentConfig, ImpairmentRanges
from .merkle import CONSTRUCTION, CaptureLeaf, MerkleError, order_leaves, split_root
from .sealed import (
    COMMITMENTS_RELATIVE,
    LEAVES_FILENAME,
    SealedAccessError,
    SealedToken,
    capture_leaf,
    capture_reference,
    configured_sealed_root,
    is_sealed_seed,
    load_policy,
    load_seed_registry,
    plain_seed_max,
    read_sealed_salt,
    salt_sha256,
    verification_leaf,
)

SPLIT_SCHEMA_VERSION = "1.0"
COMMITMENT_SCHEMA_VERSION = "1.0"
LEAVES_SCHEMA_VERSION = "1.0"
DEV_SPLIT_NAME = "dev-v1"
DEV_SPLIT_SIZE = 40
DEV_SPLIT_PROFILE = "impaired"
SEALED_SPLIT_NAME = "sealed-v1"
SEALED_SPLIT_SIZE = 60
SEALED_SPLIT_PROFILE = "impaired"
BLOCK_DERIVATION = "registry-block"
HMAC_DERIVATION = "hmac-sha256"
DERIVATIONS = (BLOCK_DERIVATION, HMAC_DERIVATION)
KINDS = ("dev", "sealed")
VERDICT_VERIFIED = "verified"
VERDICT_MISMATCH = "mismatch"
VERDICT_MISSING = "missing"
VERDICT_ENVIRONMENT_CHANGED = "environment_changed"


class SplitError(RuntimeError):
    """A split definition, generation, or commitment invariant was violated."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def declared_binding(root: Path | None = None) -> tuple[str, tuple[str, ...]]:
    """Return the declared (version-controlled) environment binding for a split."""
    project = tomllib.loads(((root or _project_root()) / "pyproject.toml").read_text(encoding="utf-8"))
    return project["project"]["requires-python"], tuple(project["project"]["dependencies"])


@dataclass(frozen=True)
class SplitSpec:
    """Everything that determines a split. Its canonical-JSON SHA-256 is the split id."""

    name: str
    kind: str
    size: int
    profile: str
    seed_derivation: str
    seed_block_start: int | None
    ranges_hash: str
    generator_version: str
    schema_version: str
    capture_schema_version: str
    python_requires: str
    declared_pins: tuple[str, ...]
    #: None is the frozen no-burst geometry. Elided from to_dict at None: the split id is
    #: the hash of that dict, so serializing the default would rename every committed split.
    burst_hash: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise SplitError(f"unknown split kind: {self.kind!r}")
        if self.seed_derivation not in DERIVATIONS:
            raise SplitError(f"unknown seed derivation: {self.seed_derivation!r}")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise SplitError("split size must be a positive integer")
        if (self.seed_derivation == BLOCK_DERIVATION) != (self.seed_block_start is not None):
            raise SplitError("seed_block_start is required exactly for registry-block splits")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "capture_schema_version": self.capture_schema_version,
            "declared_pins": list(self.declared_pins),
            "generator_version": self.generator_version,
            "kind": self.kind,
            "name": self.name,
            "profile": self.profile,
            "python_requires": self.python_requires,
            "ranges_hash": self.ranges_hash,
            "schema_version": self.schema_version,
            "seed_block_start": self.seed_block_start,
            "seed_derivation": self.seed_derivation,
            "size": self.size,
        }
        if self.burst_hash is not None:
            payload["burst_hash"] = self.burst_hash
        return payload

    @property
    def split_id(self) -> str:
        return sha256(canonical_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SplitSpec":
        try:
            return cls(
                name=value["name"],
                kind=value["kind"],
                size=value["size"],
                profile=value["profile"],
                seed_derivation=value["seed_derivation"],
                seed_block_start=value["seed_block_start"],
                ranges_hash=value["ranges_hash"],
                generator_version=value["generator_version"],
                schema_version=value["schema_version"],
                capture_schema_version=value["capture_schema_version"],
                python_requires=value["python_requires"],
                declared_pins=tuple(value["declared_pins"]),
                burst_hash=value.get("burst_hash"),
            )
        except (KeyError, TypeError) as exc:
            raise SplitError("split specification is malformed") from exc


def reservation(name: str, registry: dict[str, Any] | None = None, *, root: Path | None = None) -> dict[str, Any]:
    """Return the named seed reservation, refusing when it is absent."""
    document = registry if registry is not None else load_seed_registry(root)
    match = next((entry for entry in document["reservations"] if entry.get("name") == name), None)
    if match is None:
        raise SplitError(f"seed registry has no reservation named {name!r}")
    return match


def dev_split_spec(*, root: Path | None = None, ranges: ImpairmentRanges | None = None) -> SplitSpec:
    """The 40-signal development split over the reserved [20000, 20999] block."""
    block = reservation(DEV_SPLIT_NAME, root=root)
    if block.get("sealed") is not False:
        raise SplitError("the development reservation must not be sealed")
    python_requires, pins = declared_binding(root)
    return SplitSpec(
        name=DEV_SPLIT_NAME,
        kind="dev",
        size=DEV_SPLIT_SIZE,
        profile=DEV_SPLIT_PROFILE,
        seed_derivation=BLOCK_DERIVATION,
        seed_block_start=int(block["start"]),
        ranges_hash=(ranges or ImpairmentRanges()).ranges_hash,
        generator_version=GENERATOR_VERSION,
        schema_version=SPLIT_SCHEMA_VERSION,
        capture_schema_version=CAPTURE_SCHEMA_VERSION,
        python_requires=python_requires,
        declared_pins=pins,
    )


def sealed_split_spec(*, root: Path | None = None, ranges: ImpairmentRanges | None = None) -> SplitSpec:
    """The 60-signal sealed split. Its seeds are derived, never published."""
    block = reservation(SEALED_SPLIT_NAME, root=root)
    if block.get("sealed") is not True:
        raise SplitError("the sealed reservation must be marked sealed")
    if block.get("derivation") != HMAC_DERIVATION:
        raise SplitError("the sealed reservation must declare hmac-sha256 derivation")
    if "start" in block or "stop" in block:
        raise SplitError("the sealed reservation must not publish a usable seed range")
    python_requires, pins = declared_binding(root)
    return SplitSpec(
        name=SEALED_SPLIT_NAME,
        kind="sealed",
        size=SEALED_SPLIT_SIZE,
        profile=SEALED_SPLIT_PROFILE,
        seed_derivation=HMAC_DERIVATION,
        seed_block_start=None,
        ranges_hash=(ranges or ImpairmentRanges()).ranges_hash,
        generator_version=GENERATOR_VERSION,
        schema_version=SPLIT_SCHEMA_VERSION,
        capture_schema_version=CAPTURE_SCHEMA_VERSION,
        python_requires=python_requires,
        declared_pins=pins,
    )


# --- seed derivation ----------------------------------------------------------------


def derive_seeds(
    salt: bytes, split_name: str, size: int, *, registry: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[int, ...]:
    """seed_i = HMAC-SHA256(salt, f"{split_name}:{i}")[:8]; each must land above plain_seed_max."""
    if len(salt) < 16:
        raise SplitError("sealed salt must be at least 16 bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SplitError("split size must be a positive integer")
    boundary = plain_seed_max(registry, root=root)
    seeds = tuple(
        int.from_bytes(
            hmac.new(salt, f"{split_name}:{index}".encode("utf-8"), sha256).digest()[:8], "big"
        )
        for index in range(size)
    )
    if len(set(seeds)) != size:
        raise SplitError("derived sealed seeds collided")
    if any(seed <= boundary for seed in seeds):
        raise SplitError("a derived sealed seed fell inside the plain seed space")
    return seeds


def block_seeds(start: int, size: int, *, block: dict[str, Any] | None = None) -> tuple[int, ...]:
    """Return the contiguous plain seeds for a block-reserved split."""
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise SplitError("seed block start must be a non-negative integer")
    seeds = tuple(range(start, start + size))
    if block is not None and (start < int(block["start"]) or seeds[-1] > int(block["stop"])):
        raise SplitError("split seeds fall outside the reserved registry block")
    return seeds


def split_seeds(
    spec: SplitSpec,
    *,
    salt: bytes | None = None,
    registry: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[int, ...]:
    """Resolve a spec's seeds, reading the registry block or deriving from the salt."""
    if spec.seed_derivation == BLOCK_DERIVATION:
        block = None
        try:
            block = reservation(spec.name, registry, root=root)
        except SplitError:
            block = None
        assert spec.seed_block_start is not None
        return block_seeds(spec.seed_block_start, spec.size, block=block)
    if salt is None:
        raise SplitError("a sealed split needs its salt to resolve seeds")
    return derive_seeds(salt, spec.name, spec.size, registry=registry, root=root)


# --- generation ---------------------------------------------------------------------


@dataclass(frozen=True)
class SplitBuild:
    """A materialized split: its captures, its leaves, and its root."""

    spec: SplitSpec
    split_root_dir: Path
    seeds: tuple[int, ...]
    capture_ids: tuple[str, ...]
    leaves: tuple[CaptureLeaf, ...]
    merkle_root: str
    salt_sha256: str | None

    @property
    def split_id(self) -> str:
        return self.spec.split_id


def _built_leaf(capture_dir: Path, master_seed: int, token: SealedToken | None) -> CaptureLeaf:
    """Fold a just-generated capture into its leaf through the sealed chokepoint."""
    try:
        return capture_leaf(capture_dir, master_seed, token)
    except OSError as exc:
        raise SplitError(f"capture component is unreadable: {capture_dir}") from exc


def build_split(
    spec: SplitSpec,
    *,
    captures_root: str | os.PathLike[str],
    ranges: ImpairmentRanges | None = None,
    burst: "BurstPlacement | None" = None,
    salt: bytes | None = None,
    token: SealedToken | None = None,
    root: Path | None = None,
) -> SplitBuild:
    """Generate (or byte-verify and reuse) every capture in the split and commit to it."""
    applied_ranges = ranges or ImpairmentRanges()
    if applied_ranges.ranges_hash != spec.ranges_hash:
        raise SplitError("impairment ranges do not match the split specification")
    if (burst.ranges_hash if burst is not None else None) != spec.burst_hash:
        raise SplitError("burst placement does not match the split specification")
    seeds = split_seeds(spec, salt=salt, root=root)
    if len(seeds) != spec.size:
        raise SplitError("resolved seed count does not match the split size")
    if len(set(seeds)) != spec.size:
        raise SplitError("split seeds must be unique")
    directory = Path(captures_root)
    directory.mkdir(parents=True, exist_ok=True)
    config = ImpairmentConfig.from_profile(spec.profile)

    capture_ids: list[str] = []
    leaves: list[CaptureLeaf] = []
    for master_seed in sorted(seeds):
        capture = generate_capture(
            master_seed,
            directory,
            config=config,
            ranges=applied_ranges,
            burst=burst,
            sealed_token=token,
        )
        capture_ids.append(capture.capture_id)
        leaves.append(_built_leaf(capture.capture_dir, master_seed, token))
    # capture_id is a 48-bit truncation; a collision must be a hard error
    if len(set(capture_ids)) != spec.size:
        raise SplitError("capture_id collision inside the split")
    ordered = order_leaves(leaves)
    return SplitBuild(
        spec=spec,
        split_root_dir=directory,
        seeds=tuple(sorted(seeds)),
        capture_ids=tuple(capture_ids),
        leaves=ordered,
        merkle_root=split_root(ordered, split_id=spec.split_id),
        salt_sha256=salt_sha256(salt) if salt is not None else None,
    )


# --- commitment ---------------------------------------------------------------------


def _write_json_once(path: Path, value: dict[str, Any]) -> Path:
    """Publish an immutable artifact; differing existing content is a hard error."""
    content = canonical_json(value) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise SplitError(f"refusing to overwrite a differing published artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _publish_commitment(path: Path, document: dict[str, Any]) -> Path:
    """Publish the commitment once; differing root material is a hard error."""
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SplitError(f"published commitment is malformed: {path}") from exc
        if not isinstance(existing, dict) or any(
            existing.get(field) != document.get(field) for field in COMMITMENT_MATERIAL
        ):
            raise SplitError(f"refusing to overwrite a differing published artifact: {path}")
        return path
    return _write_json_once(path, document)


def commitment_path(split_id: str, *, commitments_dir: str | os.PathLike[str] | None = None,
                    root: Path | None = None) -> Path:
    directory = (
        Path(commitments_dir)
        if commitments_dir is not None
        else (root or _project_root()) / COMMITMENTS_RELATIVE
    )
    return directory / f"{split_id}.json"


# The fields that are the commitment; the rest records the moment of first publication.
COMMITMENT_MATERIAL = (
    "capture_count",
    "merkle_construction",
    "merkle_root",
    "salt_sha256",
    "schema_version",
    "split_id",
    "split_spec",
)


def commitment_document(build: SplitBuild, *, root: Path | None = None) -> dict[str, Any]:
    """Root material only; per-capture digests would be an off-budget correctness oracle."""
    from .gate import _environment, source_tree_provenance

    project = root or _project_root()
    return {
        "schema_version": COMMITMENT_SCHEMA_VERSION,
        "split_id": build.split_id,
        "split_spec": build.spec.to_dict(),
        "capture_count": len(build.leaves),
        "merkle_root": build.merkle_root,
        "merkle_construction": dict(CONSTRUCTION),
        "salt_sha256": build.salt_sha256,
        "created_at": _utc_now(),
        "environment": _environment(project),
        "source_tree_sha256": source_tree_provenance(project)["source_tree_sha256"],
        "leaves_location": f"<split root>/{LEAVES_FILENAME}",
        "disclosure": (
            "The leaf list (per-capture component digests and seeds) lives in the split "
            "root and is released at the final disclosure; it is deliberately absent here."
        ),
        "anchoring": (
            "This file is committed in its own commit naming the root. Git author and "
            "committer dates are arbitrary strings and history is rewritable, so dated "
            "history is corroborating evidence only, never a third-party timestamp."
        ),
    }


def leaves_document(build: SplitBuild) -> dict[str, Any]:
    return {
        "schema_version": LEAVES_SCHEMA_VERSION,
        "split_id": build.split_id,
        "merkle_root": build.merkle_root,
        "merkle_construction": dict(CONSTRUCTION),
        "leaves": [
            {**leaf.to_dict(), "index": index, "leaf_sha256": leaf.digest.hex()}
            for index, leaf in enumerate(build.leaves)
        ],
    }


def publish_split(
    build: SplitBuild,
    *,
    commitments_dir: str | os.PathLike[str] | None = None,
    leaves_dir: str | os.PathLike[str] | None = None,
    root: Path | None = None,
    registry: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the repo commitment and the split-root leaf list, both immutably."""
    if build.spec.kind == "sealed":
        block = reservation(build.spec.name, registry, root=root)
        if block.get("salt_sha256") != build.salt_sha256:
            raise SplitError(
                "the sealed reservation's salt_sha256 does not match the split's salt"
            )
    # Resolved before anything is written, so a refused destination publishes nothing.
    leaves_root = _sealed_contained_leaves_root(build, leaves_dir, root=root)
    document = commitment_document(build, root=root)
    forbidden = {leaf.payload_sha256 for leaf in build.leaves}
    forbidden.update(leaf.iq_sha256 for leaf in build.leaves)
    forbidden.update(leaf.manifest_sha256 for leaf in build.leaves)
    if any(value in forbidden for value in _flatten_strings(document)):
        raise SplitError("the repo commitment must not carry per-capture component digests")
    commitment = _publish_commitment(
        commitment_path(build.split_id, commitments_dir=commitments_dir, root=root), document
    )
    leaves = _write_json_once(leaves_root / LEAVES_FILENAME, leaves_document(build))
    return commitment, leaves


def carries_sealed_seeds(build: "SplitBuild", *, root: Path | None = None) -> bool:
    """Read sealedness off the actual seeds; a declared kind="sealed" tightens, never loosens."""
    if build.spec.kind == "sealed":
        return True
    registry = load_seed_registry(root)
    return any(is_sealed_seed(seed, registry) for seed in build.seeds)


def _sealed_contained_leaves_root(
    build: SplitBuild, leaves_dir: str | os.PathLike[str] | None, *, root: Path | None
) -> Path:
    """A sealed split's leaf list (an off-budget oracle) may only land inside the sealed root."""
    leaves_root = Path(leaves_dir) if leaves_dir is not None else build.split_root_dir
    if not carries_sealed_seeds(build, root=root):
        return leaves_root
    sealed_root = configured_sealed_root(root)
    for candidate, label in ((leaves_root, "leaf list"), (build.split_root_dir, "split root")):
        resolved = Path(os.path.realpath(os.fspath(candidate)))
        if resolved != sealed_root and sealed_root not in resolved.parents:
            raise SplitError(
                f"a sealed split's {label} must live inside the configured sealed root: {resolved}"
            )
    return leaves_root


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_strings(item)


def load_commitment(
    split_id: str,
    *,
    commitments_dir: str | os.PathLike[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    path = commitment_path(split_id, commitments_dir=commitments_dir, root=root)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitError(f"split commitment is malformed: {path}") from exc
    if not isinstance(document, dict):
        raise SplitError(f"split commitment is malformed: {path}")
    return document


# --- verification -------------------------------------------------------------------


def _verdict(verdict: str, split_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "split_id": split_id,
        "verdict": verdict,
        "capture_count": None,
        "expected_merkle_root": None,
        "recomputed_merkle_root": None,
        "environment_changed": [],
        "detail": None,
        **extra,
    }


def _environment_delta(recorded: Any, observed: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(recorded, dict):
        return ("environment record is malformed",)
    keys = sorted(set(recorded) | set(observed))
    return tuple(key for key in keys if recorded.get(key) != observed.get(key))


def _is_capture_id(name: str) -> bool:
    return len(name) == 12 and all(character in "0123456789abcdef" for character in name)


def _stored_leaves(directory: Path) -> tuple[list[CaptureLeaf], tuple[str, ...], int]:
    """Hash every stored capture into its leaf via verification_leaf; needs no salt.

    A present-but-tampered capture is returned as an unfoldable reference rather than
    raised; an incomplete one counts as missing.
    """
    leaves: list[CaptureLeaf] = []
    unfoldable: list[str] = []
    incomplete = 0
    if not directory.is_dir():
        return leaves, (), 0
    for capture_dir in sorted(directory.iterdir()):
        if not capture_dir.is_dir() or not _is_capture_id(capture_dir.name):
            continue
        try:
            leaves.append(verification_leaf(capture_dir))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            # before ValueError: JSONDecodeError is a ValueError, and truncated means incomplete
            incomplete += 1
        except (SealedAccessError, MerkleError, ValueError):
            unfoldable.append(capture_reference(capture_dir))
    return leaves, tuple(unfoldable), incomplete


def published_leaf_digests(captures_root: str | os.PathLike[str]) -> set[str] | None:
    """Leaf digests from the split root's leaves.json; absent on a fresh clone, so optional."""
    path = Path(captures_root) / LEAVES_FILENAME
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = document["leaves"]
        return {
            CaptureLeaf(
                master_seed=int(entry["master_seed"]),
                iq_sha256=entry["iq_sha256"],
                manifest_sha256=entry["manifest_sha256"],
                payload_sha256=entry["payload_sha256"],
            ).digest.hex()
            for entry in entries
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, MerkleError):
        # no path in the message: this reaches verify-split's aggregate-only output
        raise SplitError("the split root's leaf list is malformed")


def verify_split(
    split_id: str,
    *,
    captures_root: str | os.PathLike[str],
    commitments_dir: str | os.PathLike[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compare the stored artifacts of a split against its published commitment.

    Verification is not privileged and takes no session: it re-hashes what is on disk and
    needs no salt, spends nothing, writes nothing. Aggregate verdicts and roots only.
    """
    from .gate import _environment

    project = root or _project_root()
    document = load_commitment(split_id, commitments_dir=commitments_dir, root=project)
    return _verify_stored(
        split_id,
        document,
        captures_root=captures_root,
        environment=_environment(project),
    )


def _verify_stored(
    split_id: str,
    document: dict[str, Any] | None,
    *,
    captures_root: str | os.PathLike[str],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Hashing is environment-independent, so an environment delta is reported beside a real
    verdict; environment_changed is the verdict only when the pinned Merkle construction is
    one this build cannot recompute."""
    if document is None:
        return _verdict(VERDICT_MISSING, split_id, detail="no commitment is published for this split id")
    try:
        spec = SplitSpec.from_dict(document.get("split_spec", {}))
    except SplitError as exc:
        return _verdict(VERDICT_MISMATCH, split_id, detail=str(exc))
    if spec.split_id != split_id:
        return _verdict(
            VERDICT_MISMATCH, split_id, detail="the committed specification does not hash to this split id"
        )
    expected_root = document.get("merkle_root")
    count = document.get("capture_count")
    changed = list(_environment_delta(document.get("environment"), environment))

    committed_construction = document.get("merkle_construction")
    if committed_construction is not None and committed_construction != dict(CONSTRUCTION):
        return _verdict(
            VERDICT_ENVIRONMENT_CHANGED,
            split_id,
            capture_count=count if isinstance(count, int) else None,
            expected_merkle_root=expected_root,
            environment_changed=changed,
            detail=(
                "the commitment pins a Merkle construction this build does not implement, "
                "so its root cannot be recomputed here"
            ),
        )

    def verdict(name: str, **extra: Any) -> dict[str, Any]:
        return _verdict(name, split_id, environment_changed=changed, **extra)

    if not isinstance(expected_root, str) or not isinstance(count, int) or count != spec.size:
        return verdict(VERDICT_MISMATCH, detail="the commitment header is malformed")

    directory = Path(captures_root)
    leaves, unfoldable, incomplete = _stored_leaves(directory)
    if unfoldable:
        # named by opaque capture refs: verify-split never prints an id, path, or seed
        return verdict(
            VERDICT_MISMATCH,
            capture_count=count,
            expected_merkle_root=expected_root,
            detail=(
                f"{len(unfoldable)} of {count} stored captures could not be folded into the "
                f"pinned construction (capture refs {', '.join(sorted(unfoldable))})"
            ),
        )
    absent = count - len(leaves)
    if absent > 0:
        return verdict(
            VERDICT_MISSING,
            capture_count=count,
            expected_merkle_root=expected_root,
            detail=f"{absent} of {count} captures are absent from the split root"
            + (f" ({incomplete} incomplete)" if incomplete else ""),
        )
    if absent < 0:
        return verdict(
            VERDICT_MISMATCH,
            capture_count=count,
            expected_merkle_root=expected_root,
            detail=f"the split root holds {len(leaves)} captures but the commitment binds {count}",
        )
    try:
        published = published_leaf_digests(directory)
    except SplitError as exc:
        return verdict(
            VERDICT_MISMATCH, capture_count=count, expected_merkle_root=expected_root,
            detail=str(exc),
        )
    if published is not None:
        diverging = sum(1 for leaf in leaves if leaf.digest.hex() not in published)
        if diverging:
            return verdict(
                VERDICT_MISMATCH,
                capture_count=count,
                expected_merkle_root=expected_root,
                detail=f"{diverging} of {count} stored captures diverge from the published leaf list",
            )
    try:
        ordered = order_leaves(leaves)
        recomputed = split_root(ordered, split_id=split_id)
    except MerkleError as exc:
        return verdict(
            VERDICT_MISMATCH, capture_count=count, expected_merkle_root=expected_root,
            detail=str(exc),
        )
    if recomputed != expected_root:
        return verdict(
            VERDICT_MISMATCH,
            capture_count=count,
            expected_merkle_root=expected_root,
            recomputed_merkle_root=recomputed,
            detail="the recomputed Merkle root does not match the published commitment",
        )
    return verdict(
        VERDICT_VERIFIED,
        capture_count=count,
        expected_merkle_root=expected_root,
        recomputed_merkle_root=recomputed,
    )


# --- one-shot publication helpers ---------------------------------------------------


def dev_split_root(root: Path | None = None) -> Path:
    return (root or _project_root()) / "captures" / DEV_SPLIT_NAME


def build_and_publish_dev_split(
    *, root: Path | None = None, captures_root: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Materialize the dev split and publish its commitment. Reproducible by anyone."""
    project = root or _project_root()
    spec = dev_split_spec(root=project)
    directory = Path(captures_root) if captures_root is not None else dev_split_root(project)
    build = build_split(spec, captures_root=directory, root=project)
    commitment, leaves = publish_split(build, root=project)
    return {
        "split_id": build.split_id,
        "merkle_root": build.merkle_root,
        "capture_count": len(build.leaves),
        "commitment": str(commitment),
        "leaves": str(leaves),
        "split_root": str(directory),
    }


def _published_parameters(
    spec: SplitSpec, *, root: Path
) -> tuple[ImpairmentRanges | None, "BurstPlacement | None"]:
    """Load the pinned parameter values from data/ranges/<hash>.json and re-verify the hash."""
    ranges: ImpairmentRanges | None = None
    if spec.ranges_hash != ImpairmentRanges().ranges_hash:
        payload = _load_published_values(root, spec.ranges_hash)
        ranges = ImpairmentRanges(
            **{key: tuple(value) for key, value in payload["impairment_ranges"].items()}
        )
        if ranges.ranges_hash != spec.ranges_hash:
            raise SplitError(
                f"data/ranges/{spec.ranges_hash[:16]}….json does not hash to the committed "
                "ranges: the published values have been edited"
            )
    burst: BurstPlacement | None = None
    if spec.burst_hash is not None:
        payload = _load_published_values(root, spec.burst_hash)
        values = payload["burst_placement"]
        burst = BurstPlacement(
            extra_offset_samples=tuple(values["extra_offset_samples"]),
            trailing_samples=tuple(values["trailing_samples"]),
        )
        if burst.ranges_hash != spec.burst_hash:
            raise SplitError(
                f"data/ranges/{spec.burst_hash[:16]}….json does not hash to the committed "
                "burst placement: the published values have been edited"
            )
    return ranges, burst


def _load_published_values(root: Path, digest: str) -> dict[str, Any]:
    path = root / "data" / "ranges" / f"{digest}.json"
    if not path.is_file():
        raise SplitError(
            f"this split's parameters are pinned by hash {digest[:16]}… but no value file is "
            f"published at {path}; without the values the split cannot be rebuilt"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_split(
    split_id: str,
    *,
    captures_root: str | os.PathLike[str] | None = None,
    commitments_dir: str | os.PathLike[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Regenerate a published split's captures from its committed specification."""
    project = root or _project_root()
    document = load_commitment(split_id, commitments_dir=commitments_dir, root=project)
    if document is None:
        raise SplitError(f"no commitment is published for split {split_id}")
    spec = SplitSpec.from_dict(document.get("split_spec", {}))
    if spec.split_id != split_id:
        raise SplitError("the committed specification does not hash to this split id")
    if spec.kind == "sealed":
        raise SplitError(
            "a sealed split can only be materialized inside an open_sealed session with its salt"
        )
    directory = (
        Path(captures_root) if captures_root is not None else (project / "captures" / spec.name)
    )
    ranges, burst = _published_parameters(spec, root=project)
    build = build_split(spec, captures_root=directory, ranges=ranges, burst=burst, root=project)
    # write the leaf list beside the captures so verify-split can cross-check per capture
    _write_json_once(
        _sealed_contained_leaves_root(build, None, root=project) / LEAVES_FILENAME,
        leaves_document(build),
    )
    expected = document.get("merkle_root")
    return {
        "split_id": build.split_id,
        "capture_count": len(build.leaves),
        "captures_root": str(directory),
        "merkle_root": build.merkle_root,
        "committed_merkle_root": expected if isinstance(expected, str) else None,
        "matches_commitment": build.merkle_root == expected,
    }


def sealed_split_root(split_id: str, *, policy_root: Path | None = None) -> Path:
    """Sealed captures live only under the configured sealed root, never in the repo."""
    return Path(load_policy(root=policy_root).sealed_root) / split_id


__all__ = [
    "BLOCK_DERIVATION",
    "COMMITMENTS_RELATIVE",
    "DEV_SPLIT_NAME",
    "DEV_SPLIT_SIZE",
    "HMAC_DERIVATION",
    "SEALED_SPLIT_NAME",
    "SEALED_SPLIT_SIZE",
    "SplitBuild",
    "SplitError",
    "SplitSpec",
    "VERDICT_ENVIRONMENT_CHANGED",
    "VERDICT_MISMATCH",
    "VERDICT_MISSING",
    "VERDICT_VERIFIED",
    "block_seeds",
    "build_and_publish_dev_split",
    "build_split",
    "carries_sealed_seeds",
    "commitment_path",
    "derive_seeds",
    "dev_split_root",
    "dev_split_spec",
    "is_sealed_seed",
    "load_commitment",
    "materialize_split",
    "publish_split",
    "published_leaf_digests",
    "read_sealed_salt",
    "sealed_split_root",
    "sealed_split_spec",
    "split_seeds",
    "verify_split",
]
