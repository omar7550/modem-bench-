"""Pinned RFC-6962-style Merkle commitment over seed-ordered capture leaves."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Sequence

MERKLE_VERSION = "modembench-merkle-v1"
# 0x00/0x01 domain separation keeps an internal node from being a valid leaf preimage;
# a lone right node is promoted, never duplicated (CVE-2012-2459).
LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
ROOT_PREFIX = b"\x02"
DIGEST_HEX_LENGTH = 64
MAX_SEED = (1 << 64) - 1
CONSTRUCTION = {
    "version": MERKLE_VERSION,
    "leaf": "SHA256(0x00 || uint64_be(master_seed) || raw(iq.npy) || raw(manifest.json) || raw(payload.bin))",
    "node": "SHA256(0x01 || left || right)",
    "odd_node": "promoted unchanged, never duplicated",
    "leaf_order": "master_seed ascending",
    "root": "SHA256(0x02 || uint64_be(n_leaves) || raw(split_id) || tree_root)",
}


class MerkleError(ValueError):
    """A Merkle input violates the pinned construction."""


def _raw_digest(name: str, value: str) -> bytes:
    if not isinstance(value, str) or len(value) != DIGEST_HEX_LENGTH:
        raise MerkleError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise MerkleError(f"{name} must be a 64-character SHA-256 hex digest") from exc


def _seed_bytes(master_seed: int) -> bytes:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise MerkleError("master_seed must be an integer")
    if not 0 <= master_seed <= MAX_SEED:
        raise MerkleError("master_seed must fit in an unsigned 64-bit integer")
    return master_seed.to_bytes(8, "big")


@dataclass(frozen=True)
class CaptureLeaf:
    """One capture's contribution to the tree: seed identity plus component digests."""

    master_seed: int
    iq_sha256: str
    manifest_sha256: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "master_seed": self.master_seed,
            "iq_sha256": self.iq_sha256,
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": self.payload_sha256,
        }

    @property
    def digest(self) -> bytes:
        return sha256(
            LEAF_PREFIX
            + _seed_bytes(self.master_seed)
            + _raw_digest("iq_sha256", self.iq_sha256)
            + _raw_digest("manifest_sha256", self.manifest_sha256)
            + _raw_digest("payload_sha256", self.payload_sha256)
        ).digest()


def order_leaves(leaves: Iterable[CaptureLeaf]) -> tuple[CaptureLeaf, ...]:
    """Return the leaves in the normative master_seed-ascending order."""
    ordered = tuple(sorted(leaves, key=lambda leaf: leaf.master_seed))
    seeds = [leaf.master_seed for leaf in ordered]
    if len(set(seeds)) != len(seeds):
        raise MerkleError("leaf master_seeds must be unique")
    return ordered


def leaf_digests(leaves: Sequence[CaptureLeaf]) -> tuple[bytes, ...]:
    """Digest already-ordered leaves, refusing any order but master_seed ascending."""
    seeds = [leaf.master_seed for leaf in leaves]
    if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
        raise MerkleError("leaves must be uniquely ordered by master_seed ascending")
    return tuple(leaf.digest for leaf in leaves)


def _combine(level: Sequence[bytes]) -> tuple[bytes, ...]:
    parent = [
        sha256(NODE_PREFIX + level[index] + level[index + 1]).digest()
        for index in range(0, len(level) - 1, 2)
    ]
    if len(level) % 2:
        # Promoted unchanged. Duplicating it would collide distinct leaf multisets.
        parent.append(level[-1])
    return tuple(parent)


def tree_root(digests: Sequence[bytes]) -> bytes:
    """Reduce leaf digests to the bare tree root (without the 0x02 binding wrapper)."""
    if not digests:
        raise MerkleError("a Merkle tree requires at least one leaf")
    if any(not isinstance(value, bytes) or len(value) != 32 for value in digests):
        raise MerkleError("every Merkle digest must be 32 raw bytes")
    level = tuple(digests)
    while len(level) > 1:
        level = _combine(level)
    return level[0]


def bind_root(bare_root: bytes, *, n_leaves: int, split_id: str) -> str:
    """Bind the tree root to the split identity and the leaf count."""
    if isinstance(n_leaves, bool) or not isinstance(n_leaves, int) or n_leaves <= 0:
        raise MerkleError("n_leaves must be a positive integer")
    if not isinstance(bare_root, bytes) or len(bare_root) != 32:
        raise MerkleError("the tree root must be 32 raw bytes")
    return sha256(
        ROOT_PREFIX
        + n_leaves.to_bytes(8, "big")
        + _raw_digest("split_id", split_id)
        + bare_root
    ).hexdigest()


def split_root(leaves: Sequence[CaptureLeaf], *, split_id: str) -> str:
    """Return the published root over ordered leaves for ``split_id``."""
    digests = leaf_digests(leaves)
    return bind_root(tree_root(digests), n_leaves=len(digests), split_id=split_id)


def expected_positions(index: int, n_leaves: int) -> tuple[str, ...]:
    """Return the sibling positions a valid proof for ``index`` must present."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise MerkleError("index must be an integer")
    if isinstance(n_leaves, bool) or not isinstance(n_leaves, int) or n_leaves <= 0:
        raise MerkleError("n_leaves must be a positive integer")
    if not 0 <= index < n_leaves:
        raise MerkleError("index is outside the leaf range")
    positions: list[str] = []
    width = n_leaves
    cursor = index
    while width > 1:
        if cursor == width - 1 and width % 2:
            pass  # promoted node: no sibling exists at this level
        elif cursor % 2 == 0:
            positions.append("right")
        else:
            positions.append("left")
        cursor //= 2
        width = (width + 1) // 2
    return tuple(positions)


def membership_proof(digests: Sequence[bytes], index: int) -> tuple[dict[str, str], ...]:
    """Return the sibling path proving ``digests[index]`` is in the tree."""
    if not digests:
        raise MerkleError("a Merkle tree requires at least one leaf")
    expected = expected_positions(index, len(digests))
    level = tuple(digests)
    cursor = index
    path: list[dict[str, str]] = []
    while len(level) > 1:
        if cursor == len(level) - 1 and len(level) % 2:
            pass
        elif cursor % 2 == 0:
            path.append({"position": "right", "sha256": level[cursor + 1].hex()})
        else:
            path.append({"position": "left", "sha256": level[cursor - 1].hex()})
        level = _combine(level)
        cursor //= 2
    if tuple(step["position"] for step in path) != expected:
        raise MerkleError("membership proof shape disagrees with the pinned construction")
    return tuple(path)


def verify_proof(
    digest: bytes,
    proof: Sequence[dict[str, str]],
    *,
    index: int,
    n_leaves: int,
    split_id: str,
    root: str,
) -> bool:
    """Verify a raw digest at ``index``; the shape is pinned by ``(index, n_leaves)``."""
    try:
        expected = expected_positions(index, n_leaves)
    except MerkleError:
        return False
    if not isinstance(digest, bytes) or len(digest) != 32:
        return False
    if len(proof) != len(expected):
        return False
    node = digest
    for step, position in zip(proof, expected):
        if not isinstance(step, dict) or step.get("position") != position:
            return False
        try:
            sibling = _raw_digest("proof step", step.get("sha256", ""))
        except MerkleError:
            return False
        if position == "right":
            node = sha256(NODE_PREFIX + node + sibling).digest()
        else:
            node = sha256(NODE_PREFIX + sibling + node).digest()
    try:
        computed = bind_root(node, n_leaves=n_leaves, split_id=split_id)
    except MerkleError:
        return False
    return computed == root


def verify_membership(
    leaf: CaptureLeaf,
    proof: Sequence[dict[str, str]],
    *,
    index: int,
    n_leaves: int,
    split_id: str,
    root: str,
) -> bool:
    """Verify that ``leaf`` sits at ``index`` of the tree committed by ``root``."""
    try:
        digest = leaf.digest
    except MerkleError:
        return False
    return verify_proof(
        digest, proof, index=index, n_leaves=n_leaves, split_id=split_id, root=root
    )
