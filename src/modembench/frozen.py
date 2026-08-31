"""The freeze artifact (data/frozen_difficulty.json): written once by `modembench freeze`, verified at the top of every funded arm run."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .records import write_json_once

FROZEN_POLICY_VERSION = "modembench-frozen-v1"

#: The documents whose rules govern the campaign, pinned by content hash at freeze time.
GOVERNING_DOCUMENTS = (
    "docs/pre-registration.md",
    "docs/industry-anchoring.md",
    "docs/difficulty-calibration.md",
    "docs/n-r-rederivation.md",
)


class FrozenDriftError(RuntimeError):
    """The live repository disagrees with the frozen record; the arm run must not spend."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frozen_path(root: Path | None = None) -> Path:
    return (root or _repo_root()) / "data" / "frozen_difficulty.json"


def _document_shas(root: Path) -> dict[str, str]:
    missing = [name for name in GOVERNING_DOCUMENTS if not (root / name).is_file()]
    if missing:
        raise FrozenDriftError(f"governing documents are missing: {missing}")
    return {
        name: sha256((root / name).read_bytes()).hexdigest() for name in GOVERNING_DOCUMENTS
    }


def build_freeze_document(
    *,
    split_id: str,
    config: "Any" = None,
    root: Path | None = None,
    coverage_rehearsal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the freeze document entirely from the live repository; nothing is typed in."""
    from .arms import budget as budget_module
    from .agent.tools import tools_sha256

    base = root or _repo_root()
    commitment_file = base / "data" / "commitments" / f"{split_id}.json"
    if not commitment_file.is_file():
        raise FrozenDriftError(f"no published commitment for split {split_id[:16]}…")
    commitment = json.loads(commitment_file.read_text(encoding="utf-8"))
    spec = commitment["split_spec"]

    settings = config
    if settings is None:
        from .agent.harness import AgentConfig

        settings = AgentConfig(withheld_tools=("symbol_rate_candidates",))
    n = budget_module.BUDGET.n_attempts
    return {
        "frozen_policy_version": FROZEN_POLICY_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "split": {
            "split_id": split_id,
            "name": spec["name"],
            "size": spec["size"],
            "ranges_hash": spec["ranges_hash"],
            "burst_hash": spec.get("burst_hash"),
            "merkle_root": commitment["merkle_root"],
        },
        "instruments": {
            "withheld_tools": list(settings.withheld_tools),
            "tools_sha256": tools_sha256(),
        },
        "sizing": {
            "n_attempts": n,
            "iterative_round_cap": n,
            "replicates": 3,
        },
        "frozen_budget_sha256": budget_module.frozen_budget_hash(config=settings),
        "arm_invariant_digest": budget_module.arm_invariant_digest(settings),
        "document_shas": _document_shas(base),
        "coverage_rehearsal": dict(coverage_rehearsal) if coverage_rehearsal else None,
    }


def freeze(
    *,
    split_id: str,
    config: "Any" = None,
    root: Path | None = None,
    coverage_rehearsal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the freeze artifact once. Idempotent up to the timestamp; differing content is a hard error."""
    document = build_freeze_document(
        split_id=split_id, config=config, root=root, coverage_rehearsal=coverage_rehearsal
    )
    path = frozen_path(root)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        def _timeless(value: Mapping[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in value.items() if k not in ("frozen_at", "coverage_rehearsal")}
        if _timeless(existing) == _timeless(document):
            return existing
    write_json_once(path, document, description="the freeze artifact")
    return document


def verify_frozen(config: "Any" = None, *, root: Path | None = None) -> dict[str, Any]:
    """Re-derive every frozen value from the live repository and compare. Raise on any drift.

    A missing artifact is itself drift: no funded run may spend before the freeze.
    """
    path = frozen_path(root)
    if not path.is_file():
        raise FrozenDriftError(
            f"no freeze artifact at {path}: run `modembench freeze` before any funded arm "
            "run — spending before the freeze is what starts the amendment clock unanchored"
        )
    recorded = json.loads(path.read_text(encoding="utf-8"))
    # Compare against the CANONICAL config (config=None): frozen_budget_sha256 includes the
    # arm-specific digest, and the two arms legitimately differ there.
    live = build_freeze_document(split_id=recorded["split"]["split_id"], root=root)
    mismatches: list[str] = []
    for section in ("split", "instruments", "sizing"):
        for key, value in recorded[section].items():
            if live[section].get(key) != value:
                mismatches.append(
                    f"{section}.{key}: frozen {value!r}, live {live[section].get(key)!r}"
                )
    for key in ("frozen_budget_sha256", "arm_invariant_digest"):
        if live[key] != recorded[key]:
            mismatches.append(f"{key}: frozen {recorded[key][:16]}…, live {live[key][:16]}…")
    # The arm's own config is held only to the arm-invariant digest and the instrument set.
    if config is not None:
        from .arms import budget as budget_module

        live_invariant = budget_module.arm_invariant_digest(config)
        if live_invariant != recorded["arm_invariant_digest"]:
            mismatches.append(
                f"this arm's invariant digest {live_invariant[:16]}… differs from the frozen "
                f"{recorded['arm_invariant_digest'][:16]}… — a different experiment"
            )
        if list(config.withheld_tools) != recorded["instruments"]["withheld_tools"]:
            mismatches.append(
                f"this arm withholds {list(config.withheld_tools)!r}; the freeze recorded "
                f"{recorded['instruments']['withheld_tools']!r}"
            )
    for name, digest in recorded["document_shas"].items():
        if live["document_shas"].get(name) != digest:
            mismatches.append(f"document {name} changed after the freeze")
    if mismatches:
        raise FrozenDriftError(
            "the live repository disagrees with the frozen record; refusing to spend:\n  "
            + "\n  ".join(mismatches)
        )
    return recorded
