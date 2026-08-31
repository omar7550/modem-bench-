"""Seeded deterministic BPSK capture generation with channel impairments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as signal

from .framing import (
    CRC_BITS,
    DEFAULT_MODULATION,
    LENGTH_BITS,
    MAX_PAYLOAD_BYTES,
    MIN_PAYLOAD_BYTES,
    MODULATION_POLICY_VERSION,
    QPSK_PHASE_AMBIGUITY_FOLD,
    SYNC_BITS,
    bits_per_symbol,
    build_frame,
    map_symbols,
    normalize_modulation,
)
from .impairments import ImpairmentConfig, ImpairmentRanges, apply_impairments
from .reference_rx import rrc_taps
from .sealed import (
    IQ_ARTIFACT,
    MANIFEST_ARTIFACT,
    META_ARTIFACT,
    PAYLOAD_ARTIFACT,
    SEED_ARTIFACT,
    SealedToken,
    authorize_generation,
    read_private_artifact,
    write_private_artifact,
)

SCHEMA_VERSION = "2.0"
META_SCHEMA_VERSION = "2.0"
GENERATOR_VERSION = "2.0.0"
# Versions bump per family, not globally: these strings are hashed into every manifest,
# so a global bump would rename every capture_id and invalidate the dev-v1 commitment.
QPSK_SCHEMA_VERSION = "2.1"
QPSK_META_SCHEMA_VERSION = "2.1"
QPSK_GENERATOR_VERSION = "2.1.0"
SAMPLE_RATE_HZ = 1_000_000.0
SUBSTREAM_NAMES = ("payload", "sync", "length", "waveform", "impairments", "noise")

BURST_POLICY_VERSION = "modembench-burst-v1"


@dataclass(frozen=True)
class BurstPlacement:
    """The burst-placement axis: where the packet sits inside the capture.

    None is byte-transparent: the extra draws come from a new substream appended after the
    six frozen ones, so the committed bytes are untouched, and the manifest block is elided
    at default.
    """

    #: extra leading silence on top of the frozen 1,000-20,000
    extra_offset_samples: tuple[int, int] = (0, 80_000)
    #: trailing silence, so the trailing edge is an inference, not the end of file
    trailing_samples: tuple[int, int] = (5_000, 40_000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_offset_samples": list(self.extra_offset_samples),
            "trailing_samples": list(self.trailing_samples),
            "policy_version": BURST_POLICY_VERSION,
        }

    @property
    def ranges_hash(self) -> str:
        return sha256(canonical_json(self.to_dict())).hexdigest()

    def __post_init__(self) -> None:
        for name, (low, high) in (
            ("extra_offset_samples", self.extra_offset_samples),
            ("trailing_samples", self.trailing_samples),
        ):
            if int(low) != low or int(high) != high or low < 0 or high < low:
                raise ValueError(f"{name} must be a non-decreasing pair of non-negative ints")
# Tuple so the shared copy cannot be mutated through a returned metadata dict.
_IMPAIRMENT_FAMILY = (
    {"name": "fractional sample delay", "unit": "samples"},
    {"name": "carrier frequency offset", "unit": "Hz"},
    {"name": "carrier phase", "unit": "rad"},
    {"name": "linear gain", "unit": "linear"},
    {"name": "additive white Gaussian noise", "unit": "dB Es/N0"},
)


class CaptureConflictError(RuntimeError):
    """An existing deterministic capture differs from the expected content."""


@dataclass(frozen=True)
class GeneratedCapture:
    capture_id: str
    capture_dir: Path
    reused: bool


@dataclass(frozen=True)
class ArtifactVersions:
    """The version triple stamped into one capture's artifacts."""

    schema: str
    meta_schema: str
    generator: str


ARTIFACT_VERSIONS = {
    "bpsk": ArtifactVersions(SCHEMA_VERSION, META_SCHEMA_VERSION, GENERATOR_VERSION),
    "qpsk": ArtifactVersions(
        QPSK_SCHEMA_VERSION, QPSK_META_SCHEMA_VERSION, QPSK_GENERATOR_VERSION
    ),
}


def _artifact_versions(modulation: str) -> ArtifactVersions:
    return ARTIFACT_VERSIONS[normalize_modulation(modulation)]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _npy_v1_bytes(array: np.ndarray) -> bytes:
    destination = io.BytesIO()
    np.lib.format.write_array(destination, array, version=(1, 0), allow_pickle=False)
    return destination.getvalue()


def _signal_family(modulation: str) -> dict[str, Any]:
    """Describe the waveform family. The label is public because it is a generation
    parameter, constant per split; only per-capture draws are withheld. A split that mixed
    families per capture would make it an instance value and force a meta-v3."""
    if normalize_modulation(modulation) == "bpsk":
        return {
            "modulation": "BPSK",
            "mapping": {"0": 1.0, "1": -1.0},
            "pulse_shape": "root-raised-cosine",
            "pulse_span_symbols": 12,
            "impairments": _IMPAIRMENT_FAMILY,
        }
    root_half = float(1.0 / np.sqrt(2.0))
    return {
        "modulation": "QPSK",
        "bits_per_symbol": 2,
        "gray_coded": True,
        "dibit_order": "msb-first: b0 is the earlier of the two bits",
        "mapping": {
            "00": [root_half, root_half],
            "01": [root_half, -root_half],
            "10": [-root_half, root_half],
            "11": [-root_half, -root_half],
        },
        "mapping_formula": "s = [(1-2*b0) + 1j*(1-2*b1)] / sqrt(2)",
        "phase_ambiguity": {
            "fold": QPSK_PHASE_AMBIGUITY_FOLD,
            "period_rad": float(np.pi / 2.0),
            "reason": "the QPSK constellation is invariant under rotation by pi/2",
            "disambiguation": (
                "sync word: derotate by k*pi/2 for k in 0..3, demap, and take the k of "
                "minimum Hamming distance to the known sync word; ties break to the "
                "smallest k"
            ),
        },
        "pulse_shape": "root-raised-cosine",
        "pulse_span_symbols": 12,
        "impairments": _IMPAIRMENT_FAMILY,
    }


def _public_metadata(capture_id: str, modulation: str = DEFAULT_MODULATION) -> dict[str, Any]:
    """Return family-and-units-only public metadata (never instance truth)."""
    versions = _artifact_versions(modulation)
    return {
        "schema_version": versions.meta_schema,
        "generator_version": versions.generator,
        "capture_id": capture_id,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "framing": {
            "layout": ["sync", "length", "payload", "crc"],
            "sync_bits": SYNC_BITS,
            "length_bits": LENGTH_BITS,
            "length_encoding": "unsigned 16-bit big-endian payload byte count",
            "payload_length_bytes": {"min": MIN_PAYLOAD_BYTES, "max": MAX_PAYLOAD_BYTES},
            "bit_order": "msb-first-within-every-byte",
            "crc": {
                "variant": "CRC-32/IEEE 802.3 (zlib.crc32)",
                "covers": "length+payload bytes",
                "bits": CRC_BITS,
                "serialization": "unsigned 32-bit big-endian, then MSB-first bits",
            },
        },
        "signal_family": _signal_family(modulation),
        "receiver_output": {
            "format": "NumPy .npy v1.0",
            "dtype": "uint8",
            "shape": "1-D",
            "values": [0, 1],
            "sync_start_bit_range": [0, 64],
        },
    }


def _build_artifacts(
    master_seed: int,
    *,
    config: ImpairmentConfig,
    ranges: ImpairmentRanges,
    sealed_token: SealedToken | None = None,
    captures_dir: str | Path | None = None,
    modulation: str = DEFAULT_MODULATION,
    burst: BurstPlacement | None = None,
) -> tuple[str, dict[str, bytes], dict[str, bytes]]:
    """Derive every capture artifact from the seed; this is the generation boundary.

    Returns (capture_id, public_artifacts, protected_artifacts). The guard lives here, not
    only on generate_capture, because splits.py imports this engine directly. modulation
    takes no RNG draw and its manifest block is elided at the default, so a BPSK manifest
    stays byte-identical to schema 2.0 and the dev-v1 commitment still verifies.
    """
    authorize_generation(master_seed, captures_dir, sealed_token)
    family = normalize_modulation(modulation)
    versions = _artifact_versions(family)
    seed_sequence = np.random.SeedSequence(master_seed)
    # Spawn children are a pure function of their index, so appending a seventh leaves the
    # six frozen streams untouched. Never reorder this tuple.
    substream_names = SUBSTREAM_NAMES + (("burst",) if burst is not None else ())
    children = seed_sequence.spawn(len(substream_names))
    streams = {name: np.random.default_rng(child) for name, child in zip(substream_names, children)}

    payload_len = int(streams["length"].integers(MIN_PAYLOAD_BYTES, MAX_PAYLOAD_BYTES + 1))
    payload_array = streams["payload"].integers(0, 256, size=payload_len, dtype=np.uint8)
    payload = payload_array.tobytes()
    sync_bits = streams["sync"].integers(0, 2, size=SYNC_BITS, dtype=np.uint8)
    sps = int(streams["waveform"].integers(10, 41))
    beta = float(streams["waveform"].uniform(0.2, 0.5))
    offset = int(streams["waveform"].integers(1000, 20001))
    # burst draws come from the appended substream only; touching "waveform" would
    # re-deal sps and beta for every existing seed
    extra_offset = 0
    trailing = 0
    if burst is not None:
        lo, hi = burst.extra_offset_samples
        extra_offset = int(streams["burst"].integers(lo, hi + 1))
        lo, hi = burst.trailing_samples
        trailing = int(streams["burst"].integers(lo, hi + 1))
    total_offset = offset + extra_offset

    frame_bits = build_frame(sync_bits, payload)
    taps = rrc_taps(sps, beta)
    if frame_bits.size % bits_per_symbol(family):
        raise ValueError(f"frame bit count is not a whole number of {family} symbols")
    symbol_count = int(frame_bits.size) // bits_per_symbol(family)
    shaped = signal.upfirdn(taps, map_symbols(frame_bits, family), up=sps)
    clean = np.concatenate(
        (
            np.zeros(total_offset, dtype=np.complex128),
            shaped.astype(np.complex128, copy=False),
            np.zeros(trailing, dtype=np.complex128),
        )
    )
    impaired = apply_impairments(
        clean,
        offset=total_offset,
        trailing_samples=trailing,
        packet_waveform_length=int(shaped.size),
        # Es/N0 is per symbol: a QPSK frame carries the same bits in half the symbols
        n_symbols=symbol_count,
        fs=SAMPLE_RATE_HZ,
        sps=sps,
        impairment_rng=streams["impairments"],
        noise_rng=streams["noise"],
        config=config,
        ranges=ranges,
    )
    iq_bytes = _npy_v1_bytes(impaired.samples)

    payload_sha = sha256(payload).hexdigest()
    iq_sha = sha256(iq_bytes).hexdigest()
    manifest_without_id: dict[str, Any] = {
        "schema_version": versions.schema,
        "generator_version": versions.generator,
        "waveform": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sps": sps,
            "rrc_beta": beta,
            "rrc_span_symbols": 12,
            "rrc_ntaps": int(taps.size),
            "packet_offset_samples": total_offset,
            "packet_waveform_length_samples": int(shaped.size),
            "support_stop_samples": impaired.support_stop,
        },
        "framing": {
            "sync_bits": sync_bits.tolist(),
            "payload_length_bytes": payload_len,
            "frame_bit_count": int(frame_bits.size),
            "bit_order": "msb-first",
        },
        "impairments": {
            **impaired.manifest,
            "fd_group_delay_samples": impaired.manifest["fractional_timing"][
                "fd_group_delay_samples"
            ],
        },
        "hashes": {IQ_ARTIFACT: iq_sha, PAYLOAD_ARTIFACT: payload_sha},
    }
    if burst is not None:
        manifest_without_id["burst"] = {
            "enabled": True,
            "ranges": burst.to_dict(),
            "ranges_hash": burst.ranges_hash,
            "extra_offset_samples": extra_offset,
            "trailing_samples": trailing,
            "frozen_offset_samples": offset,
            "policy_version": BURST_POLICY_VERSION,
        }
    if family != DEFAULT_MODULATION:
        manifest_without_id["modulation"] = {
            "enabled": True,
            "default_family": DEFAULT_MODULATION,
            "applied_family": family,
            # no drawn_value: supplied by the caller, takes nothing from the RNG
            "source": "generation-parameter",
            "bits_per_symbol": bits_per_symbol(family),
            "symbol_count": symbol_count,
            "gray_coded": True,
            "phase_ambiguity_fold": QPSK_PHASE_AMBIGUITY_FOLD,
            "phase_disambiguation": "sync-word correlation over k*pi/2, k in 0..3",
            "policy_version": MODULATION_POLICY_VERSION,
        }
    capture_id = sha256(canonical_json(manifest_without_id)).hexdigest()[:12]
    manifest = {**manifest_without_id, "capture_id": capture_id}
    seed_data = {
        "schema_version": versions.schema,
        "master_seed": master_seed,
        "seed_sequence_spawn_map": {
            name: {"child_index": index, "spawn_key": list(child.spawn_key)}
            for index, (name, child) in enumerate(zip(substream_names, children))
        },
        "bit_generator": "PCG64",
    }
    artifacts = {
        IQ_ARTIFACT: iq_bytes,
        META_ARTIFACT: canonical_json(_public_metadata(capture_id, family)) + b"\n",
    }
    protected = {
        MANIFEST_ARTIFACT: canonical_json(manifest) + b"\n",
        PAYLOAD_ARTIFACT: payload,
        SEED_ARTIFACT: canonical_json(seed_data) + b"\n",
    }
    return capture_id, artifacts, protected


def generate_capture(
    master_seed: int,
    captures_dir: str | Path = "captures",
    *,
    profile: str = "clean",
    config: ImpairmentConfig | None = None,
    ranges: ImpairmentRanges | None = None,
    sealed_token: SealedToken | None = None,
    modulation: str = DEFAULT_MODULATION,
    burst: BurstPlacement | None = None,
) -> GeneratedCapture:
    """Generate, or byte-verify and reuse, a deterministic capture.

    Sealed truth is a pure function of the seed, so sealed seeds are refused without a live
    open_sealed token; the guard also lives on _build_artifacts.
    """
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    authorize_generation(master_seed, captures_dir, sealed_token)
    if config is not None and profile != "clean":
        raise ValueError("pass either a named profile or a custom config, not both")
    applied_config = config if config is not None else ImpairmentConfig.from_profile(profile)
    applied_ranges = ranges or ImpairmentRanges()
    capture_id, artifacts, protected = _build_artifacts(
        master_seed,
        config=applied_config,
        ranges=applied_ranges,
        sealed_token=sealed_token,
        captures_dir=captures_dir,
        modulation=normalize_modulation(modulation),
        burst=burst,
    )
    capture_dir = Path(captures_dir) / capture_id

    if capture_dir.exists():
        if not capture_dir.is_dir():
            raise CaptureConflictError(f"capture target is not a directory: {capture_dir}")
        for relative, expected in artifacts.items():
            path = capture_dir / relative
            try:
                actual = path.read_bytes()
            except OSError as exc:
                raise CaptureConflictError(f"existing capture is incomplete: {path}") from exc
            if actual != expected:
                raise CaptureConflictError(f"existing capture content differs: {path}")
        for name, expected in protected.items():
            try:
                actual = read_private_artifact(capture_dir, name, sealed_token)
            except OSError as exc:
                raise CaptureConflictError(
                    f"existing capture is incomplete: {capture_dir} ({name})"
                ) from exc
            if actual != expected:
                raise CaptureConflictError(
                    f"existing capture content differs: {capture_dir} ({name})"
                )
        return GeneratedCapture(capture_id, capture_dir, reused=True)

    # exist_ok=False is the atomic claim on a 48-bit content address
    capture_dir.mkdir(parents=True, exist_ok=False)
    for relative, content in artifacts.items():
        (capture_dir / relative).write_bytes(content)
    for name, content in protected.items():
        write_private_artifact(capture_dir, name, content, sealed_token)
    return GeneratedCapture(capture_id, capture_dir, reused=False)
