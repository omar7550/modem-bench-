"""Generate a private, content-addressed reference receiver."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
from typing import Any

from ..reference_rx import _local_interpolate, decode, recover_symbols, rrc_taps
from ..sealed import (
    MANIFEST_ARTIFACT,
    ORACLE_SUBDIR,
    private_artifact_path,
    private_dir,
    read_private_artifact,
    write_private_artifact,
)
from .ast_gate import check_source


def _applied(section: dict[str, Any] | None, *names: str, default: float = 0.0) -> float:
    if not section:
        return default
    if "applied_value" in section:
        return float(section["applied_value"])
    if section.get("enabled") is False:
        return default
    for name in names:
        if name in section:
            return float(section[name])
    return default


def oracle_artifact_name(digest: str) -> str:
    """The content-addressed name of a generated oracle inside the protected root."""
    return f"{ORACLE_SUBDIR}/receiver-{digest}.py"


def make_oracle_source(capture_dir: str | Path, token: Any = None) -> bytes:
    """Return the decode closure with capture truth inlined as literals.

    Inlines sync word, payload length and impairments, so the manifest read is gated
    through the sealed chokepoint like any other private read.
    """
    capture = Path(capture_dir).resolve()
    manifest = json.loads(read_private_artifact(capture, MANIFEST_ARTIFACT, token))
    waveform = manifest["waveform"]
    framing = manifest["framing"]
    impairments = manifest.get("impairments", {})
    fractional = impairments.get("fractional_timing", {})
    constants = {
        "sps": int(waveform["sps"]),
        "beta": float(waveform["rrc_beta"]),
        "offset": int(waveform["packet_offset_samples"]),
        "sync_bits": [int(bit) for bit in framing["sync_bits"]],
        "payload_len": int(framing["payload_length_bytes"]),
        "cfo_hz": _applied(impairments.get("cfo"), "cfo_hz"),
        "phase_rad": _applied(
            impairments.get("phase"), "phase_rad", default=float(waveform.get("phase_rad", 0.0))
        ),
        "amplitude": _applied(
            impairments.get("amplitude"), "amplitude", default=float(waveform.get("amplitude", 1.0))
        ),
        "timing_mu": _applied(fractional, "offset_symbols"),
        "fd_group_delay": int(
            impairments.get("fd_group_delay_samples", fractional.get("fd_group_delay_samples", 0))
        ),
    }
    closure = "\n\n".join(
        inspect.getsource(function).strip()
        for function in (rrc_taps, _local_interpolate, recover_symbols, decode)
    )
    source = f'''# Private generated ModemBench oracle receiver.
import numpy as np
import scipy.signal as signal

{closure}

ORACLE_CONSTANTS = {constants!r}

def receive(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    values = ORACLE_CONSTANTS
    return decode(
        iq,
        sample_rate,
        sps=values["sps"],
        beta=values["beta"],
        offset=values["offset"],
        sync_bits=np.asarray(values["sync_bits"], dtype=np.uint8),
        payload_len=values["payload_len"],
        cfo_hz=values["cfo_hz"],
        phase_rad=values["phase_rad"],
        amplitude=values["amplitude"],
        timing_mu=values["timing_mu"],
        fd_group_delay=values["fd_group_delay"],
    )
'''
    raw = source.encode("utf-8")
    verdict = check_source(raw)
    if not verdict["ok"]:
        raise RuntimeError(f"generated oracle failed AST policy: {verdict['violations']!r}")
    return raw


def materialize_oracle_source(capture_dir: str | Path, token: Any = None) -> Path:
    """Write the generated oracle only within the capture's protected store."""
    capture = Path(capture_dir).resolve()
    protected = private_dir(capture).resolve()
    if not protected.is_dir():
        raise ValueError("capture has no protected private root")
    raw = make_oracle_source(capture, token)
    digest = sha256(raw).hexdigest()
    name = oracle_artifact_name(digest)
    try:
        existing = read_private_artifact(capture, name, token)
    except OSError:
        existing = None
    if existing is not None:
        if existing != raw:
            raise RuntimeError("oracle content-address collision")
        return private_artifact_path(capture, name)
    return write_private_artifact(capture, name, raw, token)
