"""Private frame evaluator with defensive receiver-output loading."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from .framing import MAX_PAYLOAD_BYTES, MIN_PAYLOAD_BYTES, bits_to_bytes, bytes_to_bits
from .sealed import (
    MANIFEST_ARTIFACT,
    PAYLOAD_ARTIFACT,
    SealedAccessError,
    read_private_artifact,
)

MAX_OUTPUT_BITS = 2**20
MAX_ALIGNMENT_OFFSET = 64
MAX_SYNC_HAMMING = 8


def _result(
    *,
    acquisition_success: bool = False,
    crc_pass: bool = False,
    aligned_ber: float | None = None,
    error: str | None = None,
    packet_success: bool = False,
    n_payload_bits: int = 0,
    alignment_offset: int | None = None,
    sync_hamming: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the locked agent-feedback/orchestrator-internal JSON split."""
    return {
        "feedback": {
            "acquisition_success": acquisition_success,
            "crc_pass": crc_pass,
            "aligned_ber": aligned_ber,
            "error": error,
        },
        "internal": {
            "packet_success": packet_success,
            "n_payload_bits": n_payload_bits,
            "alignment_offset": alignment_offset,
            "sync_hamming": sync_hamming,
        },
    }


def truth_invalid_result() -> dict[str, dict[str, Any]]:
    """The structured result for truth that could not be loaded or parsed."""
    return _result(error="private_truth_invalid")


def _read_npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    """Parse and validate the allocation-relevant .npy header only."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)
        if version in {(2, 0), (3, 0)}:
            return np.lib.format.read_array_header_2_0(stream)
        raise ValueError(f"unsupported npy version: {version}")


def load_receiver_output(path: str | Path) -> tuple[np.ndarray | None, str | None]:
    """Validate dtype/shape/size before allocation, then load without pickle."""
    output_path = Path(path)
    if not output_path.is_file():
        return None, "output_missing"
    try:
        shape, _fortran_order, dtype = _read_npy_header(output_path)
    except (OSError, ValueError, EOFError, TypeError):
        return None, "output_invalid_header"
    if any(type(dim) is not int or dim < 0 for dim in shape):
        return None, "output_invalid_header"
    if len(shape) != 1:
        return None, "output_wrong_shape"
    if np.dtype(dtype) != np.dtype(np.uint8):
        return None, "output_wrong_dtype"
    if any(dim > MAX_OUTPUT_BITS for dim in shape):
        return None, "output_too_long"

    try:
        bits = np.load(output_path, allow_pickle=False)
    except (OSError, ValueError, EOFError, TypeError):
        return None, "output_corrupt"
    if bits.shape != shape or bits.ndim != 1 or bits.dtype != np.uint8:
        return None, "output_corrupt"
    if np.any(bits > 1):
        return None, "output_nonbinary"
    return bits, None


def _load_truth(protected_dir: Path, token: Any = None) -> tuple[dict[str, Any], bytes]:
    """Load one capture's truth through the sealed chokepoint, never by path."""
    capture_dir = Path(protected_dir).parent
    manifest_bytes = read_private_artifact(capture_dir, MANIFEST_ARTIFACT, token)
    payload = read_private_artifact(capture_dir, PAYLOAD_ARTIFACT, token)
    return parse_truth(manifest_bytes, payload)


def parse_truth(manifest_bytes: bytes, payload: bytes) -> tuple[dict[str, Any], bytes]:
    """Validate an already-loaded manifest/payload pair against the framing contract."""
    manifest = json.loads(manifest_bytes)
    framing = manifest["framing"]
    sync = framing["sync_bits"]
    payload_len = framing["payload_length_bytes"]
    if (
        not isinstance(sync, list)
        or len(sync) != 64
        or any(type(bit) is not int or bit not in (0, 1) for bit in sync)
        or type(payload_len) is not int
        or not MIN_PAYLOAD_BYTES <= payload_len <= MAX_PAYLOAD_BYTES
        or len(payload) != payload_len
    ):
        raise ValueError("invalid private framing truth")
    hashes = manifest["hashes"]
    if sha256(payload).hexdigest() != hashes[PAYLOAD_ARTIFACT]:
        raise ValueError("protected payload hash mismatch")
    return manifest, payload


def _claimed_length(bits: np.ndarray, alignment: int) -> int | None:
    start = alignment + 64
    stop = start + 16
    if bits.size < stop:
        return None
    return int.from_bytes(bits_to_bytes(bits[start:stop]), "big")


def _crc_passes(bits: np.ndarray, alignment: int, claimed_length: int | None) -> bool:
    if claimed_length is None or not MIN_PAYLOAD_BYTES <= claimed_length <= MAX_PAYLOAD_BYTES:
        return False
    length_start = alignment + 64
    payload_start = length_start + 16
    payload_stop = payload_start + claimed_length * 8
    crc_stop = payload_stop + 32
    if bits.size < crc_stop:
        return False
    try:
        length_bytes = bits_to_bytes(bits[length_start:payload_start])
        payload_bytes = bits_to_bytes(bits[payload_start:payload_stop])
        claimed_crc = int.from_bytes(bits_to_bytes(bits[payload_stop:crc_stop]), "big")
    except ValueError:
        return False
    actual_crc = zlib.crc32(length_bytes + payload_bytes) & 0xFFFF_FFFF
    return claimed_crc == actual_crc


def evaluate_bits(bits: np.ndarray, manifest: dict[str, Any], payload: bytes) -> dict[str, dict[str, Any]]:
    """Score already-validated output according to the locked deterministic rules."""
    true_sync = np.asarray(manifest["framing"]["sync_bits"], dtype=np.uint8)
    true_length = int(manifest["framing"]["payload_length_bytes"])
    truth_payload_bits = bytes_to_bits(payload)
    n_payload_bits = true_length * 8

    # Candidates are exactly integer k in [0,64] whose full sync is present.
    candidates: list[tuple[int, int]] = []
    largest_k = min(MAX_ALIGNMENT_OFFSET, int(bits.size) - true_sync.size)
    for k in range(largest_k + 1):
        hamming = int(np.count_nonzero(bits[k : k + true_sync.size] != true_sync))
        candidates.append((hamming, k))
    if not candidates:
        return _result(n_payload_bits=n_payload_bits)

    best_hamming, best_k = min(candidates)  # lowest Hamming, then smallest k
    if best_hamming > MAX_SYNC_HAMMING:
        return _result(n_payload_bits=n_payload_bits, sync_hamming=best_hamming)

    payload_start = best_k + 64 + 16
    available = max(0, min(n_payload_bits, int(bits.size) - payload_start))
    mismatches = int(
        np.count_nonzero(bits[payload_start : payload_start + available] != truth_payload_bits[:available])
    )
    # Every missing payload bit is scored at chance (0.5), preserving continuity
    # with the downstream null-BER -> 0.5 imputation convention.
    aligned_ber = (mismatches + 0.5 * (n_payload_bits - available)) / n_payload_bits
    claimed_length = _claimed_length(bits, best_k)
    crc_pass = _crc_passes(bits, best_k, claimed_length)
    packet_success = bool(
        claimed_length == true_length and aligned_ber == 0.0 and crc_pass
    )
    return _result(
        acquisition_success=True,
        crc_pass=crc_pass,
        aligned_ber=float(aligned_ber),
        packet_success=packet_success,
        n_payload_bits=n_payload_bits,
        alignment_offset=best_k,
        sync_hamming=best_hamming,
    )


def evaluate_file(
    bits_path: str | Path, protected_dir: str | Path, token: Any = None
) -> dict[str, dict[str, Any]]:
    """Evaluate a receiver .npy output; SealedAccessError propagates rather than
    becoming a quiet private_truth_invalid."""
    try:
        manifest, payload = _load_truth(Path(protected_dir), token)
    except SealedAccessError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return _result(error="private_truth_invalid")
    return evaluate_output(bits_path, manifest, payload)


def evaluate_output(
    bits_path: str | Path, manifest: dict[str, Any], payload: bytes
) -> dict[str, dict[str, Any]]:
    """Score a receiver output against truth that has already been loaded."""
    try:
        bits, error = load_receiver_output(bits_path)
    except (OSError, ValueError, EOFError, TypeError, OverflowError):
        return _result(error="output_invalid_header")
    if error is not None or bits is None:
        return _result(error=error)
    try:
        return evaluate_bits(bits, manifest, payload)
    except (ValueError, KeyError, TypeError, OverflowError):
        return _result(error="evaluation_invalid_truth")
