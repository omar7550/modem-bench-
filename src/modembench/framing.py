"""Normative packet framing and symbol mapping for ModemBench's waveform families."""

from __future__ import annotations

import zlib

import numpy as np

SYNC_BITS = 64
LENGTH_BITS = 16
CRC_BITS = 32
MIN_PAYLOAD_BYTES = 32
MAX_PAYLOAD_BYTES = 128

MODULATION_POLICY_VERSION = "modembench-modulation-v1"
#: The frozen default; artifact bytes on this path must stay identical.
DEFAULT_MODULATION = "bpsk"
MODULATION_FAMILIES = ("bpsk", "qpsk")
BITS_PER_SYMBOL = {"bpsk": 1, "qpsk": 2}
#: The QPSK constellation is invariant under rotation by pi/2.
QPSK_PHASE_AMBIGUITY_FOLD = 4


def bytes_to_bits(data: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Serialize bytes MSB-first into a one-dimensional uint8 bit array."""
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="big").astype(np.uint8, copy=False)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a byte-aligned bit array MSB-first."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or values.size % 8:
        raise ValueError("bits must be a one-dimensional, byte-aligned array")
    if np.any(values > 1):
        raise ValueError("bits must contain only 0 and 1")
    return np.packbits(values, bitorder="big").tobytes()


def serialize_length(payload_len: int) -> bytes:
    """Serialize the legal payload byte count as unsigned big-endian uint16."""
    if not MIN_PAYLOAD_BYTES <= payload_len <= MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload length must be in [{MIN_PAYLOAD_BYTES}, {MAX_PAYLOAD_BYTES}]")
    return payload_len.to_bytes(2, "big")


def crc_bytes(length_bytes: bytes, payload: bytes) -> bytes:
    """CRC-32/IEEE (zlib) over length+payload, serialized big-endian."""
    if len(length_bytes) != 2:
        raise ValueError("length field must contain exactly two bytes")
    return (zlib.crc32(length_bytes + payload) & 0xFFFF_FFFF).to_bytes(4, "big")


def build_frame(sync_bits: np.ndarray, payload: bytes) -> np.ndarray:
    """Build sync|length|payload|CRC as a uint8 MSB-first bit array."""
    sync = np.asarray(sync_bits, dtype=np.uint8)
    if sync.shape != (SYNC_BITS,) or np.any(sync > 1):
        raise ValueError(f"sync_bits must contain exactly {SYNC_BITS} binary values")
    length = serialize_length(len(payload))
    body = length + payload
    return np.concatenate((sync, bytes_to_bits(body), bytes_to_bits(crc_bytes(length, payload))))


def bpsk_symbols(bits: np.ndarray) -> np.ndarray:
    """Map b to 1-2b in float64: 0 -> +1 and 1 -> -1."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    return 1.0 - 2.0 * values.astype(np.float64)


def normalize_modulation(modulation: str) -> str:
    """Return the canonical lowercase family name, or raise for an unknown one."""
    if not isinstance(modulation, str):
        raise ValueError("modulation must be a string")
    name = modulation.strip().lower()
    if name not in MODULATION_FAMILIES:
        raise ValueError(f"unknown modulation family: {modulation!r}")
    return name


def bits_per_symbol(modulation: str) -> int:
    """Bits carried by one symbol of ``modulation``."""
    return BITS_PER_SYMBOL[normalize_modulation(modulation)]


def qpsk_symbols(bits: np.ndarray) -> np.ndarray:
    """s_i = [(1-2*b0) + 1j*(1-2*b1)] / sqrt(2), Gray-coded, b0 the earlier (MSB) bit."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if values.size % 2:    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if values.size % 2:
        raise ValueError("QPSK needs an even bit count to form whole dibits")
    dibits = values.reshape(-1, 2).astype(np.float64)
    return ((1.0 - 2.0 * dibits[:, 0]) + 1j * (1.0 - 2.0 * dibits[:, 1])) / np.sqrt(2.0)


def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    """Invert qpsk_symbols: b0 = [Re(s) < 0], b1 = [Im(s) < 0], interleaved."""
    values = np.asarray(symbols)
    if values.ndim != 1:
        raise ValueError("symbols must be a one-dimensional array")
    bits = np.empty(2 * values.size, dtype=np.uint8)
    bits[0::2] = (np.real(values) < 0.0).astype(np.uint8)
    bits[1::2] = (np.imag(values) < 0.0).astype(np.uint8)
    return bits


def map_symbols(bits: np.ndarray, modulation: str = DEFAULT_MODULATION) -> np.ndarray:
    """Dispatch to the family's symbol map; the BPSK branch stays byte-frozen."""
    family = normalize_modulation(modulation)
    return bpsk_symbols(bits) if family == "bpsk" else qpsk_symbols(bits)


def demap_symbols(symbols: np.ndarray, modulation: str = DEFAULT_MODULATION) -> np.ndarray:
    """Dispatch symbol values to the hard-decision demapper of ``modulation``."""
    family = normalize_modulation(modulation)
    if family == "bpsk":
        return (np.real(np.asarray(symbols)) < 0.0).astype(np.uint8)
    return qpsk_demap(symbols)
