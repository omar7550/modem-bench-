"""Self-contained truth-aided reference receiver for the BPSK and QPSK families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from .framing import QPSK_PHASE_AMBIGUITY_FOLD, demap_symbols, normalize_modulation


def rrc_taps(sps: int, beta: float) -> np.ndarray:
    """Unit-energy 12-symbol-span RRC pulse; t=0 and |t|=1/(4*beta) limits taken explicitly."""
    if int(sps) != sps or sps <= 0:
        raise ValueError("sps must be a positive integer")
    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in (0, 1]")

    t = np.arange(-6 * int(sps), 6 * int(sps) + 1, dtype=np.float64) / float(sps)
    taps = np.empty(t.shape, dtype=np.float64)
    at_zero = np.isclose(t, 0.0, rtol=0.0, atol=np.finfo(np.float64).eps)
    at_singularity = np.isclose(
        np.abs(t), 1.0 / (4.0 * beta), rtol=0.0, atol=8.0 * np.finfo(np.float64).eps
    )
    regular = ~(at_zero | at_singularity)

    tr = t[regular]
    taps[regular] = (
        np.sin(np.pi * tr * (1.0 - beta))
        + 4.0 * beta * tr * np.cos(np.pi * tr * (1.0 + beta))
    ) / (np.pi * tr * (1.0 - (4.0 * beta * tr) ** 2))
    taps[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    taps[at_singularity] = (beta / np.sqrt(2.0)) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
        + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
    )
    return taps / np.sqrt(np.sum(taps * taps))


def _local_interpolate(filtered: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Evaluate ``filtered`` at fractional positions using the locked 63-tap equation."""
    samples = np.asarray(filtered)
    times = np.asarray(positions, dtype=np.float64)
    if samples.ndim != 1 or times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("invalid interpolation arguments")
    output = np.empty(times.size, dtype=np.complex128)
    k = np.arange(63, dtype=np.int64)
    window = np.hanning(63)
    for index, time in enumerate(times):
        n0 = int(np.floor(time))
        mu_i = float(time - n0)
        kernel = window * np.sinc(k.astype(np.float64) - 31.0 - mu_i)
        kernel /= np.sum(kernel)
        source_indices = n0 - 31 + k
        valid = (source_indices >= 0) & (source_indices < samples.size)
        output[index] = np.sum(kernel[valid] * samples[source_indices[valid]])
    return output


def recover_symbols(
    iq: np.ndarray,
    fs: float,
    *,
    sps: int,
    beta: float,
    offset: int,
    symbol_count: int,
    cfo_hz: float = 0.0,
    phase_rad: float = 0.0,
    amplitude: float = 1.0,
    timing_mu: float = 0.0,
    fd_group_delay: int = 0,
) -> np.ndarray:
    """Normalize, derotate, matched-filter, and interpolate symbol values."""
    samples = np.asarray(iq)
    if (
        samples.ndim != 1
        or not np.isfinite(fs)
        or fs <= 0.0
        or isinstance(offset, bool)
        or int(offset) != offset
        or offset < 0
        or isinstance(symbol_count, bool)
        or int(symbol_count) != symbol_count
        or symbol_count < 0
        or not np.isfinite(cfo_hz)
        or not np.isfinite(phase_rad)
        or not np.isfinite(amplitude)
        or amplitude <= 0.0
        or not np.isfinite(timing_mu)
        or not 0.0 <= timing_mu < 1.0
        or fd_group_delay not in (0, 31)
    ):
        raise ValueError("invalid scalar/array receiver arguments")
    taps = rrc_taps(sps, beta)
    n = np.arange(samples.size, dtype=np.float64)
    normalized = samples.astype(np.complex128, copy=True) / float(amplitude)
    normalized *= np.exp(-1j * (2.0 * np.pi * float(cfo_hz) * n / float(fs) + float(phase_rad)))
    filtered = signal.convolve(normalized, taps, mode="full", method="direct")
    positions = (
        int(offset)
        + (taps.size - 1)
        + int(fd_group_delay)
        + float(timing_mu)
        + np.arange(int(symbol_count), dtype=np.float64) * int(sps)
    )
    if positions.size and positions[-1] > filtered.size - 1 + 31:
        raise ValueError("IQ capture is too short for the declared frame")
    return _local_interpolate(filtered, positions)


#: Exact unit constants: np.exp gives 6.12e-17+1j for a quarter turn, enough to move a
#: symbol sitting exactly on a decision boundary.
_QUARTER_TURN_CONJUGATE = (1 + 0j, -1j, -1 + 0j, 1j)


@dataclass(frozen=True)
class PhaseAmbiguity:
    """Which quarter turn the sync word selected; unresolved is a verdict, not a crash."""

    rotation_index: int
    rotation_rad: float
    sync_errors: int
    runner_up_errors: int
    resolved: bool
    errors_by_index: tuple[int, ...]


def resolve_phase_ambiguity(
    symbols: np.ndarray, sync_bits: np.ndarray, *, modulation: str = "qpsk"
) -> PhaseAmbiguity:
    """Derotate by k*pi/2, demap, take the k of minimum Hamming distance to the sync word.

    Ties break to the smallest k. One +pi/2 step maps (b0,b1) -> (NOT b1, b0), so a clean
    length-2m sync scores (0, m, 2m, m): a margin of m bits regardless of sync content.
    """
    if normalize_modulation(modulation) != "qpsk":
        raise ValueError("phase-ambiguity resolution is defined for QPSK only")
    values = np.asarray(symbols)
    known_sync = np.asarray(sync_bits, dtype=np.uint8)
    if values.ndim != 1 or known_sync.ndim != 1 or known_sync.size == 0:
        raise ValueError("invalid scalar/array receiver arguments")
    if np.any(known_sync > 1):
        raise ValueError("sync_bits must contain only 0 and 1")
    needed = -(-known_sync.size // 2)
    if values.size < needed:
        raise ValueError("symbol run is too short to carry the sync word")
    head = values[:needed]
    errors = tuple(
        int(
            np.count_nonzero(
                demap_symbols(head * _QUARTER_TURN_CONJUGATE[k], "qpsk")[: known_sync.size]
                != known_sync
            )
        )
        for k in range(QPSK_PHASE_AMBIGUITY_FOLD)
    )
    winner = int(np.argmin(errors))
    runner_up = min(value for index, value in enumerate(errors) if index != winner)
    return PhaseAmbiguity(
        rotation_index=winner,
        rotation_rad=winner * np.pi / 2.0,
        sync_errors=errors[winner],
        runner_up_errors=runner_up,
        resolved=errors[winner] < runner_up,
        errors_by_index=errors,
    )


def decode(
    iq: np.ndarray,
    fs: float,
    *,
    sps: int,
    beta: float,
    offset: int,
    sync_bits: np.ndarray,
    payload_len: int,
    cfo_hz: float = 0.0,
    phase_rad: float = 0.0,
    amplitude: float = 1.0,
    timing_mu: float = 0.0,
    fd_group_delay: int = 0,
) -> np.ndarray:
    """Decode a manifest-truth-aligned frame; every output bit comes from ``iq``."""
    known_sync = np.asarray(sync_bits)
    if known_sync.ndim != 1 or isinstance(payload_len, bool) or int(payload_len) != payload_len:
        raise ValueError("invalid scalar/array receiver arguments")
    frame_bit_count = known_sync.size + 16 + int(payload_len) * 8 + 32
    symbols = recover_symbols(
        iq,
        fs,
        sps=sps,
        beta=beta,
        offset=offset,
        symbol_count=frame_bit_count,
        cfo_hz=cfo_hz,
        phase_rad=phase_rad,
        amplitude=amplitude,
        timing_mu=timing_mu,
        fd_group_delay=fd_group_delay,
    )
    return (np.real(symbols) < 0.0).astype(np.uint8)


def decode_qpsk(
    iq: np.ndarray,
    fs: float,
    *,
    sps: int,
    beta: float,
    offset: int,
    sync_bits: np.ndarray,
    payload_len: int,
    cfo_hz: float = 0.0,
    phase_rad: float = 0.0,
    amplitude: float = 1.0,
    timing_mu: float = 0.0,
    fd_group_delay: int = 0,
) -> np.ndarray:
    """Decode a QPSK frame, resolving the fourfold carrier-phase ambiguity first.

    A sibling of decode, not a branch in it: the oracle materializer extracts decode's
    source text with a pinned allowlist, so decode must stay byte-frozen.
    """
    known_sync = np.asarray(sync_bits)
    if known_sync.ndim != 1 or isinstance(payload_len, bool) or int(payload_len) != payload_len:
        raise ValueError("invalid scalar/array receiver arguments")
    frame_bit_count = known_sync.size + 16 + int(payload_len) * 8 + 32
    if frame_bit_count % 2:    known_sync = np.asarray(sync_bits)
    if known_sync.ndim != 1 or isinstance(payload_len, bool) or int(payload_len) != payload_len:
        raise ValueError("invalid scalar/array receiver arguments")
    frame_bit_count = known_sync.size + 16 + int(payload_len) * 8 + 32
    if frame_bit_count % 2:
        raise ValueError("frame bit count is not a whole number of QPSK symbols")
    symbols = recover_symbols(
        iq,
        fs,
        sps=sps,
        beta=beta,
        offset=offset,
        symbol_count=frame_bit_count // 2,
        cfo_hz=cfo_hz,
        phase_rad=phase_rad,
        amplitude=amplitude,
        timing_mu=timing_mu,
        fd_group_delay=fd_group_delay,
    )
    ambiguity = resolve_phase_ambiguity(symbols, known_sync)
    return demap_symbols(symbols * _QUARTER_TURN_CONJUGATE[ambiguity.rotation_index], "qpsk")
