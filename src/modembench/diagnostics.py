"""Internal-only, truth-aided stage diagnostics for the reference receiver."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.signal as signal

from .framing import bpsk_symbols
from .impairments import fractional_delay_taps
from .reference_rx import recover_symbols, rrc_taps


DIAGNOSTIC_THRESHOLDS = {
    "signal_present_min_db": 3.0,
    "gain_error_max_abs_db": 3.0,
    "cfo_residual_max_abs_rad_per_symbol": 0.03,
    "phase_error_max_abs_rad": 0.25,
    "timing_sync_evm_max_rms": 0.35,
    "sync_hamming_max": 8,
    "payload_ber_max": 0.0,
}


def _ideal_unit_capture(
    frame_bits: np.ndarray,
    *,
    sps: int,
    beta: float,
    offset: int,
    timing_mu: float,
    fd_group_delay: int,
) -> np.ndarray:
    shaped = signal.upfirdn(rrc_taps(sps, beta), bpsk_symbols(frame_bits), up=sps)
    clean = np.concatenate(
        (np.zeros(offset, dtype=np.complex128), shaped.astype(np.complex128, copy=False))
    )
    if fd_group_delay == 31:
        return np.convolve(clean, fractional_delay_taps(timing_mu), mode="full")
    if fd_group_delay != 0:
        raise ValueError("fd_group_delay must be 0 or 31")
    return clean


def _stage(name: str, passed: bool, metrics: dict[str, Any], threshold: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "metrics": metrics, "threshold": threshold}


def diagnose(
    iq: np.ndarray,
    fs: float,
    *,
    sps: int,
    beta: float,
    offset: int,
    frame_bits: np.ndarray,
    sync_len: int,
    payload_len: int,
    decoded_bits: np.ndarray,
    evaluation: dict[str, dict[str, Any]],
    cfo_hz: float = 0.0,
    phase_rad: float = 0.0,
    amplitude: float = 1.0,
    timing_mu: float = 0.0,
    fd_group_delay: int = 0,
) -> dict[str, Any]:
    """Measure eight ordered stages and attribute the first threshold crossing."""
    observed = np.asarray(iq)
    truth = np.asarray(frame_bits, dtype=np.uint8)
    decoded = np.asarray(decoded_bits, dtype=np.uint8)
    if observed.ndim != 1 or truth.ndim != 1 or decoded.ndim != 1:
        raise ValueError("diagnostic arrays must be one-dimensional")
    if sync_len <= 0 or payload_len <= 0 or truth.size < sync_len + 16 + payload_len * 8 + 32:
        raise ValueError("invalid diagnostic framing truth")

    unit = _ideal_unit_capture(
        truth,
        sps=sps,
        beta=beta,
        offset=offset,
        timing_mu=timing_mu,
        fd_group_delay=fd_group_delay,
    )
    if unit.size != observed.size:
        raise ValueError("diagnostic truth waveform length differs from capture")
    unit_packet_power = float(np.mean(np.abs(unit[offset:]) ** 2))
    if unit_packet_power <= 0.0:
        raise ValueError("diagnostic template has no energy")
    epsilon = np.finfo(np.float64).tiny
    packet_power = float(np.mean(np.abs(observed[offset:]) ** 2))
    noise_power = float(np.mean(np.abs(observed[:offset]) ** 2)) if offset else 0.0
    signal_power = max(0.0, packet_power - noise_power)
    signal_to_noise_db = float(
        10.0 * np.log10(max(signal_power, epsilon) / max(noise_power, epsilon))
    )
    measured_amplitude = float(np.sqrt(signal_power / unit_packet_power))
    gain_ratio = measured_amplitude / float(amplitude)
    gain_error_db = float(20.0 * np.log10(max(gain_ratio, epsilon)))

    symbols = recover_symbols(
        observed,
        fs,
        sps=sps,
        beta=beta,
        offset=offset,
        symbol_count=int(truth.size),
        cfo_hz=cfo_hz,
        phase_rad=phase_rad,
        amplitude=amplitude,
        timing_mu=timing_mu,
        fd_group_delay=fd_group_delay,
    )
    expected_sync = bpsk_symbols(truth[:sync_len]).astype(np.complex128)
    received_sync = symbols[:sync_len]
    phase_trace = np.unwrap(np.angle(received_sync * expected_sync))
    symbol_axis = np.arange(sync_len, dtype=np.float64)
    slope, intercept = np.polyfit(symbol_axis, phase_trace, 1)
    residual_rotation = float(slope)
    constant_phase_error = float(np.arctan2(np.sin(intercept), np.cos(intercept)))
    phase_corrected = received_sync * np.exp(-1j * (slope * symbol_axis + intercept))
    sync_gain = np.vdot(expected_sync, phase_corrected) / float(sync_len)
    if abs(sync_gain) <= epsilon:
        sync_evm = float("inf")
    else:
        normalized_sync = phase_corrected / sync_gain
        sync_evm = float(
            np.linalg.norm(normalized_sync - expected_sync) / np.linalg.norm(expected_sync)
        )

    available_sync = min(sync_len, decoded.size)
    sync_hamming = int(np.count_nonzero(decoded[:available_sync] != truth[:available_sync]))
    sync_hamming += sync_len - available_sync
    payload_start = sync_len + 16
    payload_bits = payload_len * 8
    available_payload = max(0, min(payload_bits, decoded.size - payload_start))
    payload_errors = int(
        np.count_nonzero(
            decoded[payload_start : payload_start + available_payload]
            != truth[payload_start : payload_start + available_payload]
        )
    )
    payload_ber = float(
        (payload_errors + 0.5 * (payload_bits - available_payload)) / payload_bits
    )
    crc_pass = bool(evaluation["feedback"]["crc_pass"])
    packet_success = bool(evaluation["internal"]["packet_success"])

    thresholds = DIAGNOSTIC_THRESHOLDS
    stages = [
        _stage(
            "signal_present",
            signal_to_noise_db >= thresholds["signal_present_min_db"],
            {
                "inband_signal_to_noise_floor_db": signal_to_noise_db,
                "inband_signal_power": signal_power,
                "noise_floor_power": noise_power,
            },
            {"inband_signal_to_noise_floor_db_min": thresholds["signal_present_min_db"]},
        ),
        _stage(
            "gain_normalized",
            abs(gain_error_db) <= thresholds["gain_error_max_abs_db"],
            {
                "measured_amplitude": measured_amplitude,
                "expected_amplitude": float(amplitude),
                "measured_to_expected_ratio": gain_ratio,
                "gain_error_db": gain_error_db,
            },
            {"gain_error_abs_db_max": thresholds["gain_error_max_abs_db"]},
        ),
        _stage(
            "cfo_locked",
            abs(residual_rotation) <= thresholds["cfo_residual_max_abs_rad_per_symbol"],
            {"residual_rotation_rad_per_symbol": residual_rotation},
            {
                "residual_rotation_abs_rad_per_symbol_max": thresholds[
                    "cfo_residual_max_abs_rad_per_symbol"
                ]
            },
        ),
        _stage(
            "phase_locked",
            abs(constant_phase_error) <= thresholds["phase_error_max_abs_rad"],
            {"constant_phase_error_rad": constant_phase_error},
            {"constant_phase_error_abs_rad_max": thresholds["phase_error_max_abs_rad"]},
        ),
        _stage(
            "timing_locked",
            sync_evm <= thresholds["timing_sync_evm_max_rms"],
            {"post_interpolation_sync_relative_l2_evm": sync_evm},
            {"post_interpolation_sync_relative_l2_evm_max": thresholds["timing_sync_evm_max_rms"]},
        ),
        _stage(
            "sync_found",
            sync_hamming <= thresholds["sync_hamming_max"],
            {"sync_hamming": sync_hamming},
            {"sync_hamming_max": thresholds["sync_hamming_max"]},
        ),
        _stage(
            "payload_clean",
            payload_ber <= thresholds["payload_ber_max"],
            {"aligned_payload_ber": payload_ber},
            {"aligned_payload_ber_max": thresholds["payload_ber_max"]},
        ),
        _stage("crc", crc_pass, {"crc_pass": crc_pass}, {"crc_pass_required": True}),
    ]
    attributed_stage = next((stage["name"] for stage in stages if not stage["passed"]), None)
    return {
        "internal": True,
        "attribution_method": "heuristic first-threshold crossing only; not causal",
        "thresholds_gate_results": False,
        "packet_success": packet_success,
        "attributed_stage": attributed_stage,
        "unattributed": bool(not packet_success and attributed_stage is None),
        "stages": stages,
    }


def flattened_metrics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Flatten per-stage metric objects for compact failure report rows."""
    flattened: dict[str, Any] = {}
    for stage in diagnostics["stages"]:
        for name, value in stage["metrics"].items():
            flattened[f"{stage['name']}.{name}"] = value
    return flattened
