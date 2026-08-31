"""Deterministic channel impairments and their locked draw contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np


FD_NTAPS = 63
FD_GROUP_DELAY = 31


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class ImpairmentRanges:
    """The provisional impairment ranges that the plan may later freeze."""

    timing_mu_samples: tuple[float, float] = (0.0, 1.0)
    cfo_symbol_rate_fraction: tuple[float, float] = (-0.1, 0.1)
    phase_rad: tuple[float, float] = (0.0, 2.0 * math.pi)
    amplitude: tuple[float, float] = (0.05, 1.0)
    snr_db: tuple[float, float] = (15.0, 25.0)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {key: list(value) for key, value in result.items()}

    @property
    def ranges_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class ImpairmentControl:
    """Enable one impairment and optionally replace its unconditional draw."""

    enabled: bool
    override: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "override": self.override}


@dataclass(frozen=True)
class ImpairmentConfig:
    """Full applied impairment configuration, independent of random ranges."""

    fractional_timing: ImpairmentControl
    cfo: ImpairmentControl
    phase: ImpairmentControl
    amplitude: ImpairmentControl
    awgn: ImpairmentControl
    profile: str = "custom"

    @classmethod
    def clean(cls) -> "ImpairmentConfig":
        disabled = ImpairmentControl(False)
        return cls(disabled, disabled, disabled, disabled, disabled, profile="clean")

    @classmethod
    def impaired(cls) -> "ImpairmentConfig":
        enabled = ImpairmentControl(True)
        return cls(enabled, enabled, enabled, enabled, enabled, profile="impaired")

    @classmethod
    def from_profile(cls, profile: str) -> "ImpairmentConfig":
        if profile == "clean":
            return cls.clean()
        if profile == "impaired":
            return cls.impaired()
        raise ValueError(f"unknown impairment profile: {profile}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "fractional_timing": self.fractional_timing.to_dict(),
            "cfo": self.cfo.to_dict(),
            "phase": self.phase.to_dict(),
            "amplitude": self.amplitude.to_dict(),
            "awgn": self.awgn.to_dict(),
        }

    @property
    def config_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class ImpairmentDraws:
    """Five unconditional draws in the locked order: mu, CFO, phase, A, SNR."""

    timing_mu: float
    cfo_hz: float
    phase_rad: float
    amplitude: float
    snr_db: float


@dataclass(frozen=True)
class ImpairmentResult:
    """Applied samples plus private intermediates used by independent tests."""

    samples: np.ndarray
    pre_noise: np.ndarray
    noise: np.ndarray
    manifest: dict[str, Any]
    support_stop: int
    symbol_energy: float
    noise_variance_per_component: float | None


def fractional_delay_taps(mu: float) -> np.ndarray:
    """Return the normative 63-tap Hann-windowed-sinc delay for ``mu`` samples."""
    if not np.isfinite(mu) or not 0.0 <= mu < 1.0:
        raise ValueError("fractional timing mu must be finite and in [0, 1)")
    k = np.arange(FD_NTAPS, dtype=np.float64)
    taps = np.hanning(FD_NTAPS) * np.sinc(k - FD_GROUP_DELAY - float(mu))
    dc_gain = float(np.sum(taps))
    if not np.isfinite(dc_gain) or dc_gain == 0.0:
        raise ValueError("fractional-delay FIR has invalid DC gain")
    return taps / dc_gain


def draw_impairments(
    rng: np.random.Generator,
    *,
    symbol_rate_hz: float,
    ranges: ImpairmentRanges,
) -> ImpairmentDraws:
    """Draw every candidate unconditionally in the locked locked order."""
    if not np.isfinite(symbol_rate_hz) or symbol_rate_hz <= 0.0:
        raise ValueError("symbol_rate_hz must be positive and finite")
    mu = float(rng.uniform(*ranges.timing_mu_samples))
    cfo_fraction = float(rng.uniform(*ranges.cfo_symbol_rate_fraction))
    cfo_hz = cfo_fraction * float(symbol_rate_hz)
    phase = float(rng.uniform(*ranges.phase_rad))
    amplitude = float(np.exp(rng.uniform(np.log(ranges.amplitude[0]), np.log(ranges.amplitude[1]))))
    snr_db = float(rng.uniform(*ranges.snr_db))
    return ImpairmentDraws(mu, cfo_hz, phase, amplitude, snr_db)


def _applied_entry(
    control: ImpairmentControl,
    drawn_value: float,
    neutral_value: float | None,
    *,
    unit: str,
) -> tuple[float | None, dict[str, Any]]:
    applied = neutral_value
    overridden = bool(control.enabled and control.override is not None)
    if control.enabled:
        applied = float(control.override) if control.override is not None else float(drawn_value)
    return applied, {
        "enabled": bool(control.enabled),
        "drawn_value": float(drawn_value),
        "applied_value": applied,
        "overridden": overridden,
        "unit": unit,
    }


def apply_impairments(
    clean_capture: np.ndarray,
    *,
    offset: int,
    packet_waveform_length: int,
    n_symbols: int,
    fs: float,
    sps: int,
    impairment_rng: np.random.Generator,
    noise_rng: np.random.Generator,
    config: ImpairmentConfig,
    ranges: ImpairmentRanges | None = None,
    trailing_samples: int = 0,
) -> ImpairmentResult:
    """Apply the normative impairment chain to a clean complex128 capture.

    trailing_samples only changes length arithmetic; noise spans the whole array, so the
    trailing region gets the same noise floor and the frozen path (zero) is byte-identical.
    """
    source = np.asarray(clean_capture)
    active_ranges = ranges or ImpairmentRanges()
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.number):
        raise ValueError("clean_capture must be a one-dimensional numeric array")
    if isinstance(offset, bool) or int(offset) != offset or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if int(packet_waveform_length) != packet_waveform_length or packet_waveform_length <= 0:
        raise ValueError("packet_waveform_length must be a positive integer")
    if isinstance(trailing_samples, bool) or int(trailing_samples) != trailing_samples or trailing_samples < 0:
        raise ValueError("trailing_samples must be a non-negative integer")
    if source.size != int(offset) + int(packet_waveform_length) + int(trailing_samples):
        raise ValueError(
            "clean_capture length must equal offset + packet_waveform_length + trailing_samples"
        )
    if int(n_symbols) != n_symbols or n_symbols <= 0:
        raise ValueError("n_symbols must be a positive integer")
    if int(sps) != sps or sps <= 0 or not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs and sps must define a positive symbol rate")

    draws = draw_impairments(
        impairment_rng, symbol_rate_hz=float(fs) / int(sps), ranges=active_ranges
    )
    timing_mu, timing_entry = _applied_entry(
        config.fractional_timing, draws.timing_mu, 0.0, unit="samples"
    )
    cfo_hz, cfo_entry = _applied_entry(config.cfo, draws.cfo_hz, 0.0, unit="Hz")
    phase_rad, phase_entry = _applied_entry(config.phase, draws.phase_rad, 0.0, unit="rad")
    amplitude, amplitude_entry = _applied_entry(
        config.amplitude, draws.amplitude, 1.0, unit="linear"
    )
    snr_db, awgn_entry = _applied_entry(config.awgn, draws.snr_db, None, unit="dB Es/N0")

    assert timing_mu is not None and cfo_hz is not None
    assert phase_rad is not None and amplitude is not None
    if not 0.0 <= timing_mu < 1.0:
        raise ValueError("applied timing override must be in [0, 1)")
    if not np.isfinite(cfo_hz) or not np.isfinite(phase_rad):
        raise ValueError("applied CFO and phase must be finite")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("applied amplitude must be positive and finite")
    if snr_db is not None and not np.isfinite(snr_db):
        raise ValueError("applied SNR must be finite")

    working = source.astype(np.complex128, copy=True)
    fd_group_delay = 0
    extension = 0
    if config.fractional_timing.enabled:
        working = np.convolve(working, fractional_delay_taps(timing_mu), mode="full")
        fd_group_delay = FD_GROUP_DELAY
        extension = FD_NTAPS - 1

    support_stop = int(offset) + int(packet_waveform_length) + extension
    n = np.arange(working.size, dtype=np.float64)
    working *= np.exp(1j * (2.0 * np.pi * cfo_hz * n / float(fs) + phase_rad))
    working *= amplitude
    pre_noise = working.copy()
    symbol_energy = float(np.sum(np.abs(pre_noise[int(offset) : support_stop]) ** 2) / int(n_symbols))

    noise = np.zeros(working.size, dtype=np.complex128)
    noise_variance: float | None = None
    if config.awgn.enabled:
        assert snr_db is not None
        noise_variance = symbol_energy / (2.0 * 10.0 ** (snr_db / 10.0))
        components = noise_rng.standard_normal(2 * working.size).reshape(working.size, 2)
        noise = np.sqrt(noise_variance) * (components[:, 0] + 1j * components[:, 1])
        working += noise

    timing_entry["fd_group_delay_samples"] = fd_group_delay
    awgn_entry["symbol_energy"] = symbol_energy
    awgn_entry["noise_variance_per_component"] = noise_variance
    manifest = {
        "profile": config.profile,
        "ranges": active_ranges.to_dict(),
        "ranges_hash": active_ranges.ranges_hash,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "draw_order": ["fractional_timing", "cfo", "phase", "amplitude", "awgn"],
        "fractional_timing": timing_entry,
        "cfo": cfo_entry,
        "phase": phase_entry,
        "amplitude": amplitude_entry,
        "awgn": awgn_entry,
    }
    return ImpairmentResult(
        samples=working.astype(np.dtype("<c8")),
        pre_noise=pre_noise,
        noise=noise,
        manifest=manifest,
        support_stop=support_stop,
        symbol_energy=symbol_energy,
        noise_variance_per_component=noise_variance,
    )
