"""The four numeric diagnostic tools; estimators pinned and characterized offline.

Reads iq.npy and the public meta.json only, never private/. Results are size-capped and
runs are call-capped so the raw sample stream cannot be reconstructed in context.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as signal

from ..sealed import IQ_ARTIFACT, META_ARTIFACT

TOOLS_POLICY_VERSION = "modembench-tools-v1"

# Per-run and per-result bounds; changing either moves the frozen-config hash.
MAX_TOOL_CALLS_PER_RUN = 24
MAX_TOOL_RESULT_BYTES = 4096

# Published waveform family: integer samples-per-symbol search grid.
SPS_MIN = 10
SPS_MAX = 40

# --- spectrum estimator constants (pinned) ------------------------------------------------
SPECTRUM_NFFT_CHOICES = (256, 512, 1024, 2048, 4096)
DEFAULT_NFFT = 512
# Smoothing width as a fraction of the transform length.
SPECTRUM_SMOOTH_DIVISOR = 64
# Band-detection threshold above the noise floor (+3 dB) and containment-window guard.
SPECTRUM_BAND_THRESHOLD = 2.0
SPECTRUM_GUARD_FRACTION = 0.35
# Occupied bandwidth is the 99% power-containment width of the noise-subtracted PSD.
SPECTRUM_CONTAINMENT = 0.99

# --- autocorrelation estimator constants (pinned) -----------------------------------------
DEFAULT_MAX_LAG = 128
MIN_MAX_LAG = 16
MAX_MAX_LAG = 512
# A local minimum shallower than this is estimation noise, not a pulse null.
NULL_MIN_PROMINENCE = 0.01
MAX_REPORTED_NULLS = 6

# --- amplitude histogram constants (pinned) -----------------------------------------------
DEFAULT_BINS = 32
MIN_BINS = 8
MAX_BINS = 64
# Magnitudes beyond this multiple of the RMS land in an overflow bucket.
AMPLITUDE_CLIP_RMS = 6.0

MAX_RATE_CANDIDATES = 6

_MODULE_PATH = Path(__file__).resolve()


# What each tool narrows about the hidden manifest; rendered into the characterization report.
ESTIMABLE_QUANTITIES: tuple[dict[str, str], ...] = (
    {
        "tool": "spectrum",
        "manifest_field": "impairments.cfo.applied_value",
        "quantity": "spectral centroid",
        "strength": (
            "direct estimate of the carrier frequency offset; median error under 2 Hz, p90 "
            "under 160 Hz over the dev split"
        ),
    },
    {
        "tool": "spectrum",
        "manifest_field": "impairments.awgn.noise_variance_per_component",
        "quantity": "noise-floor PSD = 2 * sigma^2 / sample_rate",
        "strength": (
            "STRONG and previously unlisted: the median of the smoothed PSD recovers the "
            "AWGN noise power to a median 0.5% relative error, so the noise variance is "
            "effectively published. Combined with the histogram's RMS it bounds the "
            "in-record SNR, though not the manifest's per-symbol Es/N0, which also depends "
            "on the unpublished burst duty cycle"
        ),
    },
    {
        "tool": "spectrum",
        "manifest_field": "waveform.rrc_beta",
        "quantity": "occupied bandwidth / estimated symbol rate, minus one",
        "strength": (
            "NONE, measured. The earlier claim that this 'resolves beta once sps is known' "
            "is false: over the dev split the implied roll-off correlates with the true one "
            "at r = 0.25 and its mean absolute error (0.126) exceeds beta's own spread "
            "(sd 0.092), so the population mean is the better estimator. Kept in the table "
            "as a checked-in negative result"
        ),
    },
    {
        "tool": "symbol_period_statistic",
        "manifest_field": "waveform.sps",
        "quantity": "first signal-autocorrelation null lag",
        "strength": (
            "coarse period estimate: exact on 12 of 40 dev captures, within +/-2 samples on "
            "38 of 40. Weaker than symbol_rate_candidates on the same parameter"
        ),
    },
    {
        "tool": "symbol_period_statistic",
        "manifest_field": "waveform.sps",
        "quantity": "first squared-envelope autocorrelation null lag",
        "strength": (
            "NONE as a period estimate, measured. It is a different statistic: the first "
            "envelope null equals the symbol period on 0 of 40 dev captures and lands within "
            "+/-2 samples on 2 of 40, sitting at roughly 0.69 of the period. Reported as a "
            "shape cross-check with the published error, not as a period"
        ),
    },
    {
        "tool": "amplitude_histogram",
        "manifest_field": "impairments.amplitude.applied_value",
        "quantity": "reported RMS scale",
        "strength": (
            "the gain, up to the unknown noise power; derivable by the receiver from iq.npy "
            "unaided, so this narrows nothing the sandbox does not already have"
        ),
    },
    {
        "tool": "symbol_rate_candidates",
        "manifest_field": "waveform.sps",
        "quantity": "ranked integer samples-per-symbol",
        "strength": (
            "SUBSTANTIAL: reduces a 31-way discrete choice to a ranked shortlist, top-1 "
            "correct on 40 of 40 dev captures; the tool a tools-ablation arm removes"
        ),
    },
    {
        "tool": "symbol_rate_candidates + the capture's own length",
        "manifest_field": "framing.payload_length_bytes",
        "quantity": "upper bound only: floor(capture_samples / sps) symbols minus the framing overhead",
        "strength": (
            "BOUND, NOT A DETERMINATION. The earlier claim that capture length, sps and "
            "pulse_span_symbols 'determine the frame symbol count, hence the payload length' "
            "is false: the burst starts at an unpublished offset into the record (1738 to "
            "19598 samples over the dev split), so capture_samples/sps bounds the frame "
            "symbol count from above and determines nothing. Measured over the dev split the "
            "bound leaves a median of 90 of the 97 possible payload lengths feasible and "
            "binds at all (excludes any length) on only 22 of 40 captures. Note the span is "
            "genuinely shared -- the published pulse_span_symbols equals the manifest's "
            "rrc_span_symbols, 12 on every capture -- so it is the offset, not the span, "
            "that breaks the determination. Recorded so the calibration neither mistakes this for "
            "model capability nor for a leak"
        ),
    },
)


class ToolError(RuntimeError):
    """A tool contract violation: unknown tool, or a result that cannot be bounded."""


def tools_sha256() -> str:
    """Content address of the instrument: policy version plus this module's source."""
    digest = sha256()
    digest.update(TOOLS_POLICY_VERSION.encode("utf-8"))
    digest.update(_MODULE_PATH.read_bytes())
    return digest.hexdigest()


def _num(value: Any) -> float | None:
    """Round to six significant digits; non-finite values become None."""
    number = float(value)
    if not math.isfinite(number):
        return None
    return float(f"{number:.6g}")


@dataclass(frozen=True)
class CaptureSignal:
    """One capture's public inputs. Nothing here comes from ``private/``."""

    capture_id: str
    sample_rate_hz: float
    samples: np.ndarray


def load_capture(capture_dir: str | os.PathLike[str]) -> CaptureSignal:
    """Load ``iq.npy`` and the public ``meta.json``. No protected artifact is opened."""
    directory = Path(capture_dir)
    meta_raw = (directory / META_ARTIFACT).read_text(encoding="utf-8")
    meta = json.loads(meta_raw)
    if not isinstance(meta, dict):
        raise ToolError("public metadata is not a JSON object")
    sample_rate = meta.get("sample_rate_hz")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise ToolError("public sample_rate_hz is invalid")
    rate = float(sample_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ToolError("public sample_rate_hz is invalid")
    samples = np.load(directory / IQ_ARTIFACT, allow_pickle=False)
    if samples.ndim != 1 or samples.size < 64:
        raise ToolError("capture IQ is not a usable 1-D record")
    capture_id = meta.get("capture_id")
    return CaptureSignal(
        capture_id=str(capture_id) if isinstance(capture_id, str) else "",
        sample_rate_hz=rate,
        samples=np.asarray(samples, dtype=np.complex128),
    )


def _validate_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{name} must be an integer in [{low}, {high}]")
    if isinstance(value, float) and not value.is_integer():
        raise ToolError(f"{name} must be an integer in [{low}, {high}]")
    number = int(value)
    if not low <= number <= high:
        raise ToolError(f"{name} must be an integer in [{low}, {high}]")
    return number


def _smoothing_width(nfft: int) -> int:
    width = max(3, nfft // SPECTRUM_SMOOTH_DIVISOR)
    return width if width % 2 else width + 1


def _welch_psd(capture: CaptureSignal, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD, frequency-ordered, circularly smoothed."""
    nperseg = min(nfft, capture.samples.size)
    frequencies, power = signal.welch(
        capture.samples,
        fs=capture.sample_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        return_onesided=False,
        detrend=False,
        scaling="density",
    )
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    power = power[order]
    width = _smoothing_width(nfft)
    half = width // 2
    padded = np.concatenate((power[-half:], power, power[:half]))
    smoothed = np.convolve(padded, np.ones(width) / width, mode="valid")
    return frequencies, smoothed


def _widest_run(mask: np.ndarray) -> tuple[int, int]:
    """Start index and length of the widest circular run of ``True``."""
    size = mask.size
    if not mask.any():
        return 0, 0
    if mask.all():
        return 0, size
    doubled = np.concatenate((mask, mask))
    best_start, best_length = 0, 0
    index = 0
    while index < 2 * size:
        if not doubled[index]:
            index += 1
            continue
        stop = index
        while stop < 2 * size and doubled[stop]:
            stop += 1
        if stop - index > best_length:
            best_start, best_length = index, stop - index
        index = stop
    return best_start % size, min(best_length, size)


def spectrum(capture: CaptureSignal, *, nfft: int = DEFAULT_NFFT) -> dict[str, Any]:
    """Occupied bandwidth, band edges, spectral centroid and noise floor.

    Pinned estimator: smooth the two-sided Welch PSD over nfft/64 bins, take the median as
    the noise floor, detect the widest run above threshold, widen by the guard fraction,
    then measure the 99% containment edges and centroid of the noise-subtracted PSD.
    """
    nfft = _validate_int(nfft, "nfft", min(SPECTRUM_NFFT_CHOICES), max(SPECTRUM_NFFT_CHOICES))
    if nfft not in SPECTRUM_NFFT_CHOICES:
        raise ToolError(f"nfft must be one of {list(SPECTRUM_NFFT_CHOICES)}")
    frequencies, power = _welch_psd(capture, nfft)
    noise_floor = float(np.median(power))
    if not math.isfinite(noise_floor) or noise_floor <= 0.0:
        raise ToolError("noise floor is not estimable for this capture")
    start, length = _widest_run(power > noise_floor * SPECTRUM_BAND_THRESHOLD)
    if length == 0:
        raise ToolError("no band exceeds the detection threshold; the capture may be empty")
    resolution = float(frequencies[1] - frequencies[0])
    core = (np.arange(start, start + length) % frequencies.size).astype(int)
    guard = SPECTRUM_GUARD_FRACTION * length * resolution
    low_edge = float(frequencies[core[0]]) - guard
    high_edge = float(frequencies[core[-1]]) + guard
    window = (frequencies >= low_edge) & (frequencies <= high_edge)
    band_frequencies = frequencies[window]
    # Clip at zero: bins below the noise floor would make the cumulative sum non-monotone,
    # and searchsorted silently returns a meaningless index on an unsorted array.
    excess = np.clip(power[window] - noise_floor, 0.0, None)
    total = float(np.sum(excess))
    if band_frequencies.size < 3 or total <= 0.0:
        raise ToolError("in-band excess power is not estimable for this capture")
    cumulative = np.cumsum(excess) / total
    tail = (1.0 - SPECTRUM_CONTAINMENT) / 2.0
    low_index = min(int(np.searchsorted(cumulative, tail)), band_frequencies.size - 1)
    high_index = min(int(np.searchsorted(cumulative, 1.0 - tail)), band_frequencies.size - 1)
    band_low = float(band_frequencies[low_index])
    band_high = float(band_frequencies[high_index])
    centroid = float(np.sum(band_frequencies * excess) / total)
    return {
        "nfft": nfft,
        "resolution_hz": _num(resolution),
        "noise_floor_psd": _num(noise_floor),
        "peak_to_noise_floor_db": _num(10.0 * math.log10(float(power.max()) / noise_floor)),
        "band_low_hz": _num(band_low),
        "band_high_hz": _num(band_high),
        "occupied_bandwidth_hz": _num(band_high - band_low),
        "spectral_centroid_hz": _num(centroid),
        "containment_fraction": SPECTRUM_CONTAINMENT,
        "note": (
            "occupied bandwidth is the 99% power-containment width of the noise-subtracted "
            "PSD. It is a coarse cross-check on the symbol rate only: measured against truth "
            "over the development split it carries no usable information about the pulse's "
            "excess-bandwidth factor, so do not invert it for the roll-off"
        ),
    }


def _normalized_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """``|R(tau)| / |R(0)|`` for tau in [0, max_lag], via the biased FFT estimator."""
    centered = values - values.mean()
    size = centered.size
    length = 1 << int(math.ceil(math.log2(2 * size)))
    spectrum_values = np.fft.fft(centered, length)
    correlation = np.fft.ifft(spectrum_values * np.conj(spectrum_values))[: max_lag + 1]
    magnitude = np.abs(correlation)
    zero = float(magnitude[0])
    if zero <= 0.0:
        raise ToolError("autocorrelation is degenerate; the capture may be constant")
    return magnitude / zero


def _nulls(curve: np.ndarray) -> list[dict[str, Any]]:
    """Local minima of ``curve[1:]`` ranked by prominence, reported in lag order."""
    indices, properties = signal.find_peaks(-curve[1:], prominence=NULL_MIN_PROMINENCE)
    if indices.size == 0:
        return []
    lags = indices + 1
    prominences = properties["prominences"]
    keep = np.argsort(-prominences)[:MAX_REPORTED_NULLS]
    ordered = sorted(int(lags[index]) for index in keep)
    lookup = {int(lag): float(prom) for lag, prom in zip(lags, prominences)}
    return [
        {
            "lag_samples": lag,
            "value": _num(curve[lag]),
            "prominence": _num(lookup[lag]),
        }
        for lag in ordered
    ]


def symbol_period_statistic(
    capture: CaptureSignal, *, max_lag: int = DEFAULT_MAX_LAG
) -> dict[str, Any]:
    """Autocorrelation nulls of the signal (first null estimates the symbol period) and of
    its squared envelope ``|x|^2`` (a shape cross-check, not a period estimate)."""
    max_lag = _validate_int(max_lag, "max_lag", MIN_MAX_LAG, MAX_MAX_LAG)
    if max_lag >= capture.samples.size // 4:
        max_lag = max(MIN_MAX_LAG, capture.samples.size // 4)
    signal_curve = _normalized_autocorrelation(capture.samples, max_lag)
    envelope_curve = _normalized_autocorrelation(
        np.abs(capture.samples) ** 2, max_lag
    )
    signal_nulls = _nulls(signal_curve)
    envelope_nulls = _nulls(envelope_curve)
    first = signal_nulls[0] if signal_nulls else None
    envelope_first = envelope_nulls[0] if envelope_nulls else None
    return {
        "statistic": (
            "signal_autocorrelation_nulls are the nulls of the normalized magnitude "
            "autocorrelation of x; for a root-raised-cosine pulse these fall at integer "
            "multiples of the symbol period"
        ),
        "envelope_statistic": (
            "envelope_autocorrelation_nulls are the same statistic computed on |x|^2 and do "
            "NOT fall at multiples of the symbol period: measured over the development split "
            "the first one equals the symbol period on 0 of 40 captures and lands within 2 "
            "samples on 2 of 40, sitting at roughly 0.69 of it. Use it as a shape "
            "cross-check, not as a period estimate"
        ),
        "max_lag": max_lag,
        "min_prominence": NULL_MIN_PROMINENCE,
        "first_null_lag_samples": None if first is None else first["lag_samples"],
        "first_null_prominence": None if first is None else first["prominence"],
        "first_envelope_null_lag_samples": (
            None if envelope_first is None else envelope_first["lag_samples"]
        ),
        "signal_autocorrelation_nulls": signal_nulls,
        "envelope_autocorrelation_nulls": envelope_nulls,
    }


def amplitude_histogram(capture: CaptureSignal, *, bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Magnitude histogram on a grid normalized to the capture's own RMS, which is reported."""
    bins = _validate_int(bins, "bins", MIN_BINS, MAX_BINS)
    magnitude = np.abs(capture.samples)
    scale = float(np.sqrt(np.mean(magnitude * magnitude)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ToolError("capture magnitude scale is degenerate")
    normalized = magnitude / scale
    counts, edges = np.histogram(
        np.clip(normalized, 0.0, AMPLITUDE_CLIP_RMS),
        bins=bins,
        range=(0.0, AMPLITUDE_CLIP_RMS),
    )
    overflow = int(np.count_nonzero(normalized > AMPLITUDE_CLIP_RMS))
    # np.clip folded the clipped samples into the last bin; move them into the overflow.
    counts = counts.astype(np.int64)
    if overflow:
        counts[-1] = max(0, int(counts[-1]) - overflow)
    return {
        "bins": bins,
        "scale": "magnitude divided by the capture RMS magnitude",
        "rms_magnitude": _num(scale),
        "clip_rms_multiples": AMPLITUDE_CLIP_RMS,
        "bin_edges_rms_multiples": [_num(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "overflow_count": overflow,
        "sample_count": int(magnitude.size),
        "normalized_max": _num(float(normalized.max())),
    }


def symbol_rate_candidates(capture: CaptureSignal) -> dict[str, Any]:
    """Ranked integer samples-per-symbol from the cyclostationary line in ``|x|^2``
    (the classical squaring-loop / Oerder-Meyr symbol-timing line)."""
    envelope = np.abs(capture.samples) ** 2
    centered = envelope - envelope.mean()
    energy = float(np.sqrt(float(np.sum(centered * centered)) * centered.size))
    if not math.isfinite(energy) or energy <= 0.0:
        raise ToolError("squared-envelope energy is degenerate")
    index = np.arange(centered.size, dtype=np.float64)
    scored: list[tuple[float, int]] = []
    for sps in range(SPS_MIN, SPS_MAX + 1):
        line = np.abs(np.sum(centered * np.exp(-2j * np.pi * index / sps)))
        scored.append((float(line) / energy, sps))
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = scored[0][0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    candidates = [
        {
            "sps": sps,
            "symbol_rate_hz": _num(capture.sample_rate_hz / sps),
            "line_strength": _num(score),
            "confidence": _num(score / best) if best > 0 else None,
        }
        for score, sps in scored[:MAX_RATE_CANDIDATES]
    ]
    return {
        "statistic": "normalized cyclostationary line in |x|^2 at the candidate symbol rate",
        "search_grid": {"sps_min": SPS_MIN, "sps_max": SPS_MAX, "integer": True},
        "candidates": candidates,
        "margin_over_next": _num(best / runner_up) if runner_up > 0 else None,
    }


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "spectrum",
        "description": (
            "Welch power spectral density summary of the capture. Returns the noise-floor "
            "estimate, the band edges and occupied bandwidth (99% power containment of the "
            "noise-subtracted PSD), and the spectral centroid (the carrier frequency "
            "offset). The spectrum is continuous, so there is deliberately no peak list: the "
            "largest bin is a noise realization. The occupied bandwidth is a coarse "
            "cross-check on the symbol rate ONLY: measured against truth over the whole "
            "development split it carries no usable information about the pulse's "
            "excess-bandwidth factor (roll-off), so do not invert it for that parameter -- "
            "quoting the population mean of the published range beats the measurement. Call "
            "this first to bound the signal's band."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "nfft": {
                    "type": "integer",
                    "enum": list(SPECTRUM_NFFT_CHOICES),
                    "description": "Transform length. Larger resolves finer, averages less.",
                }
            },
            "required": ["nfft"],
            "additionalProperties": False,
        },
    },
    {
        "name": "symbol_period_statistic",
        "description": (
            "Nulls of the normalized magnitude autocorrelation, for the signal and for its "
            "squared envelope. For iid linear modulation the signal autocorrelation is the "
            "pulse autocorrelation, whose zero crossings fall at integer multiples of the "
            "symbol period; the first null is therefore a symbol-period estimate in samples. "
            "Peaks between nulls are low-level pulse sidelobes and are not reported. Use this "
            "when you want a period estimate that does not assume the candidate grid."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "max_lag": {
                    "type": "integer",
                    "minimum": MIN_MAX_LAG,
                    "maximum": MAX_MAX_LAG,
                    "description": "Largest lag in samples to search for nulls.",
                }
            },
            "required": ["max_lag"],
            "additionalProperties": False,
        },
    },
    {
        "name": "amplitude_histogram",
        "description": (
            "Histogram of sample magnitude on a grid normalized to the capture's own RMS "
            "magnitude, which is also returned. Use it to judge the modulation's envelope "
            "behaviour and the noise level relative to the signal; a fixed absolute grid "
            "would be uninformative because the capture gain varies widely between signals."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "bins": {
                    "type": "integer",
                    "minimum": MIN_BINS,
                    "maximum": MAX_BINS,
                    "description": "Number of histogram bins across 0..6 RMS multiples.",
                }
            },
            "required": ["bins"],
            "additionalProperties": False,
        },
    },
    {
        "name": "symbol_rate_candidates",
        "description": (
            "Ranked candidate symbol rates from the cyclostationary line in the squared "
            "envelope. Returns integer samples-per-symbol candidates in [10, 40] with the "
            "corresponding symbol rate, the line strength, and a relative confidence. The "
            "top candidate is usually but not always correct; check it against the occupied "
            "bandwidth from `spectrum` and the null spacing from `symbol_period_statistic`."
        ),
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
)

TOOL_NAMES = tuple(schema["name"] for schema in TOOL_SCHEMAS)


@dataclass(frozen=True)
class ToolResult:
    """One tool call's outcome, already serialized and already bounded."""

    name: str
    ok: bool
    payload: dict[str, Any]
    serialized: str
    error: str | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.serialized.encode("utf-8"))


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _error_result(name: str, message: str) -> ToolResult:
    payload = {"error": message}
    return ToolResult(name=name, ok=False, payload=payload, serialized=_serialize(payload), error=message)


def call_tool(name: str, arguments: dict[str, Any], capture: CaptureSignal) -> ToolResult:
    """Dispatch one tool call, returning a structured failure rather than raising.

    Oversized results are refused rather than truncated: truncated JSON is a parse error.
    """
    if name not in TOOL_NAMES:
        return _error_result(name, f"unknown tool {name!r}")
    if not isinstance(arguments, dict):
        return _error_result(name, "tool arguments must be an object")
    unexpected = set(arguments) - _ALLOWED_ARGUMENTS[name]
    if unexpected:
        return _error_result(name, f"unexpected argument(s): {sorted(unexpected)}")
    try:
        if name == "spectrum":
            payload = spectrum(capture, nfft=arguments.get("nfft", DEFAULT_NFFT))
        elif name == "symbol_period_statistic":
            payload = symbol_period_statistic(
                capture, max_lag=arguments.get("max_lag", DEFAULT_MAX_LAG)
            )
        elif name == "amplitude_histogram":
            payload = amplitude_histogram(capture, bins=arguments.get("bins", DEFAULT_BINS))
        else:
            payload = symbol_rate_candidates(capture)
    except ToolError as exc:
        return _error_result(name, str(exc))
    except (ValueError, TypeError, FloatingPointError, MemoryError) as exc:
        return _error_result(name, f"{name} failed: {type(exc).__name__}")
    serialized = _serialize(payload)
    if len(serialized.encode("utf-8")) > MAX_TOOL_RESULT_BYTES:
        return _error_result(
            name, f"result exceeds the {MAX_TOOL_RESULT_BYTES}-byte tool output cap"
        )
    return ToolResult(name=name, ok=True, payload=payload, serialized=serialized)


_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    schema["name"]: set(schema["input_schema"]["properties"]) for schema in TOOL_SCHEMAS
}


def tools_config() -> dict[str, Any]:
    """The instrument's identity, as recorded in every run record."""
    return {
        "tools_policy_version": TOOLS_POLICY_VERSION,
        "tools_sha256": tools_sha256(),
        "tool_names": list(TOOL_NAMES),
        "max_tool_calls_per_run": MAX_TOOL_CALLS_PER_RUN,
        "max_tool_result_bytes": MAX_TOOL_RESULT_BYTES,
    }
