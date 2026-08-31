"""Offline instrument characterization. This path may read truth; agent runs may not."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from ..sandbox.ast_gate import AST_POLICY_VERSION, check_source
from ..sealed import MANIFEST_ARTIFACT, read_private_artifact
from ..splits import DEV_SPLIT_NAME, dev_split_root
from .tools import (
    ESTIMABLE_QUANTITIES,
    SPS_MAX,
    SPS_MIN,
    TOOLS_POLICY_VERSION,
    DEFAULT_BINS,
    DEFAULT_MAX_LAG,
    DEFAULT_NFFT,
    amplitude_histogram,
    load_capture,
    spectrum,
    symbol_period_statistic,
    symbol_rate_candidates,
    tools_sha256,
)

CHARACTERIZATION_POLICY_VERSION = "modembench-characterization-v1"
CHARACTERIZATION_SCHEMA_VERSION = "1.0"


class SelfScoringError(RuntimeError):
    """A measurement was about to be scored against something other than manifest truth."""


# Sentinel making ManifestTruth unconstructable outside CaptureTruth.derive().
_MINT = object()


class ManifestTruth:
    """One value derived from a capture's manifest; the only type Measurement accepts as truth."""

    __slots__ = ("quantity", "value", "manifest_sha256")

    def __init__(
        self, quantity: str, value: float, manifest_sha256: str, *, mint: Any = None
    ) -> None:
        if mint is not _MINT:
            raise SelfScoringError(
                "a ManifestTruth may only be minted by CaptureTruth.derive(), which is "
                "handed the parsed manifest and nothing else; constructing one directly "
                "would let a tool output be scored as if it were truth"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"manifest truth for {quantity!r} is not finite")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "value", number)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - immutability
        raise AttributeError("ManifestTruth is immutable")

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ManifestTruth({self.quantity!r}, {self.value!r})"


class CaptureTruth:
    """The manifest, and the only thing allowed to mint a truth value from it.

    derive() sees only the parsed manifest, and the object is sealed before the first tool
    runs, so no truth value can be derived from an estimate.
    """

    __slots__ = ("_manifest", "_digest", "_sealed")

    def __init__(self, manifest: dict[str, Any], digest: str) -> None:
        self._manifest = manifest
        self._digest = digest
        self._sealed = False

    @property
    def manifest_sha256(self) -> str:
        return self._digest

    @property
    def sealed(self) -> bool:
        return self._sealed

    def derive(
        self, quantity: str, expression: Callable[[dict[str, Any]], float]
    ) -> ManifestTruth:
        """Mint the truth for ``quantity`` from the manifest alone."""
        if self._sealed:
            raise SelfScoringError(
                f"refusing to derive truth for {quantity!r} after the estimators have run: "
                "every truth value must be minted before any estimate exists, so that no "
                "measurement can score an estimator against itself"
            )
        return ManifestTruth(
            quantity, expression(self._manifest), self._digest, mint=_MINT
        )

    def seal(self) -> "CaptureTruth":
        """Close the window in which truth may be minted. Called before the tools run."""
        self._sealed = True
        return self


@dataclass(frozen=True)
class Measurement:
    """One reported quantity beside the ManifestTruth it estimates."""

    tool: str
    quantity: str
    estimate: float | None
    truth: ManifestTruth
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.truth, ManifestTruth):
            raise SelfScoringError(
                f"the truth side of {self.tool}.{self.quantity} is a "
                f"{type(self.truth).__name__}, not a ManifestTruth; only a value minted from "
                "private/manifest.json may be scored against"
            )

    @property
    def truth_value(self) -> float:
        return self.truth.value

    @property
    def absolute_error(self) -> float | None:
        if self.estimate is None:
            return None
        return float(self.estimate) - self.truth.value

    @property
    def relative_error(self) -> float | None:
        error = self.absolute_error
        if error is None or self.truth.value == 0.0:
            return None
        return error / self.truth.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "quantity": self.quantity,
            "estimate": self.estimate,
            "truth": self.truth.value,
            "unit": self.unit,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
        }


def _truth(capture_dir: str | os.PathLike[str], token: Any = None) -> CaptureTruth:
    """Read one capture's manifest through the single sanctioned reader."""
    raw = read_private_artifact(Path(capture_dir), MANIFEST_ARTIFACT, token)
    payload = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("capture manifest is not a JSON object")
    return CaptureTruth(manifest, sha256(payload).hexdigest())


def expected_rms_magnitude(manifest: dict[str, Any]) -> float:
    """Closed-form RMS magnitude from the manifest: sqrt(Es * n / L + 2 * sigma^2)."""
    waveform = manifest["waveform"]
    awgn = manifest["impairments"]["awgn"]
    symbols = float(manifest["framing"]["frame_bit_count"])
    length = float(waveform["support_stop_samples"])
    signal_power = float(awgn["symbol_energy"]) * symbols / length
    noise_power = 2.0 * float(awgn["noise_variance_per_component"])
    return math.sqrt(signal_power + noise_power)


def expected_noise_psd(manifest: dict[str, Any]) -> float:
    """Two-sided noise power spectral density: total noise power spread over the sample rate."""
    awgn = manifest["impairments"]["awgn"]
    rate = float(manifest["waveform"]["sample_rate_hz"])
    return 2.0 * float(awgn["noise_variance_per_component"]) / rate


def characterize_capture(
    capture_dir: str | os.PathLike[str],
    *,
    nfft: int = DEFAULT_NFFT,
    max_lag: int = DEFAULT_MAX_LAG,
    bins: int = DEFAULT_BINS,
    token: Any = None,
) -> dict[str, Any]:
    """Run all four tools on one capture and score every reported quantity against truth.

    Order is load-bearing: truth is minted and sealed before any estimator runs.
    """
    truth = _truth(capture_dir, token)
    sps = truth.derive("waveform.sps", lambda m: float(m["waveform"]["sps"]))
    beta = truth.derive("waveform.rrc_beta", lambda m: float(m["waveform"]["rrc_beta"]))
    symbol_rate = truth.derive(
        "waveform.symbol_rate_hz",
        lambda m: float(m["waveform"]["sample_rate_hz"]) / float(m["waveform"]["sps"]),
    )
    occupied_bandwidth = truth.derive(
        "(1 + rrc_beta) * symbol_rate_hz",
        lambda m: (1.0 + float(m["waveform"]["rrc_beta"]))
        * float(m["waveform"]["sample_rate_hz"])
        / float(m["waveform"]["sps"]),
    )
    cfo = truth.derive(
        "impairments.cfo.applied_value",
        lambda m: float(m["impairments"]["cfo"]["applied_value"]),
    )
    noise_psd = truth.derive(
        "2 * noise_variance_per_component / sample_rate_hz", expected_noise_psd
    )
    rms = truth.derive("closed-form RMS magnitude", expected_rms_magnitude)
    snr_db = truth.derive(
        "impairments.awgn.applied_value",
        lambda m: float(m["impairments"]["awgn"]["applied_value"]),
    )
    amplitude = truth.derive(
        "impairments.amplitude.applied_value",
        lambda m: float(m["impairments"]["amplitude"]["applied_value"]),
    )
    # Nothing may be minted past this line; the estimators run below it.
    truth.seal()

    capture = load_capture(capture_dir)
    spectrum_result = spectrum(capture, nfft=nfft)
    period_result = symbol_period_statistic(capture, max_lag=max_lag)
    histogram_result = amplitude_histogram(capture, bins=bins)
    rate_result = symbol_rate_candidates(capture)

    top = rate_result["candidates"][0]
    ranked = [entry["sps"] for entry in rate_result["candidates"]]
    true_sps = int(sps.value)
    # Roll-off implied by inverting occupied_bandwidth ~ (1 + beta) * Rs; measured so the
    # negative result is checked in.
    implied_beta: float | None = None
    if spectrum_result["occupied_bandwidth_hz"] is not None and top["symbol_rate_hz"]:
        implied_beta = float(spectrum_result["occupied_bandwidth_hz"]) / float(
            top["symbol_rate_hz"]
        ) - 1.0
    measurements = [
        Measurement(
            "spectrum",
            "occupied_bandwidth_hz",
            spectrum_result["occupied_bandwidth_hz"],
            occupied_bandwidth,
            "Hz",
        ),
        Measurement(
            "spectrum",
            "spectral_centroid_hz",
            spectrum_result["spectral_centroid_hz"],
            cfo,
            "Hz",
        ),
        Measurement(
            "spectrum",
            "noise_floor_psd",
            spectrum_result["noise_floor_psd"],
            noise_psd,
            "power/Hz",
        ),
        # Checked-in negative result: no tool estimates beta.
        Measurement(
            "spectrum",
            "beta_from_occupied_bandwidth",
            implied_beta,
            beta,
            "dimensionless",
        ),
        Measurement(
            "symbol_period_statistic",
            "first_null_lag_samples",
            period_result["first_null_lag_samples"],
            sps,
            "samples",
        ),
        # Not a symbol-period estimate; its error row is what says so.
        Measurement(
            "symbol_period_statistic",
            "first_envelope_null_lag_samples",
            period_result["first_envelope_null_lag_samples"],
            sps,
            "samples",
        ),
        Measurement(
            "amplitude_histogram",
            "rms_magnitude",
            histogram_result["rms_magnitude"],
            rms,
            "linear",
        ),
        Measurement(
            "symbol_rate_candidates",
            "top_symbol_rate_hz",
            top["symbol_rate_hz"],
            symbol_rate,
            "Hz",
        ),
    ]
    envelope_null = period_result["first_envelope_null_lag_samples"]
    return {
        "capture_id": capture.capture_id,
        "truth": {
            "sps": true_sps,
            "rrc_beta": beta.value,
            "symbol_rate_hz": symbol_rate.value,
            "cfo_hz": cfo.value,
            "snr_db": snr_db.value,
            "amplitude": amplitude.value,
        },
        "manifest_sha256": truth.manifest_sha256,
        "measurements": [item.as_dict() for item in measurements],
        "sps_rank": (ranked.index(true_sps) if true_sps in ranked else None),
        "sps_top1": top["sps"] == true_sps,
        "sps_margin_over_next": rate_result["margin_over_next"],
        "first_null_lag_error": (
            None
            if period_result["first_null_lag_samples"] is None
            else int(period_result["first_null_lag_samples"]) - true_sps
        ),
        "envelope_null_lag_error": (
            None if envelope_null is None else int(envelope_null) - true_sps
        ),
        "envelope_null_period_fraction": (
            None if envelope_null is None else float(envelope_null) / float(true_sps)
        ),
        "implied_beta": implied_beta,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Sample correlation, or ``None`` when either side is constant."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "median": None, "p90_abs": None, "max_abs": None}
    ordered = sorted(values)
    magnitudes = sorted(abs(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    index = min(len(magnitudes) - 1, int(math.ceil(0.9 * len(magnitudes))) - 1)
    return {
        "n": len(values),
        "median": median,
        "p90_abs": magnitudes[max(0, index)],
        "max_abs": magnitudes[-1],
    }


# Ordinary receiver idioms; several reasonable ones are rejected, and every rejection
# scores as an agent failure.
AST_PROBES: tuple[dict[str, str], ...] = (
    {
        "name": "numpy_and_scipy_only",
        "note": "the intended shape: numpy plus scipy.signal, module-level function",
        "source": (
            "import numpy as np\n"
            "import scipy.signal as signal\n"
            "def receive(iq, sample_rate):\n"
            "    taps = signal.firwin(31, 0.2)\n"
            "    y = signal.convolve(iq, taps, mode='same')\n"
            "    return (np.real(y[::19]) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "stdlib_math",
        "note": "`import math` for pi and sqrt: the single most natural import to reach for",
        "source": (
            "import math\n"
            "import numpy as np\n"
            "def receive(iq, sample_rate):\n"
            "    scale = math.sqrt(2.0)\n"
            "    return (np.real(iq[::19]) * scale < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "class_with_constructor",
        "note": (
            "a class holding carrier-loop state; `__init__` is a dunder *definition*, not a "
            "dunder read, and is accepted from `dunderdef1` on"
        ),
        "source": (
            "import numpy as np\n"
            "class Costas:\n"
            "    def __init__(self):\n"
            "        self.phase = 0.0\n"
            "    def step(self, sample):\n"
            "        self.phase = self.phase + 0.01\n"
            "        return sample * np.exp(-1j * self.phase)\n"
            "def receive(iq, sample_rate):\n"
            "    loop = Costas()\n"
            "    out = np.array([loop.step(v) for v in iq[::19]])\n"
            "    return (np.real(out) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "underscore_prefixed_attribute",
        "note": "`state._value`: any attribute beginning with an underscore is refused",
        "source": (
            "import numpy as np\n"
            "class State:\n"
            "    _value = 0.0\n"
            "def receive(iq, sample_rate):\n"
            "    s = State()\n"
            "    return (np.real(iq[::19]) + s._value < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "main_guard",
        "note": (
            "`if __name__ == '__main__':` for local testing, left in the submission; a "
            "comparison operand, not an attribute read, so accepted from `dunderdef1` on"
        ),
        "source": (
            "import numpy as np\n"
            "def receive(iq, sample_rate):\n"
            "    return (np.real(iq[::19]) < 0).astype(np.uint8)\n"
            "if __name__ == '__main__':\n"
            "    pass\n"
        ),
    },
    {
        "name": "module_level_underscore_helper",
        "note": "`def _rrc(...)` as a private module helper, then called normally",
        "source": (
            "import numpy as np\n"
            "def _rrc(sps):\n"
            "    return np.ones(sps) / sps\n"
            "def receive(iq, sample_rate):\n"
            "    taps = _rrc(19)\n"
            "    y = np.convolve(iq, taps, mode='same')\n"
            "    return (np.real(y[::19]) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "scipy_signal_from_import",
        "note": "`from scipy.signal import resample_poly`",
        "source": (
            "import numpy as np\n"
            "from scipy.signal import resample_poly\n"
            "def receive(iq, sample_rate):\n"
            "    y = resample_poly(iq, 1, 19)\n"
            "    return (np.real(y) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "numpy_fft_and_linalg",
        "note": "`np.fft` and `np.linalg` for coarse frequency and least squares",
        "source": (
            "import numpy as np\n"
            "def receive(iq, sample_rate):\n"
            "    peak = np.argmax(np.abs(np.fft.fft(iq * iq)))\n"
            "    fit = np.linalg.norm(iq[:8])\n"
            "    y = iq * np.exp(-1j * float(peak) * 0.0) / max(fit, 1e-12)\n"
            "    return (np.real(y[::19]) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "cmath_for_phase",
        "note": "`import cmath` to unwrap a phase",
        "source": (
            "import cmath\n"
            "import numpy as np\n"
            "def receive(iq, sample_rate):\n"
            "    angle = cmath.phase(complex(iq[0]))\n"
            "    return (np.real(iq[::19] * np.exp(-1j * angle)) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "getattr_dispatch",
        "note": "`getattr(signal, name)` to pick a filter by name",
        "source": (
            "import numpy as np\n"
            "import scipy.signal as signal\n"
            "def receive(iq, sample_rate):\n"
            "    fn = getattr(signal, 'firwin')\n"
            "    y = np.convolve(iq, fn(31, 0.2), mode='same')\n"
            "    return (np.real(y[::19]) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "generator_and_comprehensions",
        "note": "plain Python control flow, closures and comprehensions",
        "source": (
            "import numpy as np\n"
            "def receive(iq, sample_rate):\n"
            "    step = 19\n"
            "    picks = [iq[i] for i in range(0, iq.size, step)]\n"
            "    total = sum(abs(v) for v in picks)\n"
            "    scaled = np.array(picks) / max(total, 1e-12)\n"
            "    return (np.real(scaled) < 0).astype(np.uint8)\n"
        ),
    },
    {
        "name": "numpy_save_debug",
        "note": "`np.save` left behind from local debugging",
        "source": (
            "import numpy as np\n"
            "def receive(iq, sample_rate):\n"
            "    bits = (np.real(iq[::19]) < 0).astype(np.uint8)\n"
            "    np.save('debug.npy', bits)\n"
            "    return bits\n"
        ),
    },
)


def ast_probe_report() -> dict[str, Any]:
    """Rejection rate of the shipped static policy over ordinary receiver idioms."""
    probes = []
    rejected = 0
    for probe in AST_PROBES:
        verdict = check_source(probe["source"])
        if not verdict["ok"]:
            rejected += 1
        probes.append(
            {
                "name": probe["name"],
                "note": probe["note"],
                "accepted": bool(verdict["ok"]),
                "rules": sorted({item["rule"] for item in verdict["violations"]}),
            }
        )
    return {
        "ast_policy_version": AST_POLICY_VERSION,
        "probe_count": len(AST_PROBES),
        "rejected_count": rejected,
        "ast_rejected_rate": rejected / len(AST_PROBES) if AST_PROBES else None,
        "probes": probes,
        "interpretation": (
            "these are idioms, not receivers; the rate bounds how much of a measured agent "
            "failure rate is gate strictness rather than task difficulty"
        ),
    }


def capture_dirs(captures_root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Every capture directory of a split, in a deterministic order."""
    root = Path(captures_root) if captures_root is not None else dev_split_root()
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "iq.npy").is_file()
    )


def _negative_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measured evidence for the two claims about what the tools do not resolve."""
    betas = [row["truth"]["rrc_beta"] for row in rows if row["implied_beta"] is not None]
    implied = [row["implied_beta"] for row in rows if row["implied_beta"] is not None]
    beta_mean = _mean(betas)
    beta = {
        "n": len(betas),
        "correlation": _pearson(betas, implied),
        "mean_absolute_error": _mean([abs(a - b) for a, b in zip(implied, betas)]),
        "beta_stdev": _stdev(betas),
        # The estimator to beat: quote the population mean.
        "mean_predictor_absolute_error": (
            None if beta_mean is None else _mean([abs(b - beta_mean) for b in betas])
        ),
        "verdict": (
            "the implied roll-off is a worse estimator of rrc_beta than the population "
            "mean; the occupied bandwidth carries no usable beta information"
        ),
    }
    beta["worse_than_quoting_the_mean"] = bool(
        beta["mean_absolute_error"] is not None
        and beta["mean_predictor_absolute_error"] is not None
        and beta["mean_absolute_error"] > beta["mean_predictor_absolute_error"]
    )
    fractions = [
        row["envelope_null_period_fraction"]
        for row in rows
        if row["envelope_null_period_fraction"] is not None
    ]
    envelope_errors = [
        row["envelope_null_lag_error"]
        for row in rows
        if row["envelope_null_lag_error"] is not None
    ]
    fraction_mean = _mean(fractions)
    fraction_stdev = _stdev(fractions)
    envelope = {
        "n": len(envelope_errors),
        "exact": sum(1 for value in envelope_errors if value == 0),
        "within_2": sum(1 for value in envelope_errors if abs(value) <= 2),
        "period_fraction_mean": fraction_mean,
        "period_fraction_stdev": fraction_stdev,
        "period_fraction_relative_spread": (
            None
            if not fraction_mean or fraction_stdev is None
            else fraction_stdev / fraction_mean
        ),
        "verdict": (
            "the first squared-envelope autocorrelation null is not at the symbol period "
            "and must not be read as one; it is a shape cross-check"
        ),
    }
    return {
        "beta_from_occupied_bandwidth": beta,
        "envelope_null_is_not_the_symbol_period": envelope,
    }


def characterize_split(
    captures_root: str | os.PathLike[str] | None = None,
    *,
    nfft: int = DEFAULT_NFFT,
    max_lag: int = DEFAULT_MAX_LAG,
    bins: int = DEFAULT_BINS,
    token: Any = None,
) -> dict[str, Any]:
    """Characterize every capture of a split and summarize each tool's error."""
    directories = capture_dirs(captures_root)
    if not directories:
        raise ValueError("no captures found; the characterization would be vacuous")
    rows = [
        characterize_capture(directory, nfft=nfft, max_lag=max_lag, bins=bins, token=token)
        for directory in directories
    ]
    quantities: dict[tuple[str, str], list[float]] = {}
    relatives: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        for item in row["measurements"]:
            key = (item["tool"], item["quantity"])
            if item["absolute_error"] is not None:
                quantities.setdefault(key, []).append(item["absolute_error"])
            if item["relative_error"] is not None:
                relatives.setdefault(key, []).append(item["relative_error"])
    per_quantity = [
        {
            "tool": tool,
            "quantity": quantity,
            "absolute_error": _summary(quantities.get((tool, quantity), [])),
            "relative_error": _summary(relatives.get((tool, quantity), [])),
        }
        for tool, quantity in sorted(set(quantities) | set(relatives))
    ]
    top1 = sum(1 for row in rows if row["sps_top1"])
    top3 = sum(1 for row in rows if row["sps_rank"] is not None and row["sps_rank"] < 3)
    null_errors = [row["first_null_lag_error"] for row in rows if row["first_null_lag_error"] is not None]
    return {
        **_negative_results(rows),
        "schema_version": CHARACTERIZATION_SCHEMA_VERSION,
        "characterization_policy_version": CHARACTERIZATION_POLICY_VERSION,
        "tools_policy_version": TOOLS_POLICY_VERSION,
        "tools_sha256": tools_sha256(),
        "split": DEV_SPLIT_NAME if captures_root is None else str(captures_root),
        "capture_count": len(rows),
        "settings": {"nfft": nfft, "max_lag": max_lag, "bins": bins},
        "per_quantity": per_quantity,
        "symbol_rate_ranking": {
            "top1": top1,
            "top3": top3,
            "n": len(rows),
            "top1_rate": top1 / len(rows),
        },
        "first_null_lag_error": {
            "exact": sum(1 for value in null_errors if value == 0),
            "within_1": sum(1 for value in null_errors if abs(value) <= 1),
            "within_2": sum(1 for value in null_errors if abs(value) <= 2),
            "n": len(null_errors),
            "max_abs": max((abs(value) for value in null_errors), default=None),
        },
        "estimable_quantities": [dict(entry) for entry in ESTIMABLE_QUANTITIES],
        "ast_gate": ast_probe_report(),
        "rows": rows,
    }


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the checked-in error table."""
    lines: list[str] = []
    lines.append("# Tool characterization\n")
    lines.append(
        "Offline instrument characterization: every tool in `modembench.agent.tools` run over "
        f"all {report['capture_count']} captures of `{report['split']}`, with each reported "
        "quantity compared against the closed-form value implied by that capture's manifest. "
        "This path reads protected truth; it never runs inside an agent run.\n"
    )
    lines.append(f"- tools policy: `{report['tools_policy_version']}`")
    lines.append(f"- tools sha256: `{report['tools_sha256']}`")
    settings = report["settings"]
    lines.append(
        f"- settings: nfft={settings['nfft']}, max_lag={settings['max_lag']}, bins={settings['bins']}\n"
    )
    lines.append("## Error by reported quantity\n")
    lines.append("| tool | quantity | n | median error | p90 abs | max abs | median rel | p90 abs rel |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in report["per_quantity"]:
        absolute = entry["absolute_error"]
        relative = entry["relative_error"]
        lines.append(
            f"| `{entry['tool']}` | `{entry['quantity']}` | {absolute['n']} | "
            f"{_format(absolute['median'])} | {_format(absolute['p90_abs'])} | "
            f"{_format(absolute['max_abs'])} | {_format(relative['median'])} | "
            f"{_format(relative['p90_abs'])} |"
        )
    ranking = report["symbol_rate_ranking"]
    nulls = report["first_null_lag_error"]
    lines.append("\n## Discrete accuracy\n")
    lines.append(
        f"- `symbol_rate_candidates` top-1: **{ranking['top1']}/{ranking['n']}** "
        f"({ranking['top1_rate']:.0%}); top-3: {ranking['top3']}/{ranking['n']}"
    )
    lines.append(
        f"- `symbol_period_statistic` first-null lag: exact {nulls['exact']}/{nulls['n']}, "
        f"within +/-1 {nulls['within_1']}/{nulls['n']}, within +/-2 {nulls['within_2']}/{nulls['n']}, "
        f"worst {nulls['max_abs']} samples\n"
    )
    beta = report["beta_from_occupied_bandwidth"]
    envelope = report["envelope_null_is_not_the_symbol_period"]
    lines.append("\n## Measured negative results\n")
    lines.append(
        "What the instrument does **not** resolve, measured rather than asserted. Both rows "
        "used to be claims in a tool description; a deleted claim with no checked-in "
        "replacement is a claim nobody can re-test.\n"
    )
    lines.append(
        f"- **Roll-off (`rrc_beta`) is not recoverable from the occupied bandwidth.** Over "
        f"{beta['n']} captures the implied roll-off correlates with the true one at "
        f"r = {_format(beta['correlation'])}, with mean absolute error "
        f"{_format(beta['mean_absolute_error'])} against beta's own spread (sd "
        f"{_format(beta['beta_stdev'])}). Quoting the population mean scores "
        f"{_format(beta['mean_predictor_absolute_error'])} — better than the measurement, so "
        "the measurement carries no usable information. `beta` is the one receiver-critical "
        "parameter no tool estimates, and the agent is told so."
    )
    lines.append(
        f"- **The squared-envelope autocorrelation null is not the symbol period.** It "
        f"equals the period on {envelope['exact']}/{envelope['n']} captures and lands within "
        f"+/-2 samples on {envelope['within_2']}/{envelope['n']}, sitting instead at "
        f"{_format(envelope['period_fraction_mean'])} of the period with a "
        f"{envelope['period_fraction_relative_spread']:.0%} relative spread. It is reported "
        "to the agent as a shape cross-check, with this error published, and its row appears "
        "in the error table above.\n"
    )
    gate = report["ast_gate"]
    lines.append("## AST gate strictness\n")
    lines.append(
        f"Policy `{gate['ast_policy_version']}` rejects **{gate['rejected_count']}/"
        f"{gate['probe_count']}** ordinary receiver idioms "
        f"({gate['ast_rejected_rate']:.0%}). Every rejection scores as an agent failure, so "
        "this bounds how much of a measured failure rate is gate strictness rather than task "
        "difficulty.\n"
    )
    lines.append("| idiom | accepted | rules fired | note |")
    lines.append("| --- | :---: | --- | --- |")
    for probe in gate["probes"]:
        rules = ", ".join(f"`{rule}`" for rule in probe["rules"]) or "-"
        mark = "yes" if probe["accepted"] else "**no**"
        lines.append(f"| `{probe['name']}` | {mark} | {rules} | {probe['note']} |")
    lines.append("\n## Estimable quantities\n")
    lines.append(
        "What each tool narrows about the hidden manifest. A tool that hands over a hidden "
        "parameter is not disqualified by that; it is disqualified by not saying so.\n"
    )
    lines.append("| tool | manifest field | quantity | strength |")
    lines.append("| --- | --- | --- | --- |")
    for entry in report["estimable_quantities"]:
        lines.append(
            f"| `{entry['tool']}` | `{entry['manifest_field']}` | {entry['quantity']} | "
            f"{entry['strength']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_characterization(
    report: dict[str, Any], out_dir: str | os.PathLike[str]
) -> tuple[Path, Path]:
    """Write the JSON table and its rendered Markdown, atomically."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "tool-characterization.json"
    markdown_path = directory / "tool-characterization.md"
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    _replace(json_path, payload)
    _replace(markdown_path, render_markdown(report))
    return json_path, markdown_path


def _replace(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def default_report(
    captures_root: str | os.PathLike[str] | None = None, token: Any = None
) -> dict[str, Any]:
    """The report as it is checked in, with a generation date but no other volatile field."""
    report = characterize_split(captures_root, token=token)
    return {"generated_on": date.today().isoformat(), **report}


__all__ = [
    "AST_PROBES",
    "CHARACTERIZATION_POLICY_VERSION",
    "Measurement",
    "ast_probe_report",
    "capture_dirs",
    "characterize_capture",
    "characterize_split",
    "default_report",
    "expected_noise_psd",
    "expected_rms_magnitude",
    "render_markdown",
    "write_characterization",
]
