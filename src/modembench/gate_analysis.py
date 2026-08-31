"""The comparison gate: paired Δ, BCa interval, exact McNemar, routed verdict."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .conclusions import (
    GATE_THRESHOLD,
    Gate,
    State,
    Th17Authoring,
    Th17b,
    T12,
    Validity,
    deciding_rule,
)

GATE_ANALYSIS_POLICY_VERSION = "modembench-gate-analysis-v1"

#: TH-2. An alias of conclusions.GATE_THRESHOLD, the single spelling in the package.
DELTA_THRESHOLD = GATE_THRESHOLD
#: TH-5/TH-22 draws for the reported interval.
BOOTSTRAP_DRAWS = 10_000
#: TH-22: coverage below this, with a PASS, must be stated as a number beside the verdict.
COVERAGE_FLOOR = 0.93

_CONFIDENCE = 0.95


class PairingError(RuntimeError):
    """The two arms are not the same experiment; there is no comparison to compute."""


@dataclass(frozen=True)
class Cell:
    """One paired observation: the same (signal, replicate) under both arms."""

    signal: str
    replicate: int
    one_shot_success: bool
    iterative_success: bool

    @property
    def discordant(self) -> bool:
        return self.one_shot_success != self.iterative_success


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairingError(message)


def _signal_outcomes_one_shot(record: Mapping[str, Any]) -> dict[str, bool]:
    """``{signal: selected packet_success}`` from one best-of-N arm record."""
    out: dict[str, bool] = {}
    for detail in record.get("signals_detail") or ():
        label = str(detail.get("signal_label") or "")
        selection = detail.get("selection") or {}
        _require(bool(label), "a one-shot signal detail carries no signal_label")
        _require(
            "packet_success" in selection,
            f"signal {label!r}: no selection.packet_success — is this an arm-run record?",
        )
        out[label] = bool(selection["packet_success"])
    _require(bool(out), "the one-shot arm record contains no signals")
    return out


def _signal_outcomes_iterative(record: Mapping[str, Any]) -> dict[str, bool]:
    """``{signal: final outcome packet_success}`` from one iterative arm record."""
    out: dict[str, bool] = {}
    for detail in record.get("results") or record.get("signals_detail") or ():
        label = str(detail.get("signal") or detail.get("signal_label") or "")
        outcome = detail.get("outcome") or {}
        _require(bool(label), "an iterative signal detail carries no signal label")
        _require(
            outcome.get("kind") != "run_invalid",
            f"signal {label!r}: a run_invalid outcome reached the analysis — invalid runs "
            "are re-run upstream, so this arm run is unfinished, not unlucky",
        )
        _require(
            "packet_success" in outcome,
            f"signal {label!r}: no outcome.packet_success — is this an iterative arm record?",
        )
        out[label] = bool(outcome["packet_success"])
    _require(bool(out), "the iterative arm record contains no signals")
    return out


def build_cells(
    one_shot_records: Sequence[Mapping[str, Any]],
    iterative_records: Sequence[Mapping[str, Any]],
) -> tuple[Cell, ...]:
    """Pair the arms into cells; any pairing mismatch is a hard error, not a smaller n."""
    _require(bool(one_shot_records) and bool(iterative_records), "both arms need records")
    _require(
        len(one_shot_records) == len(iterative_records),
        f"replicate counts differ: {len(one_shot_records)} one-shot records vs "
        f"{len(iterative_records)} iterative",
    )
    hashes = {
        str(r.get("frozen_budget_sha256"))
        for r in (*one_shot_records, *iterative_records)
    }
    _require(
        len(hashes) == 1 and "None" not in hashes,
        f"frozen_budget_sha256 differs across records: {sorted(hashes)} — these runs were "
        "made under different frozen configurations and cannot be paired",
    )

    cells: list[Cell] = []
    for replicate, (one_record, it_record) in enumerate(
        zip(one_shot_records, iterative_records)
    ):
        ones = _signal_outcomes_one_shot(one_record)
        its = _signal_outcomes_iterative(it_record)
        _require(
            set(ones) == set(its),
            f"replicate {replicate}: capture sets differ "
            f"(one-shot only: {sorted(set(ones) - set(its))[:3]}, "
            f"iterative only: {sorted(set(its) - set(ones))[:3]})",
        )
        for signal in sorted(ones):
            cells.append(
                Cell(
                    signal=signal,
                    replicate=replicate,
                    one_shot_success=ones[signal],
                    iterative_success=its[signal],
                )
            )
    return tuple(cells)


# --- the estimate and its interval ----------------------------------------------------------
def delta_hat(cells: Sequence[Cell]) -> float:
    """Δ̂ = mean(iterative) − mean(one-shot) over the paired cells."""
    n = len(cells)
    if n == 0:
        raise PairingError("no cells")
    return (
        sum(c.iterative_success for c in cells) - sum(c.one_shot_success for c in cells)
    ) / n


def _delta_of_signals(
    by_signal: Mapping[str, Sequence[Cell]], drawn: Sequence[str]
) -> float:
    it = 0
    one = 0
    total = 0
    for name in drawn:
        for cell in by_signal[name]:
            it += cell.iterative_success
            one += cell.one_shot_success
            total += 1
    return (it - one) / total


def bootstrap_interval(
    cells: Sequence[Cell],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 20260814,
    confidence: float = _CONFIDENCE,
) -> dict[str, Any]:
    """Signal-clustered paired bootstrap; BCa bound reported, percentile printed beside.

    Signals are the resampling unit (TH-5): replicates of one signal share the capture, so
    cell-level resampling overstates n. BCa because the percentile bound under-covers here,
    which is anti-conservative toward PASS (TH-22). Seeded so the analysis re-runs to the digit.
    """
    by_signal: dict[str, list[Cell]] = {}
    for cell in cells:
        by_signal.setdefault(cell.signal, []).append(cell)
    signals = sorted(by_signal)
    n_signals = len(signals)
    _require(n_signals >= 2, "the clustered bootstrap needs at least two signals")

    rng = np.random.default_rng(seed)
    observed = delta_hat(cells)
    replicates = np.empty(draws)
    for index in range(draws):
        drawn = rng.choice(signals, size=n_signals, replace=True)
        replicates[index] = _delta_of_signals(by_signal, drawn)

    alpha = 1.0 - confidence
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    percentile = (
        float(np.percentile(replicates, lo_q)),
        float(np.percentile(replicates, hi_q)),
    )

    # BCa: z0 from the share of resamples below the estimate; acceleration from the
    # jackknife over signal clusters (leave one signal out, all its cells with it).
    below = float(np.mean(replicates < observed))
    # a degenerate bootstrap has no distribution to correct
    if below in (0.0, 1.0):
        z0 = 0.0
    else:
        z0 = float(_norm_ppf(below))
    jack = np.array(
        [
            _delta_of_signals(by_signal, [s for s in signals if s != held_out])
            for held_out in signals
        ]
    )
    jack_mean = jack.mean()
    numerator = float(np.sum((jack_mean - jack) ** 3))
    denominator = 6.0 * float(np.sum((jack_mean - jack) ** 2) ** 1.5)
    acceleration = 0.0 if denominator == 0.0 else numerator / denominator

    def _bca_quantile(nominal: float) -> float:
        z = _norm_ppf(nominal)
        adjusted = z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))
        return float(np.percentile(replicates, 100 * _norm_cdf(adjusted)))

    bca = (_bca_quantile(alpha / 2), _bca_quantile(1 - alpha / 2))
    return {
        "delta_hat": observed,
        "confidence": confidence,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "signal cluster: a drawn signal contributes all its cells in both arms",
        "bca": {"lower": bca[0], "upper": bca[1], "z0": z0, "acceleration": acceleration},
        "percentile": {"lower": percentile[0], "upper": percentile[1]},
        "reported_bound": "bca.lower — TH-22; the percentile bound is printed beside, never substituted",
        "n_signals": n_signals,
        "n_cells": len(cells),
    }


def _norm_ppf(p: float) -> float:
    """Standard normal quantile via statistics.NormalDist; stdlib-only on purpose."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def _norm_cdf(x: float) -> float:
    from statistics import NormalDist

    return NormalDist().cdf(x)


# --- McNemar --------------------------------------------------------------------------------
def mcnemar_exact(cells: Sequence[Cell]) -> dict[str, Any]:
    """Exact two-sided binomial McNemar on the discordant pairs; a diagnostic, not the gate."""
    b = sum(1 for cell in cells if cell.iterative_success and not cell.one_shot_success)
    c = sum(1 for cell in cells if cell.one_shot_success and not cell.iterative_success)
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
        p_value = min(1.0, 2.0 * tail)
    return {
        "test": "exact binomial McNemar, two-sided, alpha=0.05",
        "b_iterative_only": b,
        "c_one_shot_only": c,
        "discordant": n,
        "p_value": p_value,
        "role": (
            "diagnostic — the verdict is the BCa lower bound against "
            f"+{DELTA_THRESHOLD} and nothing else"
        ),
    }


# --- the verdict ----------------------------------------------------------------------------
def analyze(
    one_shot_records: Sequence[Mapping[str, Any]],
    iterative_records: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 20260814,
    coverage_calibration: Mapping[str, Any] | None = None,
    validity: Validity = Validity.CLEAR,
    th17_authoring: Th17Authoring = Th17Authoring.LT_050,
    th17b: Th17b = Th17b.F2_NOT_INDICATED,
    t12: T12 = T12.NOT_RUN,
) -> dict[str, Any]:
    """Cells to interval to routed conclusion; validity flags default clean for the dev gate."""
    cells = build_cells(one_shot_records, iterative_records)
    interval = bootstrap_interval(cells, draws=draws, seed=seed)
    diagnostic = mcnemar_exact(cells)

    lower = interval["bca"]["lower"]
    passed = lower >= DELTA_THRESHOLD
    gate = Gate.PASS if passed else Gate.FAIL
    state = State(
        gate=gate,
        validity=validity,
        th17_authoring=th17_authoring,
        th17b=th17b,
        t12=t12,
    )
    rule = deciding_rule(state)

    coverage_note: Any
    if coverage_calibration is None:
        coverage_note = "NOT RUN — dev analysis only; the sealed analysis refuses without it"
    else:
        coverage_note = dict(coverage_calibration)
        measured = float(coverage_note.get("two_sided_coverage", float("nan")))
        if passed and measured < COVERAGE_FLOOR:
            coverage_note["stated_beside_verdict"] = (
                f"measured two-sided coverage {measured:.3f} < {COVERAGE_FLOOR}: the "
                "effective one-sided error exceeds nominal and this PASS may not be "
                "described as a 95%-level finding"
            )

    return {
        "gate_analysis_policy_version": GATE_ANALYSIS_POLICY_VERSION,
        "threshold": DELTA_THRESHOLD,
        "threshold_rule": (
            f"PASS iff the BCa 2.5th percentile of Delta >= +{DELTA_THRESHOLD}; nothing "
            "else is the gate"
        ),
        "n_cells": len(cells),
        "n_signals": interval["n_signals"],
        "replicates": len(one_shot_records),
        "delta_hat": interval["delta_hat"],
        "interval": interval,
        "mcnemar": diagnostic,
        "verdict": gate.name,
        "conclusion": rule.conclusion.name,
        "conclusion_rule_id": rule.id,
        "conclusion_because": rule.because,
        "coverage_calibration": coverage_note,
        "success_rates": {
            "one_shot": sum(c.one_shot_success for c in cells) / len(cells),
            "iterative": sum(c.iterative_success for c in cells) / len(cells),
        },
        "written_regardless_of_direction": True,
    }


# --- TH-22: coverage calibration ------------------------------------------------------------
def coverage_calibration(
    *,
    n_signals: int,
    replicates: int,
    one_shot_rate: float,
    delta: float = 0.0,
    icc: float = 0.5,
    campaigns: int = 1_000,
    draws: int = 2_000,
    seed: int = 20260814,
    confidence: float = _CONFIDENCE,
) -> dict[str, Any]:
    """Two-sided coverage of the BCa interval under a signal-random-effects model.

    A shared per-signal latent induces the within-cluster correlation icc and the
    between-arm pairing. TH-22 requires >= 1,000 campaigns at 10,000 draws before sealed access.
    """
    rng = np.random.default_rng(seed)
    from statistics import NormalDist

    ppf = NormalDist().inv_cdf
    covered = 0
    for _ in range(campaigns):
        cells: list[Cell] = []
        for s in range(n_signals):
            # one latent per signal, shared by both arms and every replicate
            latent = rng.standard_normal() * math.sqrt(icc)
            for r in range(replicates):
                noise_one = rng.standard_normal() * math.sqrt(1 - icc)
                noise_it = rng.standard_normal() * math.sqrt(1 - icc)
                one = (latent + noise_one) < ppf(one_shot_rate)
                it = (latent + noise_it) < ppf(min(1.0 - 1e-9, one_shot_rate + delta))
                cells.append(
                    Cell(
                        signal=f"s{s}",
                        replicate=r,
                        one_shot_success=bool(one),
                        iterative_success=bool(it),
                    )
                )
        interval = bootstrap_interval(
            cells, draws=draws, seed=int(rng.integers(2**31)), confidence=confidence
        )
        true_delta = _true_delta(one_shot_rate, delta, icc)
        if interval["bca"]["lower"] <= true_delta <= interval["bca"]["upper"]:
            covered += 1
    coverage = covered / campaigns
    return {
        "two_sided_coverage": coverage,
        "nominal": confidence,
        "campaigns": campaigns,
        "draws": draws,
        "n_signals": n_signals,
        "replicates": replicates,
        "one_shot_rate": one_shot_rate,
        "delta": delta,
        "icc": icc,
        "seed": seed,
        "floor": COVERAGE_FLOOR,
        "monte_carlo_se": math.sqrt(coverage * (1 - coverage) / campaigns),
    }


def _true_delta(one_shot_rate: float, delta: float, icc: float) -> float:
    """Marginal success rates are exact normal-CDF arguments, independent of icc."""
    return min(1.0 - 1e-9, one_shot_rate + delta) - one_shot_rate
