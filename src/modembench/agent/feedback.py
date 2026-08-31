"""Feedback wall: the one function allowed to hand evaluator output to the agent.

Only result["feedback"] crosses, exactly four fields; aligned_ber is quantized to a dyadic
1/64 grid on every split so the payload length cannot be recovered from it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..records import SEALED_REDACTION_MARKER
from ..sandbox.runner import (
    RESULT_STATUSES,
    SEALED_BER_GRID_DENOMINATOR,
    quantize_ber_to_grid,
)

FEEDBACK_POLICY_VERSION = "modembench-feedback-v1"

# The complete agent-visible field set; anything else is a contract change.
FORWARDED_KEYS: tuple[str, ...] = ("acquisition_success", "crc_pass", "aligned_ber", "error")

# Closed enum of feedback.error values. A free string here is an open channel from the
# evaluator to the model. The runner also assigns sandbox statuses into this field, so
# RESULT_STATUSES is unioned in below rather than restated.
_EVALUATOR_ERROR_CODES: frozenset[str] = frozenset(
    {
        "output_missing",
        "output_invalid_header",
        "output_wrong_shape",
        "output_wrong_dtype",
        "output_too_long",
        "output_corrupt",
        "output_nonbinary",
        "private_truth_invalid",
        "evaluation_invalid_truth",
        "evaluator_invalid",
    }
)
FORWARDED_ERROR_CODES: frozenset[str] = _EVALUATOR_ERROR_CODES | RESULT_STATUSES

# Orchestrator-internal keys; never forwarded, and asserted absent from what is.
INTERNAL_KEYS: tuple[str, ...] = (
    "packet_success",
    "n_payload_bits",
    "alignment_offset",
    "sync_hamming",
    # Imported rather than spelled so a rename cannot leave the wall checking a dead key.
    SEALED_REDACTION_MARKER,
)

# Reused from the sealed path so the two grids cannot drift apart.
AGENT_BER_GRID_DENOMINATOR = SEALED_BER_GRID_DENOMINATOR


class FeedbackWallError(RuntimeError):
    """The evaluator contract changed, or internal truth reached the forwarding function."""


def quantize_ber(value: Any) -> float | None:
    """Snap a BER onto the published dyadic grid. ``None`` passes through as ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackWallError(f"aligned_ber is not a number: {type(value).__name__}")
    # Finiteness first: a NaN or infinity would otherwise escape as a bare ValueError from
    # round() instead of FeedbackWallError.
    if not math.isfinite(float(value)):
        raise FeedbackWallError(f"aligned_ber is not finite: {value!r}")
    return quantize_ber_to_grid(value)


def forward_feedback(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only object an agent is allowed to receive from the evaluator.

    Raises :class:`FeedbackWallError` if the evaluator's feedback block has grown a field, is
    missing one, or contains an orchestrator-internal key.
    """
    if not isinstance(result, Mapping):
        raise FeedbackWallError("evaluator result is not a mapping")
    feedback = result.get("feedback")
    if not isinstance(feedback, Mapping):
        raise FeedbackWallError("evaluator result carries no feedback block")
    present = set(feedback)
    expected = set(FORWARDED_KEYS)
    if present != expected:
        raise FeedbackWallError(
            "evaluator feedback contract changed: "
            f"unexpected {sorted(present - expected)}, missing {sorted(expected - present)}"
        )
    forwarded: dict[str, Any] = {}
    for key in FORWARDED_KEYS:
        value = feedback[key]
        forwarded[key] = quantize_ber(value) if key == "aligned_ber" else value
    # Validate values, not just the key set.
    for key in ("acquisition_success", "crc_pass"):
        if not isinstance(forwarded[key], bool):
            raise FeedbackWallError(
                f"{key} is not a bool: {type(forwarded[key]).__name__} — the evaluator contract "
                "publishes it as a flag, and a non-bool is either a regression or a carrier"
            )
    ber = forwarded["aligned_ber"]
    if ber is not None and not (0.0 <= ber <= 1.0):
        raise FeedbackWallError(f"aligned_ber is not a rate in [0, 1]: {ber!r}")
    error = forwarded["error"]
    if error is not None and error not in FORWARDED_ERROR_CODES:
        raise FeedbackWallError(
            f"error is not one of the published codes: {error!r} — a free string here is an "
            "open channel from the evaluator to the model (see FORWARDED_ERROR_CODES)"
        )
    leaked = sorted(set(forwarded) & set(INTERNAL_KEYS))
    if leaked:
        raise FeedbackWallError(f"internal keys reached the forwarded feedback: {leaked}")
    return forwarded


def feedback_config() -> dict[str, Any]:
    """The wall's identity, as recorded in every run record."""
    return {
        "feedback_policy_version": FEEDBACK_POLICY_VERSION,
        "forwarded_keys": list(FORWARDED_KEYS),
        "aligned_ber_grid_denominator": AGENT_BER_GRID_DENOMINATOR,
        "aligned_ber_grid": f"multiples of 1/{AGENT_BER_GRID_DENOMINATOR}",
        "quantized_on_every_split": True,
    }
