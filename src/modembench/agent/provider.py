"""Model-provider seam: one protocol, one live client, one replay client."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Iterable, Protocol, runtime_checkable

from ..sandbox.profile import SANDBOX_EXEC
from .accounting import TokenUsage, usage_from_response

PROVIDER_POLICY_VERSION = "modembench-provider-v1"
RECORDING_SCHEMA_VERSION = "1.0"

# The plan budgets a mid-tier headline model. Flipping to "claude-opus-5" is a plan
# amendment, not a tweak: roughly 2.5x the bill, and it moves the frozen-config hash.
HEADLINE_MODEL = "claude-sonnet-5"

# Frozen sampling/reasoning configuration; build_request_payload rejects disagreement.
FROZEN_EFFORT = "high"
FROZEN_THINKING = "adaptive"
# Thinking is billed against max_tokens; the ceiling must leave room for the receiver.
MAX_OUTPUT_TOKENS = 32000
# Recorded in the frozen config; nothing in this module can turn it on.
SERVER_SIDE_FALLBACKS_ENABLED = False

# Keys that must never appear in a request payload.
FORBIDDEN_REQUEST_KEYS = frozenset(
    {"fallbacks", "temperature", "top_p", "top_k", "budget_tokens"}
)


class ProviderError(RuntimeError):
    """A provider contract violation the caller cannot classify away."""


class ProviderUnavailable(ProviderError):
    """The live provider cannot be constructed on this machine."""


class ReplayMismatch(ProviderError):
    """A replayed request diverged from the recorded one."""


# Reasons a response is unusable through no fault of the agent; all map to run_invalid.
INVALID_MODEL_IDENTITY = "model_identity_mismatch"
INVALID_REFUSAL = "refusal"
INVALID_MAX_TOKENS = "max_tokens_truncation"
INVALID_TRANSPORT = "transport_error"
INVALID_PAUSE_TURN = "pause_turn_unhandled"
PROVIDER_INVALID_REASONS = frozenset(
    {
        INVALID_MODEL_IDENTITY,
        INVALID_REFUSAL,
        INVALID_MAX_TOKENS,
        INVALID_TRANSPORT,
        INVALID_PAUSE_TURN,
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def require_scoring_platform() -> None:
    """Refuse to spend money on a machine that cannot score the result."""
    if sys.platform != "darwin":
        raise ProviderUnavailable(
            "pre-flight failed: receiver scoring requires macOS sandbox-exec, so no API "
            f"call will be made on platform {sys.platform!r}"
        )
    if not SANDBOX_EXEC.is_file():
        raise ProviderUnavailable(
            f"pre-flight failed: {SANDBOX_EXEC} is missing, so no API call will be made"
        )


@dataclass(frozen=True)
class ProviderRequest:
    """One fully-specified model call. Immutable, hashable, and the unit replay keys on."""

    model: str
    system: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    max_tokens: int = MAX_OUTPUT_TOKENS
    effort: str = FROZEN_EFFORT
    thinking: str = FROZEN_THINKING
    output_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """One model reply, with the served model identity kept separate from the requested one."""

    model: str
    stop_reason: str | None
    content: tuple[dict[str, Any], ...]
    usage: TokenUsage = field(default_factory=TokenUsage)
    invalid_reason: str | None = None
    request_id: str | None = None
    stop_details: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.invalid_reason is None

    def text_blocks(self) -> tuple[str, ...]:
        return tuple(
            str(block.get("text", ""))
            for block in self.content
            if block.get("type") == "text"
        )

    def tool_uses(self) -> tuple[dict[str, Any], ...]:
        return tuple(block for block in self.content if block.get("type") == "tool_use")


@runtime_checkable
class Provider(Protocol):
    """Everything the harness needs from a model. Two implementations, no third."""

    name: str

    def complete(self, request: ProviderRequest) -> ProviderResponse:  # pragma: no cover
        ...


def _assert_no_prefill(messages: Iterable[dict[str, Any]]) -> None:
    # A trailing assistant turn is a prefill, which 400s on current models; submission uses
    # output_config.format instead.
    ordered = list(messages)
    if ordered and ordered[-1].get("role") == "assistant":
        raise ProviderError(
            "refusing to send an assistant prefill: receiver submission uses "
            "output_config.format, and a trailing assistant turn 400s on current models"
        )


def _assert_assistant_turns_are_replayable(messages: Iterable[dict[str, Any]]) -> None:
    # Thinking blocks must be echoed back exactly as received; a block reduced to a bare
    # type marker would 400 on the next turn, so refuse it before sending.
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind in VERBATIM_BLOCK_TYPES and set(block) <= {"type"}:
                raise ProviderError(
                    f"a {kind!r} block was reduced to a bare type marker; the API requires "
                    "these blocks echoed back exactly as received, so replaying this turn "
                    "would 400 mid-sweep"
                )


def build_request_payload(request: ProviderRequest) -> dict[str, Any]:
    """Render the single wire payload this project is allowed to send."""
    if request.effort != FROZEN_EFFORT:
        raise ProviderError(
            f"effort is frozen at {FROZEN_EFFORT!r}; a per-invocation effort would make two "
            "runs of the same sweep incomparable without moving the config hash"
        )
    if request.thinking != FROZEN_THINKING:
        raise ProviderError(f"thinking is frozen at {FROZEN_THINKING!r}")
    if not request.model:
        raise ProviderError("a request must name a model")
    _assert_no_prefill(request.messages)
    _assert_assistant_turns_are_replayable(request.messages)
    output_config: dict[str, Any] = {"effort": request.effort}
    if request.output_format is not None:
        output_config["format"] = request.output_format
    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": int(request.max_tokens),
        # tools -> system -> messages, so the system-block cache breakpoint covers the tools.
        "tools": list(request.tools),
        "system": list(request.system),
        "messages": list(request.messages),
        "thinking": {"type": request.thinking},
        "output_config": output_config,
    }
    offending = FORBIDDEN_REQUEST_KEYS & set(payload)
    if offending:
        raise ProviderError(f"forbidden request keys present: {sorted(offending)}")
    return payload


def request_digest(request: ProviderRequest) -> str:
    """Content address of a request; binds a replay to what was recorded."""
    return sha256(canonical_json(build_request_payload(request))).hexdigest()


def _classify_stop(stop_reason: str | None) -> str | None:
    if stop_reason == "refusal":
        return INVALID_REFUSAL
    if stop_reason == "max_tokens":
        return INVALID_MAX_TOKENS
    if stop_reason == "pause_turn":
        # Only server-side tools produce this and none are enabled.
        return INVALID_PAUSE_TURN
    return None


def classify_response(requested_model: str, served_model: str, stop_reason: str | None) -> str | None:
    """The response-side validity verdict. ``None`` means usable."""
    if served_model != requested_model:
        return INVALID_MODEL_IDENTITY
    return _classify_stop(stop_reason)


# Block types the API requires echoed back exactly as received on a subsequent turn;
# rebuilding them drops the signature (or the redacted data blob) and the next request 400s.
VERBATIM_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
# Blocks the harness itself reads (text_blocks, tool_uses); normalized rather than dumped.
NORMALIZED_BLOCK_TYPES = frozenset({"text", "tool_use"})


def _dump_block(block: Any) -> dict[str, Any] | None:
    # model_dump(mode="json") returns every field, including ones this file has never heard
    # of, so future block fields survive the round trip.
    dumper = getattr(block, "model_dump", None)
    if callable(dumper):
        try:
            value = dumper(mode="json")
        except TypeError:  # pragma: no cover - non-pydantic model with a model_dump
            value = dumper()
        if isinstance(value, dict) and value:
            return dict(value)
    dumper = getattr(block, "to_dict", None)
    if callable(dumper):
        value = dumper()
        if isinstance(value, dict) and value:
            return dict(value)
    attributes = {
        name: value
        for name, value in vars(block).items()
        if not name.startswith("_")
    } if hasattr(block, "__dict__") else {}
    if attributes:
        return {"type": str(getattr(block, "type", None)), **attributes}
    return None


def _block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    kind = getattr(block, "type", None)
    if kind in NORMALIZED_BLOCK_TYPES:
        if kind == "text":
            return {"type": "text", "text": getattr(block, "text", "")}
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": dict(getattr(block, "input", {}) or {}),
        }
    # Everything else is carried through verbatim rather than reconstructed.
    dumped = _dump_block(block)
    if dumped is None:
        raise ProviderError(
            f"cannot preserve a {kind!r} content block verbatim: the SDK object exposes "
            "neither model_dump nor to_dict, so echoing it back would corrupt the turn"
        )
    dumped.setdefault("type", str(kind))
    return dumped


class AnthropicProvider:
    """The live provider. Official ``anthropic`` SDK only, imported lazily; never raw HTTP."""

    name = "anthropic"

    def __init__(self, *, model: str = HEADLINE_MODEL, client: Any = None) -> None:
        if client is None:
            require_scoring_platform()
        self.model = model
        self._client = client

    @staticmethod
    def _load_sdk() -> Any:
        try:
            import anthropic  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise ProviderUnavailable(
                "the official `anthropic` SDK is not installed; install it to run live "
                "sweeps (the test suite never needs it)"
            ) from exc
        return anthropic

    def client(self) -> Any:
        if self._client is None:
            anthropic = self._load_sdk()
            # No base_url override and no custom transport; the SDK resolves the endpoint.
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload = build_request_payload(request)
        client = self.client()
        try:
            # client.messages, not client.beta.messages: `fallbacks` only exists on the
            # beta surface, so fallbacks are structurally unreachable here.
            with client.messages.stream(**payload) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - every transport failure is one outcome
            # Never interpolate the exception: SDK error strings can embed the request body.
            return ProviderResponse(
                model="",
                stop_reason=None,
                content=(),
                invalid_reason=INVALID_TRANSPORT,
                request_id=None,
                stop_details={"error_type": type(exc).__name__},
            )
        served = str(getattr(message, "model", "") or "")
        stop_reason = getattr(message, "stop_reason", None)
        details = getattr(message, "stop_details", None)
        return ProviderResponse(
            model=served,
            stop_reason=stop_reason,
            content=tuple(_block_to_dict(block) for block in getattr(message, "content", ())),
            usage=usage_from_response(getattr(message, "usage", {})),
            invalid_reason=classify_response(request.model, served, stop_reason),
            request_id=getattr(message, "_request_id", None),
            stop_details=(
                {"category": getattr(details, "category", None)} if details is not None else None
            ),
        )


def response_record(response: ProviderResponse) -> dict[str, Any]:
    """The serialized form of one response, as stored in a replay recording."""
    return {
        "model": response.model,
        "stop_reason": response.stop_reason,
        "content": [dict(block) for block in response.content],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "cache_creation_5m_tokens": response.usage.cache_creation_5m_tokens,
            "cache_creation_1h_tokens": response.usage.cache_creation_1h_tokens,
            "cache_read_tokens": response.usage.cache_read_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "request_id": response.request_id,
    }


def build_recording(
    entries: Iterable[tuple[str | None, ProviderResponse]], *, model: str
) -> dict[str, Any]:
    """Assemble a replay document from ``(request_digest, response)`` pairs."""
    return {
        "schema_version": RECORDING_SCHEMA_VERSION,
        "policy_version": PROVIDER_POLICY_VERSION,
        "model": model,
        "responses": [
            {"request_sha256": digest, **response_record(response)}
            for digest, response in entries
        ],
    }


def write_recording(path: str | os.PathLike[str], document: dict[str, Any]) -> Path:
    """Write a recording atomically; refuse to change one that already exists."""
    target = Path(path)
    content = canonical_json(document) + b"\n"
    if target.exists():
        if target.read_bytes() != content:
            raise ProviderError(f"refusing to overwrite a differing recording: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, target)
    return target


class ReplayProvider:
    """Serves recorded responses in order; a request that does not hash to the recorded
    ``request_sha256`` is a hard error."""

    name = "replay"

    def __init__(self, document: dict[str, Any], *, strict: bool = True) -> None:
        if not isinstance(document, dict) or document.get("schema_version") != RECORDING_SCHEMA_VERSION:
            raise ReplayMismatch("recording is not a version-1.0 replay document")
        responses = document.get("responses")
        if not isinstance(responses, list):
            raise ReplayMismatch("recording has no response list")
        self.document = document
        self.model = str(document.get("model") or "")
        self._responses = responses
        self._index = 0
        self._strict = strict
        self.requests: list[ProviderRequest] = []

    @classmethod
    def from_path(cls, path: str | os.PathLike[str], *, strict: bool = True) -> "ReplayProvider":
        raw = Path(path).read_text(encoding="utf-8")
        return cls(json.loads(raw), strict=strict)

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._responses)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        # Build the payload even in replay so its guards are re-proved.
        digest = request_digest(request)
        self.requests.append(request)
        if self.exhausted:
            raise ReplayMismatch(
                f"recording holds {len(self._responses)} response(s); the harness asked for more"
            )
        entry = self._responses[self._index]
        self._index += 1
        recorded = entry.get("request_sha256")
        if self._strict and recorded is not None and recorded != digest:
            raise ReplayMismatch(
                f"replayed request {self._index} does not match the recording "
                f"(recorded {recorded[:16]}, replayed {digest[:16]})"
            )
        served = str(entry.get("model") or self.model)
        stop_reason = entry.get("stop_reason")
        usage = entry.get("usage") or {}
        return ProviderResponse(
            model=served,
            stop_reason=stop_reason,
            content=tuple(dict(block) for block in entry.get("content", ())),
            usage=TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                cache_creation_5m_tokens=int(usage.get("cache_creation_5m_tokens", 0)),
                cache_creation_1h_tokens=int(usage.get("cache_creation_1h_tokens", 0)),
                cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            invalid_reason=classify_response(request.model, served, stop_reason),
            request_id=entry.get("request_id"),
        )
