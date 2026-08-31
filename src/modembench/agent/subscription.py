"""Subscription transport: runs the tool-using harness over the `claude -p` CLI.

Plain text instead of the native tool-use API, so schemas are unenforced and thinking is
dropped between turns. Both arms use the same transport, so this affects absolute rates
but cancels out of the paired comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from .accounting import TokenUsage
from .provider import (
    PROVIDER_POLICY_VERSION,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    classify_response,
)

SUBSCRIPTION_TRANSPORT_VERSION = "modembench-subscription-v1"

# Quoted verbatim into every serialized prompt.
PROTOCOL = """\
=== HOW TO ANSWER (transport protocol; follow it exactly) ===

You are running over a text transport. To act, reply with EXACTLY ONE fenced JSON block and
nothing else outside it:

To call a diagnostic tool (only while your tool budget lasts):

```json
{"tool_call": {"name": "<tool name>", "arguments": {...}}}
```

To submit your receiver (this ends the run — there are no tool calls after it):

```json
{"submission": {"receiver_source": "<the complete contents of receiver.py>", "approach": "<one or two sentences>"}}
```

One block per reply. A reply that is not one of these two forms is treated as a failed
submission, so do not narrate outside the fence.
"""

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def serialize_conversation(request: ProviderRequest) -> str:
    """Serialize the whole request as one prompt: system, tools, protocol, then the turns."""
    parts: list[str] = ["=== STANDING INSTRUCTIONS (system) ==="]
    parts.extend(str(block.get("text", "")) for block in request.system)
    if request.tools:
        parts.append("=== AVAILABLE TOOLS (schemas) ===")
        parts.append(json.dumps(list(request.tools), indent=1, sort_keys=True))
    parts.append(PROTOCOL)
    parts.append("=== CONVERSATION SO FAR ===")
    for message in request.messages:
        role = str(message.get("role"))
        for block in message.get("content") or ():
            kind = block.get("type")
            if kind == "text":
                parts.append(f"[{role}]\n{block.get('text', '')}")
            elif kind == "tool_use":
                parts.append(
                    f"[{role} tool_call]\n"
                    + json.dumps(
                        {"name": block.get("name"), "arguments": block.get("input")},
                        sort_keys=True,
                    )
                )
            elif kind == "tool_result":
                content = block.get("content")
                if isinstance(content, (list, tuple)):
                    content = "\n".join(
                        str(item.get("text", "")) for item in content if isinstance(item, dict)
                    )
                parts.append(f"[tool result]\n{content}")
            elif kind in ("thinking", "redacted_thinking"):
                # Reasoning is never part of the serialized condition.
                continue
            else:
                parts.append(f"[{role} {kind}]\n{json.dumps(block, sort_keys=True)}")
    parts.append("=== YOUR REPLY (one fenced JSON block, per the protocol) ===")
    return "\n\n".join(parts)


def parse_reply(text: str) -> tuple[dict[str, Any], ...]:
    """CLI reply text -> standard content blocks. Never raises: a bad reply is a text block."""
    for candidate in _FENCE.findall(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        call = payload.get("tool_call")
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            return (
                {
                    "type": "tool_use",
                    "id": f"sub_{int(time.time() * 1000) % 10**9}",
                    "name": call["name"],
                    "input": call.get("arguments") or {},
                },
            )
        submission = payload.get("submission")
        if isinstance(submission, dict):
            # The shape _parse_submission expects: a text block holding the submission JSON.
            return ({"type": "text", "text": json.dumps(submission, sort_keys=True)},)
    return ({"type": "text", "text": text},)


def _usage_from_envelope(envelope: dict[str, Any]) -> TokenUsage:
    usage = envelope.get("usage") or {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        cache_creation_5m_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )


class SubscriptionProvider:
    """`Provider` over `claude -p`. One stateless CLI invocation per `complete()`."""

    name = "subscription-cli"

    def __init__(
        self,
        model: str,
        *,
        binary: str = "claude",
        timeout_s: int = 900,
        retries: int = 3,
        workdir: str | None = None,
    ) -> None:
        self.model = model
        self.binary = binary
        self.timeout_s = timeout_s
        self.retries = retries
        # Empty scratch cwd, never the repository: the CLI hands the model file and bash
        # tools, so private/ must not be where it is standing.
        self._workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mb-sub-"))
        self._workdir.mkdir(parents=True, exist_ok=True)

    def transport_config(self) -> dict[str, Any]:
        try:
            version = subprocess.run(
                [self.binary, "--version"], capture_output=True, timeout=30
            ).stdout.decode().strip()
        except Exception:  # noqa: BLE001 - record "unavailable" rather than dying
            version = "unavailable"
        return {
            "transport": SUBSCRIPTION_TRANSPORT_VERSION,
            "provider_policy_version": PROVIDER_POLICY_VERSION,
            "cli_version": version,
            "tool_protocol": "fenced-json v1 (tool_call | submission)",
            "native_tool_use": False,
            "schema_enforced_output": False,
        }

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if request.model != self.model:
            raise ProviderError(
                f"this provider serves {self.model!r}; the request names {request.model!r}"
            )
        prompt = serialize_conversation(request)
        last_error = ""
        for attempt in range(self.retries):
            if attempt:
                time.sleep(30 * attempt)
            try:
                proc = subprocess.run(
                    [
                        self.binary,
                        "-p",
                        "--no-session-persistence",
                        "--output-format",
                        "json",
                        "--model",
                        self.model,
                    ],
                    input=prompt.encode("utf-8"),
                    capture_output=True,
                    cwd=str(self._workdir),
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                last_error = f"CLI timeout after {self.timeout_s}s"
                continue
            if proc.returncode != 0:
                last_error = proc.stderr.decode()[:2000] or f"exit {proc.returncode}"
                continue
            stdout = proc.stdout.decode("utf-8")
            try:
                envelope, end = json.JSONDecoder().raw_decode(stdout)
            except json.JSONDecodeError as exc:
                last_error = f"unparseable CLI envelope: {exc}"
                continue
            text = str(envelope.get("result") or "")
            served = str(envelope.get("model") or self.model)
            stop = "end_turn" if not envelope.get("is_error") else "error"
            return ProviderResponse(
                model=served,
                stop_reason=stop,
                content=parse_reply(text),
                usage=_usage_from_envelope(envelope),
                invalid_reason=classify_response(request.model, served, stop),
                request_id=str(envelope.get("session_id") or "") or None,
            )
        # Transport failure is run_invalid via the provider-invalid path; re-run, not scored.
        return ProviderResponse(
            model=self.model,
            stop_reason=None,
            content=(),
            invalid_reason="transport_error",
        )
