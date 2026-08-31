"""CLI for the out-of-process protected evaluator.

Either a capture's protected dir (read through the sealed chokepoint, so sealed captures
refuse) or --truth-stdin, where the orchestrator supplies truth it already read under its
own logged session; a token cannot cross a process boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .evaluator import evaluate_file, evaluate_output, parse_truth, truth_invalid_result
from .sealed import SealedAccessError

TRUTH_SCHEMA_VERSION = "1.0"
REFUSED_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ModemBench frame bits")
    parser.add_argument("bits_npy", type=Path)
    parser.add_argument("protected_dir", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--truth-stdin",
        action="store_true",
        help="read already-authorized truth from stdin instead of opening a capture",
    )
    return parser


def encode_truth(manifest_bytes: bytes, payload: bytes) -> str:
    """Serialize truth for the stdin channel. Bytes only; never a path."""
    return json.dumps(
        {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "manifest": manifest_bytes.decode("utf-8"),
            "payload_hex": payload.hex(),
        },
        separators=(",", ":"),
    )


def _decode_truth(raw: str) -> tuple[dict[str, Any], bytes]:
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema_version") != TRUTH_SCHEMA_VERSION:
        raise ValueError("truth payload is malformed")
    return parse_truth(document["manifest"].encode("utf-8"), bytes.fromhex(document["payload_hex"]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.truth_stdin:
        try:
            manifest, payload = _decode_truth(sys.stdin.read())
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            print(json.dumps(truth_invalid_result(), sort_keys=True, separators=(",", ":")))
            return 1
        result = evaluate_output(args.bits_npy, manifest, payload)
    else:
        if args.protected_dir is None:
            build_parser().error("a capture's protected directory or --truth-stdin is required")
        try:
            result = evaluate_file(args.bits_npy, args.protected_dir)
        except SealedAccessError as exc:
            # distinct exit code, never a scoring result
            print(json.dumps({"error": str(exc)}, sort_keys=True, separators=(",", ":")))
            return REFUSED_EXIT_CODE
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["internal"]["packet_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
