"""Trusted child-shim source generation."""

from __future__ import annotations

from hashlib import sha256
import json

SHIM_VERSION = "modembench-shim-v1"


def make_shim_source(
    *, sample_rate: float, cpu_seconds: int, fsize_bytes: int, project_src: str
) -> str:
    values = {
        "sample_rate": float(sample_rate),
        "cpu_seconds": int(cpu_seconds),
        "fsize_bytes": int(fsize_bytes),
        "project_src": project_src,
    }
    constants = json.dumps(values, sort_keys=True)
    return f'''# Generated trusted ModemBench shim: {SHIM_VERSION}
import importlib.util
import json
import os
import resource
import sys
import numpy as np

CONFIG = {constants}

def status(code, error, detail=None):
    try:
        with open("status.json", "w", encoding="utf-8") as stream:
            json.dump({{"code": code, "error": error, "detail": detail}}, stream, sort_keys=True)
    except BaseException:
        pass
    raise SystemExit(code)

try:
    resource.setrlimit(resource.RLIMIT_CPU, (CONFIG["cpu_seconds"], CONFIG["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CONFIG["fsize_bytes"], CONFIG["fsize_bytes"]))
    project_src = os.path.realpath(CONFIG["project_src"])
    sys.path[:] = [
        entry for entry in sys.path
        if os.path.realpath(entry or os.getcwd()) != project_src
    ]
except BaseException as exc:
    status(14, "shim_internal_error", type(exc).__name__)

try:
    iq = np.load("iq.npy", allow_pickle=False)
except BaseException as exc:
    status(13, "iq_load_failed", type(exc).__name__)

try:
    spec = importlib.util.spec_from_file_location("receiver", "receiver.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("receiver module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receive = getattr(module, "receive", None)
except SystemExit as exc:
    status(11, "receiver_raised", "SystemExit:" + repr(exc.code))
except BaseException as exc:
    status(11, "receiver_raised", type(exc).__name__)

if receive is None or not callable(receive):
    status(10, "missing_receive")

try:
    value = receive(iq, CONFIG["sample_rate"])
except SystemExit as exc:
    status(11, "receiver_raised", "SystemExit:" + repr(exc.code))
except BaseException as exc:
    status(11, "receiver_raised", type(exc).__name__)

try:
    bits = np.asarray(value)
    if bits.ndim != 1:
        status(12, "bad_output", "wrong_shape")
    if bits.dtype != np.uint8:
        status(12, "bad_output", "wrong_dtype")
    if bits.size > 2**20:
        status(12, "bad_output", "too_long")
    if np.any(bits > 1):
        status(12, "bad_output", "nonbinary")
    np.save("bits.npy", bits, allow_pickle=False)
    with open("status.json", "w", encoding="utf-8") as stream:
        json.dump({{"code": 0, "error": None}}, stream, sort_keys=True)
except SystemExit:
    raise
except BaseException as exc:
    status(14, "shim_internal_error", type(exc).__name__)
raise SystemExit(0)
'''


def shim_sha256(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()
