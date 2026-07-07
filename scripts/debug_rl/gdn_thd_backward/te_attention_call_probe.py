#!/usr/bin/env python3
"""Run correctness while fingerprinting Transformer Engine attention calls."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import torch


def _tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _value(value):
    tensor = _tensor(value)
    if tensor is not None:
        return tensor
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return repr(value)


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _install_probe() -> None:
    import transformer_engine.pytorch as te

    original = te.DotProductAttention.forward
    calls = 0

    def wrapped(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        record = {
            "call": calls,
            "is_compiling": torch.compiler.is_compiling(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "args": [_value(item) for item in args],
            "kwargs": {key: _value(value) for key, value in sorted(kwargs.items())},
            "module": {
                "qkv_format": getattr(self, "qkv_format", None),
                "attn_mask_type": repr(getattr(self, "attn_mask_type", None)),
                "deterministic": getattr(self, "deterministic", None),
            },
        }
        print("TE_ATTN_CALL " + json.dumps(record, sort_keys=True), flush=True)
        output = original(self, *args, **kwargs)
        record = {"call": calls, "output": _tensor(_first_tensor(output))}
        print("TE_ATTN_RETURN " + json.dumps(record, sort_keys=True), flush=True)
        return output

    te.DotProductAttention.forward = wrapped


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: te_attention_call_probe.py CORRECTNESS_PY [ARGS ...]")
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    _install_probe()
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
