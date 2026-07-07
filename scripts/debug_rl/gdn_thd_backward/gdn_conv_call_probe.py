#!/usr/bin/env python3
"""Run the correctness CLI while fingerprinting the first FLA conv calls."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

import torch


def _tensor_summary(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _chunk_summaries(tensor: torch.Tensor, dim: int = 2) -> list[dict]:
    if tensor.ndim <= dim or tensor.shape[dim] % 2:
        return []
    return [_tensor_summary(chunk) for chunk in tensor.chunk(2, dim=dim)]


def _install_l2norm_probe() -> None:
    import fla.modules.l2norm as l2norm_module

    original = l2norm_module.l2norm
    maximum = int(os.environ.get("GDN_L2NORM_PROBE_MAX_CALLS", "8"))
    calls = 0

    def wrapped(x, *args, **kwargs):
        nonlocal calls
        calls += 1
        should_record = calls <= maximum and x.ndim == 4
        if should_record:
            print(
                "GDN_L2NORM_CALL "
                + json.dumps(
                    {
                        "call": calls,
                        "is_compiling": torch.compiler.is_compiling(),
                        "stack": traceback.format_stack(limit=8),
                        "x": _tensor_summary(x),
                        "x_chunks_dim2": _chunk_summaries(x),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        output = original(x, *args, **kwargs)
        if should_record:
            config = getattr(l2norm_module.l2norm_fwd_kernel, "best_config", None)
            dump_prefix = os.environ.get("GDN_L2NORM_DUMP_PREFIX")
            if dump_prefix:
                torch.save(
                    {"x": x.detach().cpu(), "output": output.detach().cpu()},
                    f"{dump_prefix}_call{calls}.pt",
                )
            print(
                "GDN_L2NORM_RETURN "
                + json.dumps(
                    {
                        "call": calls,
                        "args": [repr(arg) for arg in args],
                        "kwargs": {key: repr(value) for key, value in kwargs.items()},
                        "best_config": str(config),
                        "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
                        "output": _tensor_summary(output),
                        "output_chunks_dim2": _chunk_summaries(output),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return output

    l2norm_module.l2norm = wrapped


def _install_probe() -> None:
    import fla
    import fla.modules.convolution as convolution

    original = convolution.causal_conv1d
    maximum = int(os.environ.get("GDN_CONV_PROBE_MAX_CALLS", "4"))
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        x = kwargs.get("x", args[0] if args else None)
        weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
        cu_seqlens = kwargs.get("cu_seqlens")
        should_record = calls <= maximum
        if should_record:
            print(
                "GDN_CONV_CALL "
                + json.dumps(
                    {
                        "call": calls,
                        "fla_version": getattr(fla, "__version__", None),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "x": _tensor_summary(x),
                        "weight": _tensor_summary(weight),
                        "cu_seqlens": (
                            cu_seqlens.detach().cpu().tolist()
                            if cu_seqlens is not None
                            else None
                        ),
                        "activation": kwargs.get("activation"),
                        "extra_kwargs": sorted(
                            key
                            for key in kwargs
                            if key not in {"x", "weight", "cu_seqlens", "activation"}
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        result = original(*args, **kwargs)
        if should_record:
            output = result[0] if isinstance(result, tuple) else result
            print(
                "GDN_CONV_RETURN "
                + json.dumps(
                    {"call": calls, "output": _tensor_summary(output)}, sort_keys=True
                ),
                flush=True,
            )
        return result

    convolution.causal_conv1d = wrapped
    _install_l2norm_probe()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: gdn_conv_call_probe.py CORRECTNESS_PY [ARGS ...]")
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    _install_probe()
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
