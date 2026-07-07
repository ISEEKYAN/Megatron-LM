#!/usr/bin/env python3
"""Replay an exact dumped Q tensor across FLA L2Norm Triton configs."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
import triton


def _sha(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _clear_autotuner(kernel) -> None:
    for name in ("cache", "config_map"):
        value = getattr(kernel, name, None)
        if hasattr(value, "clear"):
            value.clear()
    kernel.best_config = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlite-dump", required=True)
    parser.add_argument("--mbridge-dump", required=True)
    args = parser.parse_args()

    import fla.modules.l2norm as l2

    mlite = torch.load(args.mlite_dump, map_location="cpu", weights_only=True)
    mbridge = torch.load(args.mbridge_dump, map_location="cpu", weights_only=True)
    x = mlite["x"].cuda()
    expected = mbridge["output"].chunk(2, dim=2)[0].contiguous()
    assert _sha(x) == _sha(mbridge["x"].chunk(2, dim=2)[0])

    records = []
    for block_tokens in (8, 16, 32, 64, 128):
        for num_warps in (1, 2, 4, 8, 16):
            l2.l2norm_fwd_kernel.configs = [
                triton.Config({"BT": block_tokens}, num_warps=num_warps)
            ]
            _clear_autotuner(l2.l2norm_fwd_kernel)
            output = l2.l2norm(x)
            torch.cuda.synchronize()
            output_sha = _sha(output)
            records.append(
                {
                    "block_tokens": block_tokens,
                    "num_warps": num_warps,
                    "output_sha256": output_sha,
                    "matches_mbridge": output_sha == _sha(expected),
                    "matches_mlite": output_sha == _sha(mlite["output"]),
                }
            )

    x_float = x.float()
    pytorch_output = (
        x_float * torch.rsqrt(torch.sum(x_float * x_float, dim=-1, keepdim=True) + 1e-6)
    ).to(x.dtype)
    print(
        json.dumps(
            {
                "input_sha256": _sha(x),
                "mlite_output_sha256": _sha(mlite["output"]),
                "mbridge_output_sha256": _sha(expected),
                "pytorch_output_sha256": _sha(pytorch_output),
                "configs": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
