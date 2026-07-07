#!/usr/bin/env python3
"""Expose the FLA L2Norm autotune choice for one flattened token count."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch


def _sha(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()

    import fla.modules.l2norm as l2

    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(
        args.tokens,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    y = l2.l2norm(x)
    torch.cuda.synchronize()
    config = getattr(l2.l2norm_fwd_kernel, "best_config", None)
    print(
        "L2NORM_AUTOTUNE "
        + json.dumps(
            {
                "tokens": args.tokens,
                "input_prefix_sha256": _sha(x[:4096]),
                "output_prefix_sha256": _sha(y[:4096]),
                "output_sha256": _sha(y),
                "best_config": str(config),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
