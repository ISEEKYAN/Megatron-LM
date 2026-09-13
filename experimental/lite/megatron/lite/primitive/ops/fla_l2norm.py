# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixed execution policy for the original FLA L2Norm kernels.

The policy is code-versioned; runtime benchmarks never select it.
Changing it changes numerical execution and requires a new policy version.
Policy v2 selects the measured stable BT8/w8 small-feature configuration.
The large-feature branch retains four warps. Other FLA operators are outside
this adapter's scope.
"""

from __future__ import annotations

import torch

try:
    from fla.modules.l2norm import (
        l2norm_bwd_kernel,
        l2norm_bwd_kernel1,
        l2norm_fwd_kernel,
        l2norm_fwd_kernel1,
    )
    from fla.utils import input_guard

    # Unwrap only autotuning; retain the upstream JIT functions and arithmetic.
    _FWD, _BWD = l2norm_fwd_kernel.fn, l2norm_bwd_kernel.fn
    _FWD_LARGE, _BWD_LARGE = l2norm_fwd_kernel1.fn, l2norm_bwd_kernel1.fn
    HAS_FLA = True
except ImportError:
    _FWD = _BWD = _FWD_LARGE = _BWD_LARGE = None
    HAS_FLA = False

    def input_guard(fn):
        return fn


def kernel_policy() -> dict[str, int]:
    """Persist this alongside a validation run/checkpoint's implementation revision."""
    return {
        "version": 2,
        "BT": 8,
        "num_warps": 8,
        "num_stages": 3,
        "large_num_warps": 4,
    }


def assert_kernel_configs_match(per_rank: list[list[dict]]) -> None:
    """Validate actual compiled-launch records gathered by a distributed probe.

    Call on every participating rank after all_gather_object. Check agreement
    before the pin so a single-rank arithmetic mutation has the same named error
    on every rank. Never substitute configured values for observed metadata.
    """
    seen = {}
    for records in per_rank:
        assert records, "FLA_L2NORM_ACTUAL_KERNEL_RECORDS_REQUIRED"
        for record in records:
            key = (record["name"], tuple(record["shape"]), record["dtype"])
            config = record["config"]
            if key in seen:
                assert seen[key] == config, (
                    "FLA_AUTOTUNE_WINNER_MUST_MATCH_ACROSS_RANKS",
                    key,
                    seen[key],
                    config,
                )
            seen[key] = config
    for key, config in seen.items():
        expected = {"num_warps": 4, "num_stages": 3}
        if key[0] in ("l2norm_fwd_kernel", "l2norm_bwd_kernel"):
            expected.update(BT=8, num_warps=8)
        assert config == expected, ("FLA_L2NORM_PINNED_CONFIG_REQUIRED", key, config)


class _FixedL2Norm(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, eps):
        shape = x.shape
        flat = x.contiguous().view(-1, shape[-1])
        rows, width = flat.shape
        block = min(65536 // flat.element_size(), 1 << (width - 1).bit_length())
        if width > block:
            raise ValueError(
                "FLA L2Norm feature dimension exceeds the original kernel limit"
            )
        y = torch.empty_like(flat)
        rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
        if width <= 512:
            _FWD[((rows + 7) // 8,)](
                x=flat,
                y=y,
                rstd=rstd,
                eps=eps,
                T=rows,
                D=width,
                BD=block,
                NB=(rows + 65535) // 65536,
                BT=8,
                num_warps=8,
                num_stages=3,
            )
        else:
            _FWD_LARGE[(rows,)](
                x=flat,
                y=y,
                rstd=rstd,
                eps=eps,
                D=width,
                BD=block,
                num_warps=4,
                num_stages=3,
            )
        ctx.save_for_backward(y, rstd)
        ctx.shape, ctx.eps, ctx.block = shape, eps, block
        return y.view(shape)

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dy = dy.contiguous().view_as(y)
        rows, width = y.shape
        dx = torch.empty_like(y)
        if width <= 512:
            _BWD[((rows + 7) // 8,)](
                y=y,
                rstd=rstd,
                dy=dy,
                dx=dx,
                eps=ctx.eps,
                T=rows,
                D=width,
                BD=ctx.block,
                NB=(rows + 65535) // 65536,
                BT=8,
                num_warps=8,
                num_stages=3,
            )
        else:
            _BWD_LARGE[(rows,)](
                y=y,
                rstd=rstd,
                dy=dy,
                dx=dx,
                eps=ctx.eps,
                D=width,
                BD=ctx.block,
                num_warps=4,
                num_stages=3,
            )
        return dx.view(ctx.shape), None


def fixed_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _FixedL2Norm.apply(x, eps)
