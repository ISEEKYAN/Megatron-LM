# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic row RMS arithmetic matching vLLM's fused QKV norm.

The reduction uses a power-of-two row tile and FP32 rsqrt, with one BF16
store after multiplying gamma. It does not depend on process-wide Torch
mean overrides. Adapted from vLLM168a040 models/common/ops/fused_qk_rmsnorm.
No cache, model, request, or rollout object is accepted.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_norm(
    x, gamma, out, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    mask = d < WIDTH
    values = tl.load(x + row * WIDTH + d, mask=mask, other=0).to(tl.float32)
    weight = tl.load(gamma + d, mask=mask, other=0).to(tl.float32)
    variance = tl.sum(values * values, 0) / WIDTH
    normalized = values * tl.rsqrt(variance + EPS) * weight
    tl.store(out + row * WIDTH + d, normalized, mask=mask)


def row_rms_norm(x, weight, eps, *, reduction_width=None):
    """Use an optional shared row tile without changing the mathematical VJP.

    Paired Q/KV norms use their maximum width for both reductions. Padding
    with zeros preserves RMS math, while tile size fixes CUDA summation order.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim < 1
        or weight.device != x.device
        or weight.dtype != torch.float32
        or weight.shape != (x.shape[-1],)
        or x.shape[-1] < 1
    ):
        raise ValueError(
            'Deployment row RMS requires CUDA BF16 rows and an FP32 gamma master'
        )
    width = x.shape[-1]
    if reduction_width is not None and (
        type(reduction_width) is not int or reduction_width < width
    ):
        raise ValueError('Reduction width must be an integer at least the row width')
    tile_width = width if reduction_width is None else reduction_width
    values = x.contiguous()
    gamma = weight.bfloat16().contiguous()
    result = torch.empty_like(values)
    rows = values.numel() // width
    if rows:
        block = triton.next_power_of_2(tile_width)
        _row_norm[(rows,)](
            values,
            gamma,
            result,
            WIDTH=width,
            BLOCK=block,
            EPS=eps,
            num_warps=8 if block >= 2048 else 4,
        )
    return result
