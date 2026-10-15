# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense CR1/2 pool + norm, from vLLM0bec's fused_compress_quant_cache.

Specialized supported-shape contract: CUDA FP32 raw[B,S,512*ratio], ratio 1
(values) or 2 (values followed by gates); gamma[512] is decoded to BF16. Pool
and RMS arithmetic remain FP32, output[B,floor(S/ratio),512] is BF16, and an
incomplete trailing group is dropped. Callers own the DS4.1 geometry and ratio;
other head widths, ratios, devices and raw dtypes are unsupported. No request
ring, cache, inference context or model class is accepted.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pool_norm(raw, gamma, out, RATIO: tl.constexpr, EPS: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 512)
    if RATIO == 2:
        rows = raw + (group * 2 + tl.arange(0, 2)[:, None]) * 1024
        values = tl.load(rows + d[None, :])
        scores = tl.load(rows + 512 + d[None, :])
        pooled = tl.sum(values * tl.softmax(scores, 0), 0)
    else:
        pooled = tl.load(raw + group * 512 + d)
    weight = tl.load(gamma + d).to(tl.float32)
    variance = tl.sum(pooled * pooled, 0) / 512
    normed = pooled * tl.rsqrt(variance + EPS) * weight
    tl.store(out + group * 512 + d, normed.to(tl.bfloat16))


def compress_norm(raw, weight, ratio, eps):
    if ratio not in (1, 2) or raw.ndim != 3 or raw.shape[-1] != 512 * ratio:
        raise ValueError('Deployment compressor requires dense CR1/2 rows')
    if (
        not raw.is_cuda
        or raw.dtype != torch.float32
        or weight.device != raw.device
        or weight.shape != (512,)
        or weight.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError('Deployment compressor requires CUDA FP32 raw and gamma[512]')
    cutoff = raw.shape[1] // ratio * ratio
    groups = raw.shape[0] * (cutoff // ratio)
    result = torch.empty(
        (raw.shape[0], cutoff // ratio, 512), device=raw.device, dtype=torch.bfloat16
    )
    if groups:
        packed = raw[:, :cutoff].contiguous()
        _pool_norm[(groups,)](
            packed, weight.bfloat16(), result, RATIO=ratio, EPS=eps, num_warps=4
        )
    return result
