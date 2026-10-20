# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram injection with vLLM's MXFP8 wkv and fused post-wkv gate kernel.

Forward: wkv is the BI MXFP8 linear (`linear.linear(fp8=True)`) over the frozen
table rows; the gate and residual add run vLLM's `_fused_engram_post_wkv_kernel`
with the launch geometry of `Engram.forward`. Backward replays the reference
formula in FP32 (gate and add); the kernel's approximate rsqrt/sqrt only shape
the forward.
"""

import torch

from .linear import linear

CLAMP = 1e-6


def reference(hidden, kv, q_weight, k_weight, eps, mask):
    """FP32 gate and residual add of the post-wkv kernel; ``mask`` False rows pass through."""
    copies, dim = hidden.shape[-2:]
    h = hidden.float()
    key = kv[..., : copies * dim].float().unflatten(-1, (copies, dim))
    value = kv[..., copies * dim :].float()
    hidden_rms = torch.rsqrt(h.square().sum(-1) / dim + eps)
    key_rms = torch.rsqrt(key.square().sum(-1) / dim + eps)
    dot = (h * q_weight.float() * k_weight.float() * key).sum(-1)
    dot = dot * (hidden_rms * key_rms * dim**-0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    if mask is not None:
        gate = gate * mask.unsqueeze(-1).to(gate.dtype)
    return h + gate.unsqueeze(-1) * value.unsqueeze(-2)


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, kv, q_master, k_master, mask, eps):
        if not hidden.is_cuda:
            raise RuntimeError("Deployment Engram gate requires CUDA")
        import triton

        from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel

        # vLLM holds q/k as BF16 (the release/export RNE of the FP32 masters). The gate sees
        # those values; their gradient goes to the masters unrounded (identity STE), since it
        # is a reduction over every token and a BF16 rounding of it costs ~2e-3.
        q_weight, k_weight = q_master.to(torch.bfloat16), k_master.to(torch.bfloat16)
        ctx.master_dtypes = (q_master.dtype, k_master.dtype)
        tokens, copies, dim = hidden.shape
        ctx.save_for_backward(hidden, kv, q_weight, k_weight, mask)
        ctx.eps = eps
        output = torch.empty_like(hidden)
        if not tokens:
            return output
        block = triton.next_power_of_2(dim)
        _fused_engram_post_wkv_kernel[(tokens * copies,)](
            hidden,
            kv,
            q_weight,
            k_weight,
            mask if mask is not None else hidden,
            output,
            tokens,
            *hidden.stride(),
            *kv.stride(),
            *q_weight.stride(),
            *k_weight.stride(),
            mask.stride(0) if mask is not None else 0,
            *output.stride(),
            eps,
            CLAMP,
            DIM=dim,
            HC_MULT=copies,
            BLOCK_SIZE=block,
            HAS_MASK=mask is not None,
            num_warps=8 if block >= 2048 else 4,
        )
        return output

    @staticmethod
    def backward(ctx, grad):
        hidden, kv, q_weight, k_weight, mask = ctx.saved_tensors
        need = ctx.needs_input_grad[:4]
        with torch.enable_grad(), torch.autocast(device_type=grad.device.type, enabled=False):
            # q/k leaves in FP32 (their BF16 values exactly): the gradient reaches the FP32
            # masters without a BF16 rounding of the leaf gradient.
            leaves = (hidden, kv, q_weight.float(), k_weight.float())
            live = [t.detach().requires_grad_(n) for t, n in zip(leaves, need)]
            out = reference(*live, ctx.eps, mask)
            wanted = [t for t, n in zip(live, need) if n]
            grads = iter(torch.autograd.grad(out, wanted, grad.float(), allow_unused=True))
        result = [next(grads) if n else None for n in need]
        dtypes = (hidden.dtype, kv.dtype, *ctx.master_dtypes)
        result = [None if g is None else g.to(dt) for g, dt in zip(result, dtypes)]
        return (*result, None, None)


def engram(hidden, rows, wkv_weight, q_weight, k_weight, mask=None, *, eps):
    """``hidden`` [T,hc,D] bf16; ``rows`` [T, (n-1)*heads*head_dim] decoded bf16 table rows
    (exact FP8 x E8M0 values); ``mask`` [T] bool or None. Returns the injected stream."""
    kv = linear(rows, wkv_weight, fp8=True)
    if mask is not None:
        mask = mask.contiguous()
    return _Gate.apply(hidden.contiguous(), kv.contiguous(), q_weight, k_weight, mask, eps)
