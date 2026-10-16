# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible mhc arithmetic and owned gradients."""

import importlib

import torch

from .autograd import visible_forward
from .linear import decoded_bf16_master


def _reference_mixes(hidden, fn, scale, base, mixes, *, broadcast):
    """Broadcast consumes copy zero and the sum of the per-copy weights.

    This equals the concatenated projection on repeated layer-0 token inputs.
    Its VJP describes the actual CUDA arguments, including zero dependence on
    unused copies. The composed token VJP sums back through the broadcast.
    """
    from megatron.lite.primitive.modules.attention import mhc

    copies = mixes.copies
    if not broadcast:
        return mhc.reference_mixes(
            hidden,
            fn,
            scale,
            base,
            copies,
            mixes.norm_eps,
            mixes.hc_eps,
            mixes.iterations,
        )
    token = hidden[..., 0, :].float()
    folded = fn.reshape(-1, copies, hidden.shape[-1]).sum(1)
    projected = torch.nn.functional.linear(token, folded) * torch.rsqrt(
        token.square().mean(-1, keepdim=True) + mixes.norm_eps
    )
    sizes = [copies, copies, copies**2]
    pre, post, comb = projected.split(sizes, -1)
    bp, bpost, bc = base.float().split(sizes)
    pre = torch.sigmoid(pre * scale[0] + bp) + mixes.hc_eps
    post = 2 * torch.sigmoid(post * scale[1] + bpost)
    comb = (comb * scale[2] + bc).reshape(*token.shape[:-1], copies, copies)
    return pre, post, mhc._sinkhorn_iterations(comb, mixes.iterations, mixes.hc_eps)


def mhc_coefficients(
    hidden, fn, scale, base, *, copies, norm_eps, hc_eps, iterations, broadcast
):
    from types import SimpleNamespace

    config = SimpleNamespace(
        copies=copies, norm_eps=norm_eps, hc_eps=hc_eps, iterations=iterations
    )

    def reference(h, w, s, b):
        return _reference_mixes(h, w, s, b, config, broadcast=broadcast)

    def visible(h, w, s, b):
        import vllm.model_executor.kernels.mhc.tilelang as mhc_kernels

        shape = h.shape
        residual = h.reshape(-1, copies, shape[-1]).contiguous()
        # Deployment uses the normalized epilogue. Its split reduction differs
        # from the standalone coefficient epilogue even when the input matches.
        kwargs = {
            'norm_weight': torch.ones(shape[-1], device=h.device, dtype=torch.bfloat16),
            'norm_eps': norm_eps,
        }
        if broadcast:
            w = w.reshape(-1, copies, shape[-1]).sum(1)
            kwargs['x'] = residual[:, 0].contiguous()
        post, comb, _, pre = mhc_kernels.mhc_pre_delayed_tilelang(
            residual, w, s, b, norm_eps, hc_eps, hc_eps, 2.0, iterations, **kwargs
        )
        return (
            pre.reshape(*shape[:2], copies),
            post.reshape(*shape[:2], copies),
            comb.reshape(*shape[:2], copies, copies),
        )

    return visible_forward(visible, reference, hidden, fn, scale, base)


def mhc_post(output, residual, post, comb):
    def visible(y, h, p, c):
        import vllm.model_executor.kernels.mhc.tilelang as kernels

        shape = h.shape
        out = kernels.mhc_post_tilelang(
            y.reshape(-1, shape[-1]).contiguous(),
            h.reshape(-1, shape[-2], shape[-1]).contiguous(),
            p.reshape(-1, shape[-2], 1).contiguous(),
            c.reshape(-1, shape[-2], shape[-2]).contiguous(),
        )
        return out.reshape_as(h)

    return visible_forward(visible, reference_post, output, residual, post, comb)


def reference_post(output, residual, post, comb):
    mixed = torch.einsum('bsij,bsid->bsjd', comb.float(), residual.float())
    return (mixed + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)).to(
        residual.dtype
    )


def _mega_mhc_post_pre(
    output, residual, pre, post, comb, fn, scale, base, gamma, mixes
):
    """Caller-owned Mega-mHC buffers, matching the native shifted kernel ABI."""
    from vllm.utils.deep_gemm import mega_mhc

    rows, width = output.shape
    copies = residual.shape[1]
    if rows > 1 << 20:
        raise ValueError('Mega-mHC requires at most 2**20 rows')
    stream = torch.empty_like(residual)
    next_pre = output.new_empty(rows, copies, 1, dtype=torch.float32)
    next_post = torch.empty_like(post)
    next_comb = torch.empty_like(comb)
    normed = output.new_empty(rows, width, dtype=torch.bfloat16)
    mega_mhc(
        x=output,
        residual=residual,
        shifted_prev_mix=pre.unsqueeze(-1),
        post_mix=post,
        comb_res_mix=comb,
        fn=fn,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=copies,
        hc_norm_eps=mixes.norm_eps,
        hc_pre_eps=mixes.hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=mixes.hc_eps,
        num_sinkhorn_iters=mixes.iterations,
        rmsnorm_weight=gamma.bfloat16().contiguous(),
        rmsnorm_eps=mixes.norm_eps,
        rmsnorm_scale=1.0,
        new_residual=stream,
        new_prev_mix=next_pre,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=normed,
    )
    return stream, next_post, next_comb, normed, next_pre.squeeze(-1)


def mhc_joint(hidden, previous_pre, norm_weight, mixes, pending=None):
    """Keep carried post + next pre in the deployment's joint CUDA boundary.

    The owned graph carries live post operands explicitly; no module cache or
    inference model is used. Layer-0 broadcast consumes copy zero, folded
    projection weights and its RMS; previous_pre is unused in that arm. Other
    calls consume the carried pre-mix. Reference VJPs use the same dependency
    graph, FP32 master leaves and straight-through BF16 value boundaries.
    """
    shape = hidden.shape
    copies, width = shape[-2:]
    empty = hidden.new_empty(0)
    y, h, post, comb = (empty, empty, empty, empty) if pending is None else pending

    def reference(values, pre, gamma, fn, scale, base, output, residual, p, c):
        stream = values if pending is None else reference_post(output, residual, p, c)
        broadcast = pending is None and mixes.broadcast_projection
        coefficients = _reference_mixes(
            stream, fn, scale, base, mixes, broadcast=broadcast
        )
        collapsed = (
            stream[..., 0, :]
            if broadcast
            else (stream.float() * pre.unsqueeze(-1)).sum(-2).to(stream.dtype)
        )
        normed = torch.nn.functional.rms_norm(
            collapsed, (width,), decoded_bf16_master(gamma), mixes.norm_eps
        )
        return stream, *coefficients, normed

    def visible(values, pre, gamma, fn, scale, base, output, residual, p, c):
        import vllm.model_executor.kernels.mhc.tilelang as kernels

        kwargs = dict(norm_weight=gamma.bfloat16(), norm_eps=mixes.norm_eps)
        if pending is None:
            stream = values.reshape(-1, copies, width).contiguous()
            if mixes.broadcast_projection:
                fn = fn.reshape(-1, copies, width).sum(1)
                kwargs['x'] = stream[:, 0].contiguous()
            else:
                kwargs['pre_mix'] = pre.reshape(-1, copies).contiguous()
            next_post, next_comb, normed, next_pre = kernels.mhc_pre_delayed_tilelang(
                stream,
                fn,
                scale,
                base,
                mixes.norm_eps,
                mixes.hc_eps,
                mixes.hc_eps,
                2.0,
                mixes.iterations,
                **kwargs,
            )
        elif width % 1024 == 0 and copies == 4:
            stream, next_post, next_comb, normed, next_pre = _mega_mhc_post_pre(
                output.reshape(-1, width).contiguous(),
                residual.reshape(-1, copies, width).contiguous(),
                pre.reshape(-1, copies).contiguous(),
                p.reshape(-1, copies, 1).contiguous(),
                c.reshape(-1, copies, copies).contiguous(),
                fn,
                scale,
                base,
                gamma,
                mixes,
            )
        else:
            stream, next_post, next_comb, normed, next_pre, _ = (
                kernels.mhc_fused_post_pre_delayed_tilelang(
                    output.reshape(-1, width).contiguous(),
                    residual.reshape(-1, copies, width).contiguous(),
                    p.reshape(-1, copies, 1).contiguous(),
                    c.reshape(-1, copies, copies).contiguous(),
                    fn,
                    scale,
                    base,
                    mixes.norm_eps,
                    mixes.hc_eps,
                    mixes.hc_eps,
                    2.0,
                    mixes.iterations,
                    pre_mix=pre.reshape(-1, copies).contiguous(),
                    **kwargs,
                )
            )
        return (
            stream.reshape_as(values),
            next_pre.reshape(*shape[:2], copies),
            next_post.reshape(*shape[:2], copies),
            next_comb.reshape(*shape[:2], copies, copies),
            normed.reshape(*shape[:2], width),
        )

    return visible_forward(
        visible,
        reference,
        hidden,
        previous_pre,
        norm_weight,
        mixes.fn,
        mixes.scale,
        mixes.base,
        y,
        h,
        post,
        comb,
    )


def hc_collapse(hidden, pre):
    def visible(h, p):
        kernels = importlib.import_module('vllm.model_executor.kernels.batch_invariant')

        out = kernels.hc_collapse(
            h.reshape(-1, h.shape[-2], h.shape[-1]).contiguous(),
            p.reshape(-1, p.shape[-1]).contiguous(),
        )
        return out.reshape(*h.shape[:2], h.shape[-1])

    def reference(h, p):
        return (h.float() * p.unsqueeze(-1)).sum(-2).to(h.dtype)

    return visible_forward(visible, reference, hidden, pre)
