# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-stream overlap kernels with the W9-b5 owned mHC graph VJP."""

import torch


def _post_backward(grad, output, residual, post, comb):
    """Adapt Core's H_res.T @ residual adjoint; comb already has that orientation."""
    from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_bwd

    copies, width = residual.shape[-2:]
    dc, dh, dp, dy, _ = _triton_h_post_bda_bwd(
        grad.reshape(-1, 1, copies, width),
        comb.reshape(-1, 1, copies, copies),
        residual.reshape(-1, 1, copies, width),
        post.reshape(-1, 1, copies),
        output.reshape(-1, 1, width),
        None,
    )
    return dy.reshape_as(output), dh.reshape_as(residual), dp.reshape_as(post), dc.reshape_as(comb)


def _rms_adjoint(grad, x, inverse, weight=None, weight_grad=False):
    """Use Torch's existing CUDA RMS adjoint; CPU is a math-reference path."""
    if x.is_cuda:
        dx, dw = torch.ops.aten._fused_rms_norm_backward(
            grad.contiguous(),
            x.contiguous(),
            [x.shape[-1]],
            inverse.contiguous(),
            weight,
            [True, weight_grad],
        )
        return dx, dw
    scaled = grad if weight is None else grad * weight
    dx = inverse * (scaled - x * inverse.square() * (scaled * x).mean(-1, keepdim=True))
    dw = (grad * x * inverse).reshape(-1, x.shape[-1]).sum(0) if weight_grad else None
    return dx, dw


def _aggregate_adjoint(grad, stream, pre):
    """Core7b's existing cuTile aggregate adjoint, with device-owned layout."""
    if stream.is_cuda:
        from megatron.core.fusions.fused_mhc_kernels import _cutile_h_aggregate_bwd

        copies, width = stream.shape[-2:]
        dh, dp = _cutile_h_aggregate_bwd(
            grad.reshape(-1, 1, width),
            stream.reshape(-1, 1, copies, width),
            pre.reshape(-1, 1, copies),
        )
        return dh.reshape_as(stream), dp.reshape_as(pre)
    g = grad.float().unsqueeze(-2)
    return (g * pre.unsqueeze(-1)).to(stream.dtype), (g * stream.float()).sum(-1).to(pre.dtype)


def _aggregate_recompute(stream, pre):
    if stream.is_cuda:
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_aggregate_fwd

        copies, width = stream.shape[-2:]
        return _triton_h_aggregate_fwd(
            stream.reshape(-1, 1, copies, width), pre.reshape(-1, 1, copies)
        ).reshape(*stream.shape[:-2], width)
    return (stream.float() * pre.unsqueeze(-1)).sum(-2).to(stream.dtype)


def _joint_backward(saved, options, grads, required):
    """Differentiate each boundary once, using Core's fused Sinkhorn and Post adjoints."""
    from megatron.core.fusions.fused_mhc_kernels import _triton_sinkhorn_bwd

    values, pre, gamma, fn, scale, base, output, residual, post, comb, stream = saved
    eps, _, sinkhorn_eps, iterations, broadcast, pending = options
    copies, width = stream.shape[-2:]
    token = (stream[..., 0, :] if broadcast else stream.flatten(-2)).float()
    weight = (fn.reshape(-1, copies, width).sum(1) if broadcast else fn).float()
    inverse = torch.rsqrt(token.square().mean(-1, keepdim=True) + eps)
    projected = torch.nn.functional.linear(token, weight) * inverse
    a, b, c = projected.split((copies, copies, copies * copies), -1)
    ba, bb, bc = base.float().split((copies, copies, copies * copies))
    sa, sb = torch.sigmoid(a * scale[0] + ba), torch.sigmoid(b * scale[1] + bb)
    gs, ga, gb, gc, gn = grads
    da = torch.zeros_like(a) if ga is None else ga.float() * sa * (1 - sa)
    db = torch.zeros_like(b) if gb is None else gb.float() * 2 * sb * (1 - sb)
    if gc is None:
        dc = torch.zeros_like(c)
    else:
        logits = (c * scale[2] + bc).reshape(*token.shape[:-1], copies, copies)
        # Core's adjoint takes exp(logits - row_max), before the first row normalization.
        initial = torch.exp(logits - logits.amax(-1, keepdim=True))
        dc = _triton_sinkhorn_bwd(gc.float(), initial, iterations, sinkhorn_eps).flatten(-2)
    dbase = (
        torch.cat((da, db, dc), -1).reshape(-1, base.numel()).sum(0).to(base.dtype)
        if required[5]
        else None
    )
    dscale = (
        torch.stack(((da * a).sum(), (db * b).sum(), (dc * c).sum())).to(scale.dtype)
        if required[4]
        else None
    )
    dp = torch.cat((da * scale[0], db * scale[1], dc * scale[2]), -1)
    dtoken, _ = _rms_adjoint(torch.matmul(dp, weight), token, inverse)
    dfn = None
    if required[3]:
        dlinear = dp * inverse
        dweight = dlinear.reshape(-1, weight.shape[0]).T @ token.reshape(-1, token.shape[-1])
        dfn = (
            dweight.to(fn.dtype).unsqueeze(1).expand(-1, copies, -1).reshape_as(fn)
            if broadcast
            else dweight.to(fn.dtype)
        )

    # Round at the same boundary as the eager adjoint and release wide FP32 temporaries.
    dtoken = dtoken.to(stream.dtype)
    del token, inverse, projected
    collapsed = stream[..., 0, :] if broadcast else _aggregate_recompute(stream, pre)
    x = collapsed.float()
    rn = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    gn = torch.zeros_like(x) if gn is None else gn.float()
    dx, dgamma = _rms_adjoint(gn, x, rn, gamma.bfloat16().float(), required[2])
    dx = dx.to(stream.dtype)
    dgamma = dgamma.to(gamma.dtype) if required[2] else None
    del x, rn, gn, collapsed
    if broadcast:
        dstream = torch.zeros_like(stream)
        dstream[..., 0, :] = dtoken.to(stream.dtype) + dx
        dpre = None
    else:
        aggregate_stream, aggregate_pre = _aggregate_adjoint(dx, stream, pre)
        dstream = dtoken.reshape_as(stream).to(stream.dtype) + aggregate_stream
        dpre = aggregate_pre if required[1] else None
    if gs is not None:
        dstream = dstream + gs
    if pending:
        dy, dh, dpost, dcomb = _post_backward(dstream, output, residual, post, comb)
        return None, dpre, dgamma, dfn, dscale, dbase, dy, dh, dpost, dcomb
    return dstream.to(values.dtype), dpre, dgamma, dfn, dscale, dbase, None, None, None, None


class _Joint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, pre, gamma, fn, scale, base, output, residual, post, comb, options):
        if not values.is_cuda:
            raise RuntimeError("Deployment mHC overlap requires CUDA")
        from vllm.models.deepseek_v41.nvidia.ops.mhc import (
            mhc_pre_delayed_overlap,
            mhc_shifted_post_pre,
        )

        eps, hc_eps, sinkhorn_eps, iterations, broadcast, pending = options
        copies, width = values.shape[-2:]
        saved_fn = fn
        ctx.set_materialize_grads(False)
        ctx.options = options
        kwargs = dict(
            norm_weight=gamma.bfloat16(), norm_eps=eps, stream=torch.cuda.current_stream()
        )
        if pending:
            stream, new_post, new_comb, normed, new_pre, _ = mhc_shifted_post_pre(
                output.reshape(-1, width).contiguous(),
                residual.reshape(-1, copies, width).contiguous(),
                post.reshape(-1, copies, 1).contiguous(),
                comb.reshape(-1, copies, copies).contiguous(),
                fn,
                scale,
                base,
                eps,
                hc_eps,
                hc_eps,
                2.0,
                iterations,
                pre_mix=pre.reshape(-1, copies).contiguous(),
                **kwargs,
            )
        else:
            stream = values.reshape(-1, copies, width).contiguous()
            if broadcast:
                fn = fn.reshape(-1, copies, width).sum(1)
                kwargs["x"] = stream[:, 0].contiguous()
            else:
                kwargs["pre_mix"] = pre.reshape(-1, copies).contiguous()
            new_post, new_comb, normed, new_pre = mhc_pre_delayed_overlap(
                stream, fn, scale, base, eps, hc_eps, sinkhorn_eps, 2.0, iterations, **kwargs
            )
        prefix = values.shape[:-2]
        ctx.save_for_backward(
            values,
            pre,
            gamma,
            saved_fn,
            scale,
            base,
            output,
            residual,
            post,
            comb,
            stream.reshape_as(values),
        )
        return (
            stream.reshape_as(values),
            new_pre.reshape(*prefix, copies),
            new_post.reshape(*prefix, copies),
            new_comb.reshape(*prefix, copies, copies),
            normed.reshape(*prefix, width),
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        required = ctx.needs_input_grad[:10]
        with torch.autocast(device_type=ctx.saved_tensors[0].device.type, enabled=False):
            gradients = _joint_backward(ctx.saved_tensors, ctx.options, grad_outputs, required)
        return (*[g if need else None for g, need in zip(gradients, required)], None)


def joint(values, previous_pre, gamma, mixes, *, pending=None, broadcast=False):
    empty = values.new_empty(0)
    operands = (empty, empty, empty, empty) if pending is None else pending
    options = (
        mixes.norm_eps,
        mixes.compute_h_eps,
        mixes.sinkhorn_eps,
        mixes.sinkhorn_iterations,
        broadcast,
        pending is not None,
    )
    scale = torch.cat((mixes.alpha_pre, mixes.alpha_post, mixes.alpha_res))
    return _Joint.apply(
        values,
        previous_pre,
        gamma,
        mixes.mapping_proj.weight,
        scale,
        mixes.bias,
        *operands,
        options,
    )


class _Post(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, residual, post, comb):
        if not residual.is_cuda:
            raise RuntimeError("Deployment mHC post requires CUDA")
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

        ctx.save_for_backward(output, residual, post, comb)
        copies, width = residual.shape[-2:]
        return mhc_post_tilelang(
            output.reshape(-1, width).contiguous(),
            residual.reshape(-1, copies, width).contiguous(),
            post.reshape(-1, copies, 1).contiguous(),
            comb.reshape(-1, copies, copies).contiguous(),
        ).reshape_as(residual)

    @staticmethod
    def backward(ctx, grad):
        output, residual, post, comb = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            return _post_backward(grad, output, residual, post, comb)


def post_mix(output, residual, post, comb):
    return _Post.apply(output, residual, post, comb)


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, pre):
        if not values.is_cuda:
            raise RuntimeError("Deployment mHC collapse requires CUDA")
        from vllm.model_executor.kernels.batch_invariant import hc_collapse

        ctx.save_for_backward(values, pre)
        copies, width = values.shape[-2:]
        return hc_collapse(
            values.reshape(-1, copies, width).contiguous(), pre.reshape(-1, copies).contiguous()
        ).reshape(*values.shape[:-2], width)

    @staticmethod
    def backward(ctx, grad):
        values, pre = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            g = grad.float().unsqueeze(-2)
            dh, dp = g * pre.float().unsqueeze(-1), (g * values.float()).sum(-1)
        return dh.to(values.dtype), dp.to(pre.dtype)


def collapse(values, pre):
    return _Collapse.apply(values, pre)


def aggregate(module, values, pre, out=None):
    """Core's aggregation ABI, with the inference BF16 collapse rounding."""
    streams = values.unflatten(-1, (module.n, module.hidden_size)).bfloat16()
    result = collapse(streams, pre).to(values.dtype)
    return result if out is None else out.copy_(result)
