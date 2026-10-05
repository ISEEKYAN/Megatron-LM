# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional CUDA visible-math providers, with MLite-owned reference VJPs.

The DS41 EP1 precision recipe selects these explicitly. They consume live
parameters, never rollout model objects or recorded outputs. CUDA forward
requires vLLM's generic mHC/FlashMLA kernel package; no rollout, receiver or
legacy batch_invariant module is imported. Reference VJPs retain FP32 leaves.
Higher order derivatives are not supported.
"""
import importlib

import torch
from torch.autograd.function import once_differentiable


class _Visible(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, reference, *inputs):
        ctx.reference = reference
        ctx.save_for_backward(*inputs)
        return visible(*inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        tensors = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast(
            device_type=tensors[0].device.type, enabled=False
        ):
            leaves = [
                x.detach().requires_grad_(needed)
                for x, needed in zip(tensors, ctx.needs_input_grad[2:])
            ]
            outputs = ctx.reference(*leaves)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            active = [
                (y, g)
                for y, g in zip(outputs, grad_outputs)
                if g is not None and y.requires_grad
            ]
            wanted = [x for x in leaves if x.requires_grad]
            grads = (
                torch.autograd.grad(
                    [y for y, _ in active],
                    wanted,
                    [g for _, g in active],
                    allow_unused=True,
                )
                if active and wanted
                else [None] * len(wanted)
            )
        cursor = iter(grads)
        return (
            None,
            None,
            *(next(cursor) if x.requires_grad else None for x in leaves),
        )


def visible_forward(visible, reference, *inputs):
    return _Visible.apply(visible, reference, *inputs)


def router_topk(logits, bias, *, topk, scaling_factor):
    """CUDA selection order and FP32 normalization, with an owned score VJP.

    Bias only selects experts. Keep the selected slots fixed for backward, as
    in the reference router; never backpropagate through ranking or bias.
    """
    indices = torch.empty(
        (logits.shape[0], topk), device=logits.device, dtype=torch.int32
    )

    def visible(x):
        from vllm import _custom_ops as ops

        kernels = importlib.import_module(
            'vllm.model_executor.layers.fused_moe.router.dsv4_topk'
        )

        scores, correction = x.contiguous(), bias.contiguous()
        if kernels.can_use_dsv4_topk(scores, correction, topk, True, indices.dtype):
            weights, selected = kernels.dsv4_topk(
                scores, correction, indices.dtype, scaling_factor
            )
            indices.copy_(selected)
            return weights
        weights = torch.empty_like(indices, dtype=torch.float32)
        token_expert_indices = torch.empty_like(indices)
        ops.topk_hash_softplus_sqrt(
            weights,
            indices,
            token_expert_indices,
            scores,
            renormalize=True,
            routed_scaling_factor=scaling_factor,
            e_score_correction_bias=correction,
        )
        return weights

    def reference(x):
        scores = torch.nn.functional.softplus(x.float()).sqrt()
        selected = scores.gather(1, indices.long())
        return selected / selected.sum(-1, keepdim=True) * scaling_factor

    weights = visible_forward(visible, reference, logits)
    return weights, indices.long()


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


def sparse_attention(
    q, kv, sink, mask, main_indices, *, main_size, window_size, width, scale
):
    # Native prefill gathers compressed KV before SWA and preserves selected
    # index order. Construct the same physical row order from the owned state.
    reordered = torch.cat([kv[:, window_size:], kv[:, :window_size]], 1)
    indices = torch.full((*q.shape[:2], width), -1, device=q.device, dtype=torch.int32)
    lengths = torch.zeros(q.shape[:2], device=q.device, dtype=torch.int32)
    for batch in range(q.shape[0]):
        for token in range(q.shape[1]):
            selected = (
                []
                if main_indices is None
                else [
                    v for v in main_indices[batch, token].tolist() if 0 <= v < main_size
                ]
            )
            selected += [
                main_size + v
                for v in torch.where(mask[batch, token, :window_size])[0].tolist()
            ]
            indices[batch, token, : len(selected)] = torch.tensor(
                selected, device=q.device, dtype=torch.int32
            )
            lengths[batch, token] = len(selected)

    def reference(query, values, bias):
        rows = []
        for batch in range(query.shape[0]):
            gathered = values[batch][indices[batch].clamp_min(0).long()]
            logits = (
                torch.einsum('shd,std->sht', query[batch].float(), gathered.float())
                * scale
            )
            logits = logits.masked_fill((indices[batch] < 0).unsqueeze(1), -torch.inf)
            probabilities = torch.cat(
                [logits, bias.expand(query.shape[1], -1).unsqueeze(-1)], -1
            ).softmax(-1)[..., :-1]
            rows.append(
                torch.einsum('sht,std->shd', probabilities, gathered.float()).to(
                    query.dtype
                )
            )
        return torch.stack(rows)

    def visible(query, values, bias):
        from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        rows = []
        heads = query.shape[2]
        padded_heads = 64 if heads <= 64 else 128
        padded_sink = torch.nn.functional.pad(
            bias, (0, padded_heads - heads)
        ).contiguous()
        for batch in range(query.shape[0]):
            padded_q = torch.nn.functional.pad(
                query[batch], (0, 0, 0, padded_heads - heads)
            ).contiguous()
            output = torch.empty_like(padded_q)
            flash_mla_sparse_fwd(
                q=padded_q,
                kv=values[batch].unsqueeze(1).contiguous(),
                indices=indices[batch].unsqueeze(1),
                sm_scale=scale,
                attn_sink=padded_sink,
                topk_length=lengths[batch],
                out=output,
            )
            rows.append(output[:, :heads])
        return torch.stack(rows)

    return visible_forward(visible, reference, q, reordered, sink)


class _BF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, persistent):
        compute = weight.bfloat16()
        ctx.save_for_backward(x, compute)
        ctx.input_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1]).bfloat16()
        if persistent:
            # Generic numerical kernel, matching W5's fixed-tile FP32 router.
            import vllm.model_executor.determinism.batch_invariant as bi_kernels

            output = bi_kernels.matmul_persistent(
                flat, compute.T, out_dtype=torch.float32
            )
        else:
            output = torch.mm(flat, compute.T, out_dtype=torch.float32)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, compute = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (grad.float() @ compute.float()).to(ctx.input_dtype)
            dw = (
                grad.reshape(-1, compute.shape[0]).float().T
                @ x.reshape(-1, compute.shape[1]).float()
            )
        return dx, dw, None


def bf16_fp32_linear(x, weight, *, persistent):
    return _BF16Linear.apply(x, weight, persistent)


def shared_swiglu(gate_up, limit):
    """SiLU rounds to input dtype before multiplying up; rounding uses an STE.

    Backward differentiates FP32 SiLU and clamp. The up VJP uses the rounded
    activation value; the gate VJP uses the smooth FP32 derivative. This is an
    explicitly owned surrogate, not finite differences of the discrete kernel.
    """

    def visible(y):
        out = torch.empty(
            (*y.shape[:-1], y.shape[-1] // 2), device=y.device, dtype=y.dtype
        )
        if limit > 0:
            torch.ops._C.silu_and_mul_with_clamp(out, y, limit, 1.0, 0.0)
        else:
            torch.ops._C.silu_and_mul(out, y)
        return out

    def reference(y):
        gate, up = y.float().chunk(2, -1)
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        activated = torch.nn.functional.silu(gate)
        rounded = activated + (activated.to(y.dtype).float() - activated).detach()
        return (rounded * up).to(y.dtype)

    return visible_forward(visible, reference, gate_up)


def decoded_bf16_master(weight):
    """BF16 deployment value, FP32 master gradient without a BF16 gradient cast."""
    return weight + (weight.bfloat16().float() - weight).detach()


def rms_norm(x, weight, shape, eps):
    def reference(values, master):
        return torch.nn.functional.rms_norm(
            values, shape, decoded_bf16_master(master), eps
        )

    def visible(values, master):
        import vllm.model_executor.determinism.batch_invariant as bi_kernels

        out = bi_kernels.rms_norm_batch_invariant(
            values.reshape(-1, values.shape[-1]), master.bfloat16(), eps
        )
        return out.reshape_as(values)

    return visible_forward(visible, reference, x, weight)


def qkv_rms_norm(x, weight, shape, eps, *, reduction_width=None):
    """Fused-QKV row arithmetic with an explicit optional paired reduction tile.

    The FP32/STE reference owns the unchanged mathematical VJP.
    """
    from megatron.lite.primitive.kernels import deployment_rms_norm

    if tuple(shape) != (x.shape[-1],):
        raise ValueError('Deployment QKV RMS requires one normalized row dimension')

    def reference(values, master):
        return torch.nn.functional.rms_norm(
            values, shape, decoded_bf16_master(master), eps
        )

    return visible_forward(
        lambda values, master: deployment_rms_norm.row_rms_norm(
            values, master, eps, reduction_width=reduction_width
        ),
        reference,
        x,
        weight,
    )


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
        kernels = importlib.import_module('vllm.model_executor.kernels.mhc.triton')

        out = kernels.hc_collapse_triton(
            h.reshape(-1, h.shape[-2], h.shape[-1]).contiguous(),
            p.reshape(-1, p.shape[-1]).contiguous(),
        )
        return out.reshape(*h.shape[:2], h.shape[-1])

    def reference(h, p):
        return (h.float() * p.unsqueeze(-1)).sum(-2).to(h.dtype)

    return visible_forward(visible, reference, hidden, pre)


def compressor(x, wkv, wgate, gamma, ratio, eps):
    # Match the deployment's merged BF16-input/weight GEMM, FP32 output.
    weights = wkv if wgate is None else torch.cat((wkv, wgate))
    raw = bf16_fp32_linear(x, weights, persistent=False)

    def visible(scores, master):
        from megatron.lite.primitive.kernels import deployment_compressor

        return deployment_compressor.compress_norm(scores, master, ratio, eps)

    def reference(scores, master):
        cutoff = scores.shape[1] // ratio * ratio
        values = scores[:, :cutoff, :512]
        if ratio == 2:
            gates = scores[:, :cutoff, 512:].unflatten(1, (-1, 2))
            values = (values.unflatten(1, (-1, 2)) * gates.softmax(2)).sum(2)
        # Native pooling retains FP32 through normalization; only latent rounds.
        return torch.nn.functional.rms_norm(
            values, (512,), decoded_bf16_master(master), eps
        ).bfloat16()

    return visible_forward(visible, reference, raw, gamma)


def log_softmax(logits):
    """Fixed-row CUDA log probabilities with an owned FP32 logsumexp VJP.

    The input is the materialized model-owned head output. No projection or
    quantizer is repeated here. CPU uses the ordinary Torch implementation.
    """
    if not logits.is_cuda:
        return torch.log_softmax(logits.float(), dim=-1)

    def visible(x):
        from vllm.model_executor.determinism.batch_invariant import log_softmax

        return log_softmax(x.float(), dim=-1)

    def reference(x):
        value = x.float()
        return value - torch.logsumexp(value, dim=-1, keepdim=True)

    return visible_forward(visible, reference, logits)


def engram_reference(hidden, kv, query, key, *, eps, token_mask=None):
    """Owned FP32/decoded-BF16 STE reference; retain every live master leaf."""
    copies, dim = hidden.shape[-2:]
    keys, values = kv.float().split((copies * dim, dim), -1)
    keys = keys.unflatten(-1, (copies, dim))
    h = hidden.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        keys.square().mean(-1) + eps
    )
    dot = (
        (h * keys * decoded_bf16_master(query) * decoded_bf16_master(key)).sum(-1)
        * rstd
        * dim**-0.5
    )
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * values.unsqueeze(-2)).to(hidden.dtype)


def engram_post(hidden, kv, query, key, *, eps, token_mask=None):
    """Native visible BF16 gate/residual, with the declared owned reference VJP.

    Query/key masters remain FP32 leaves. Visible arithmetic consumes their
    BF16 deployment values. No gradient is taken through the mask or hashing.
    Higher-order derivatives follow the same unsupported contract as _Visible.
    """

    def reference(h, v, q, k):
        return engram_reference(h, v, q, k, eps=eps, token_mask=token_mask)

    def visible(h, v, q, k):
        from megatron.lite.primitive.kernels.deployment_engram import post_wkv

        shape = h.shape
        result = post_wkv(
            h.reshape(-1, shape[-2], shape[-1]),
            v.reshape(-1, v.shape[-1]),
            q.bfloat16(),
            k.bfloat16(),
            eps=eps,
            token_mask=None if token_mask is None else token_mask.reshape(-1),
        )
        return result.reshape_as(h)

    return visible_forward(visible, reference, hidden, kv, query, key)
