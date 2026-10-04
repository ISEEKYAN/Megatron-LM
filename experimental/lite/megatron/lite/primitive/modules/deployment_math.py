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


def mhc_coefficients(
    hidden, fn, scale, base, *, copies, norm_eps, hc_eps, iterations, broadcast
):
    from megatron.lite.primitive.modules.attention.mhc import reference_mixes

    def reference(h, w, s, b):
        return reference_mixes(h, w, s, b, copies, norm_eps, hc_eps, iterations)

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
    """Generic deployment SiLU-mul: SiLU rounds to BF16 before up multiply."""

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
        return (torch.nn.functional.silu(gate) * up).to(y.dtype)

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


def mhc_joint(hidden, previous_pre, norm_weight, mixes, pending=None):
    """Keep carried post + next pre in the deployment's joint CUDA boundary.

    The owned graph carries live post operands explicitly; no module cache or
    inference model is used. Reference VJPs are the corresponding MLite math.
    """
    from megatron.lite.primitive.modules.attention import mhc

    shape = hidden.shape
    copies, width = shape[-2:]
    empty = hidden.new_empty(0)
    y, h, post, comb = (empty, empty, empty, empty) if pending is None else pending

    def reference(values, pre, gamma, fn, scale, base, output, residual, p, c):
        stream = values if pending is None else reference_post(output, residual, p, c)
        coefficients = mhc.reference_mixes(
            stream,
            fn,
            scale,
            base,
            copies,
            mixes.norm_eps,
            mixes.hc_eps,
            mixes.iterations,
        )
        collapsed = (stream.float() * pre.unsqueeze(-1)).sum(-2).to(stream.dtype)
        normed = torch.nn.functional.rms_norm(
            collapsed, (width,), decoded_bf16_master(gamma), mixes.norm_eps
        )
        return stream, *coefficients, normed

    def visible(values, pre, gamma, fn, scale, base, output, residual, p, c):
        import vllm.model_executor.kernels.mhc.tilelang as kernels

        if width % 1024 == 0 and copies == 4:
            raise NotImplementedError('Deployment math does not yet cover Mega-mHC')
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
        from megatron.lite.primitive.kernels.deployment_compressor import compress_norm

        return compress_norm(scores, master, ratio, eps)

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
