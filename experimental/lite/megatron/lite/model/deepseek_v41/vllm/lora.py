# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared Lite adapter owners; native vLLM forwards and explicit STE adjoints."""
import torch
from torch.nn.functional import pad

from megatron.lite.primitive.modules.lora import lora_active

from .batching import aligned_rows, dense_mm, grouped_mm, metadata


def serving_targets(cfg, num_layers):
    """Resolve the shared Lite target set to R1's exact serving module names."""
    from megatron.lite.model.deepseek_v41.lite.lora import TARGETS, lora_layers
    from megatron.lite.primitive.modules.lora import LoraConfig, normalize_lora_config

    cfg = normalize_lora_config(cfg)
    if not cfg.enabled:
        return ()
    selected = set(
        TARGETS if cfg.target_modules == LoraConfig().target_modules else cfg.target_modules
    )
    if selected - set(TARGETS) or cfg.dropout:
        raise ValueError(f'DS4.1 LoRA supports targets {TARGETS} without dropout')
    names = {
        'wq_a': ('attn.fused_wqa_wkv',),
        'wkv': ('attn.fused_wqa_wkv',),
        'wq_b': ('attn.wq_b',),
        'wo_a': ('attn.wo_a',),
        'wo_b': ('attn.wo_b',),
        'shared_experts': tuple(f'ffn.experts.shared_{w}' for w in ('w1', 'w3', 'w2')),
        'experts': ('ffn.experts.gate_up', 'ffn.experts.down'),
    }
    start, end = lora_layers(cfg, num_layers)
    return tuple(
        sorted(
            {
                f'language_model.model.layers.{i}.{name}'
                for i in range(start, end)
                for target in selected
                for name in names[target]
            }
        )
    )


def adapter(owner, name):
    value = getattr(owner, name, None)
    return value if lora_active(value) else None


def _metadata(x):
    return metadata(x)


class _Shrink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        from vllm.lora.ops.triton_ops import lora_shrink

        ctx.save_for_backward(x, a)
        result = torch.zeros((1, x.shape[0], a.shape[0]), dtype=torch.float32, device=x.device)
        if x.shape[0]:
            lora_shrink(x.contiguous(), [a[None, None].contiguous()], result, *_metadata(x), 1.0)
        return result[0]

    @staticmethod
    def backward(ctx, grad):
        x, a = ctx.saved_tensors
        return dense_mm(grad, a).to(x.dtype), dense_mm(grad.T, x).to(a.dtype)


class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, hidden, *weights):
        from vllm.lora.ops.triton_ops import lora_expand

        ctx.save_for_backward(hidden.bfloat16(), *weights)
        ctx.hidden_dtype = hidden.dtype
        result = base.contiguous().clone()  # native expand mutates; never alias a base/VJP view
        if base.shape[0]:
            lora_expand(
                hidden.contiguous(),
                [b[None, None].contiguous() for b in weights],
                result,
                *_metadata(base),
                add_inputs=True,
            )
        return result

    @staticmethod
    def backward(ctx, grad):
        hidden, *weights = ctx.saved_tensors
        # Each output slice shares the rank; one grouped dX/dW for all slices.
        counts = (grad.shape[0],) * len(weights)
        pieces = torch.cat(grad.split([b.shape[0] for b in weights], -1), 0)
        hs = hidden.reshape(-1, hidden.shape[-1])
        bs = torch.stack(weights)
        dh = grouped_mm(pieces, bs, counts).reshape_as(hidden).to(ctx.hidden_dtype)
        dw = grouped_mm(pieces, hs, counts, wgrad=True).to(weights[0].dtype)
        return grad, dh, *dw.unbind(0)


class _GroupShrink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        from vllm.lora.ops.triton_ops import lora_shrink

        rows, groups, _ = x.shape
        flat = x.transpose(0, 1).reshape(-1, x.shape[-1]).contiguous()
        ctx.save_for_backward(flat, a)
        ctx.rows = rows
        output = x.new_zeros((1, rows * groups, a.shape[1]), dtype=torch.float32)
        if rows:
            # Treat groups as independent native adapter IDs, retaining the BI
            # split-K/reduction kernel rather than substituting a torch GEMM.
            lora_shrink(flat, [a[:, None].contiguous()], output, *metadata(flat, groups), 1.0)
        return output.reshape(groups, rows, a.shape[1])

    @staticmethod
    def backward(ctx, grad):
        x, a = ctx.saved_tensors
        counts = (ctx.rows,) * a.shape[0]
        g = grad.reshape(-1, grad.shape[-1])
        dx = grouped_mm(g, a, counts).reshape(a.shape[0], ctx.rows, a.shape[-1]).transpose(0, 1)
        da = grouped_mm(g, x, counts, wgrad=True)
        return dx.to(x.dtype), da.to(a.dtype)


def shrink(x, a):
    return _Shrink.apply(x, a)


def add(x, base, module):
    if module is None:
        return base
    shape = base.shape
    x = x.reshape(-1, x.shape[-1])
    # R1's loader folds alpha/r into B in BF16 once; export keeps raw factors.
    b = (module.lora_b * module.scale).to(module.lora_b.dtype)
    return _Expand.apply(base.reshape(-1, shape[-1]), shrink(x, module.lora_a)[None], b).reshape(
        shape
    )


def grouped_add(x, base, module):
    if module is None:
        return base
    hidden = _GroupShrink.apply(x, module.lora_a)
    weights = (module.lora_b * module.scale).to(module.lora_b.dtype)
    return _Expand.apply(base.flatten(1), hidden, *weights.unbind(0)).reshape_as(base)


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, counts):
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        if dg is None:
            raise RuntimeError('Pinned DeepGEMM is required for routed LoRA')
        if len(counts) != weights.shape[0] or sum(counts) != x.shape[0]:
            raise ValueError('One contiguous count is required per local LoRA expert')
        alignment = dg.get_theoretical_mk_alignment_for_contiguous_layout()
        positions, ids, _, capacity = aligned_rows(counts, alignment, x.device)
        padded = x.new_zeros((capacity, x.shape[1])).index_copy_(0, positions, x)
        layout = torch.full((capacity,), -1, device=x.device, dtype=torch.int32)
        layout.index_copy_(0, positions, ids)
        output = x.new_zeros((capacity, weights.shape[1]))
        if x.shape[0]:
            dg.m_grouped_bf16_gemm_nt_contiguous(
                padded.contiguous(), weights.contiguous(), output, layout
            )
        ctx.save_for_backward(x, weights)
        ctx.counts = counts
        return output.index_select(0, positions)

    @staticmethod
    def backward(ctx, grad):
        x, weights = ctx.saved_tensors
        dx = grouped_mm(grad, weights, ctx.counts).to(x.dtype)
        dw = grouped_mm(grad, x, ctx.counts, wgrad=True).to(weights.dtype)
        return dx, dw, None


def expert_gate_up(hidden, module, counts):
    b = torch.stack(module.per_expert(module.fc1_lora_b))
    b = (b * module.scale).to(b.dtype)
    width = (hidden.shape[1] + 63) // 64 * 64
    return _Grouped.apply(
        pad(hidden.bfloat16(), (0, width - hidden.shape[1])),
        pad(b, (0, width - b.shape[-1])),
        tuple(counts),
    )


def expert_down(x, module, counts):
    from .linear import linear

    a = torch.stack(module.per_expert(module.fc2_lora_a))
    rank = (module.rank + 63) // 64 * 64
    hidden = _Grouped.apply(x, pad(a, (0, 0, 0, rank - module.rank)), tuple(counts))
    b = (module.fc2_lora_b * module.scale).to(module.fc2_lora_b.dtype)
    return linear(hidden, pad(b, (0, rank - module.rank)), fp8=False).bfloat16()


class _ActivationInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_bf16, fc1, scores, limit):
        ctx.save_for_backward(fc1, scores)
        ctx.limit = limit
        return native_bf16

    @staticmethod
    def backward(ctx, grad):
        fc1, scores = ctx.saved_tensors
        from megatron.core.fusions.fused_bias_swiglu import (
            clamped_weighted_swiglu_back,
            weighted_swiglu_back,
        )

        args = (grad.float(), fc1.float(), scores.float()[:, None])
        df, ds = (
            weighted_swiglu_back(*args)
            if ctx.limit is None
            else clamped_weighted_swiglu_back(*args, ctx.limit)
        )
        ds = ds.flatten()
        return None, df.to(fc1.dtype), ds.to(scores.dtype), None


def activation_input(native_bf16, fc1, scores, limit):
    return _ActivationInput.apply(native_bf16, fc1, scores, limit)
