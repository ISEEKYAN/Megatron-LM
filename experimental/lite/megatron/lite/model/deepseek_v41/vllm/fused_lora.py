# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""R1 MegaMoE LoRA hooks; decoded grouped replay is an approximate adjoint.

Only the native kernel can add gate/up delta before L1's BF16 rounding. The
adjoint recomputes the older BF16 base-plus-delta graph because the hook exports
post-SwiGLU activations, not its FP32 L1 accumulator. This is not an exact VJP of
that accumulator. No vLLM or DeepGEMM kernel is copied here.
"""
from contextlib import nullcontext

import torch

from megatron.lite.primitive.modules.lora import disable_lora_adapters

from . import experts
from .batching import metadata
from .lora import adapter, scaled_b, shrink


def _layout(codes, scale, *, first):
    """Vectorized release -> native L1/L2 layout, with no decoded weight copy."""
    e, n, k = codes.shape
    sf = scale.view(torch.uint8).contiguous().view(torch.int32)
    q = sf.shape[-1]
    if first:
        codes = codes.view(e, 2, n // 16, 8, k).permute(0, 2, 1, 3, 4).reshape(e, n, k)
        sf = sf.view(e, 2, n // 128, 4, 2, 8, q).permute(0, 2, 4, 1, 5, 3, 6).reshape(e, n, q)
    else:
        sf = sf.view(e, n // 128, 4, 32, q).transpose(2, 3).reshape(e, n, q)
    # SF is N-major. Allocate only this layer's kernel-layout payload, discarded
    # after the native forward; the model keeps its original release bytes.
    sf = sf.transpose(1, 2).contiguous().transpose(1, 2)
    return codes.contiguous(), sf


def frozen_weights(owner):
    from .frozen import FrozenWeight

    a = [getattr(owner.experts.linear_fc1, f'weight{i}') for i in range(owner.num_local_experts)]
    b = [getattr(owner.experts.linear_fc2, f'weight{i}') for i in range(owner.num_local_experts)]
    shared = owner.shared_experts
    if shared is None:
        raise NotImplementedError('Fused LoRA hooks require the native fused shared expert')
    s1, s2 = shared.linear_fc1.weight, shared.linear_fc2.weight
    if not all(isinstance(w, FrozenWeight) for w in (*a, *b, s1, s2)):
        raise ValueError('Fused LoRA currently requires an immutable release-native base')
    if any(w.kind != 'mxfp4' for w in (*a, *b)) or any(w.kind != 'fp8_block' for w in (s1, s2)):
        raise ValueError('MegaMoE hooks require routed MXFP4 and shared block-FP8 weights')
    routed = tuple(
        _layout(torch.stack([w.codes for w in ws]), torch.stack([w.scale for w in ws]), first=f)
        for ws, f in ((a, True), (b, False))
    )
    shared_weights = []
    for w, first in ((s1, True), (s2, False)):
        # Shared FP8 scales are 32x32 blocks, expanded along rows, packed along K.
        scale = w.scale.view(torch.uint8).repeat_interleave(32, dim=0).unsqueeze(0)
        c, sf = _layout(w.codes.unsqueeze(0), scale, first=first)
        shared_weights.append((c.squeeze(0), sf.squeeze(0)))
    return (*routed, *shared_weights)


def factors(owner):
    routed = adapter(owner.experts, 'lora')
    shared = owner.shared_experts
    slots = [adapter(shared, f'{w}_lora') for w in ('w1', 'w3', 'w2')]
    if routed is None or any(s is None for s in slots):
        raise NotImplementedError('Fused LoRA requires routed and all shared-expert targets')
    rank = routed.rank
    if rank % 8 or not 0 < rank <= 64 or any(s.lora_a.shape[0] != rank for s in slots):
        raise ValueError('Native LoRA hook ranks must agree, be divisible by 8 and at most 64')
    return routed, *slots


def shrink_slices(x, weights):
    """One serving shrink launch for the gate/up and shared w1/w3 slices."""
    from vllm.lora.ops.triton_ops import lora_shrink

    rank = weights[0].shape[0]
    if any(w.shape != (rank, x.shape[-1]) for w in weights):
        raise ValueError('Fused shrink slices must have a common rank and input width')
    out = torch.zeros((len(weights), x.shape[0], rank), dtype=torch.float32, device=x.device)
    if x.shape[0]:
        lora_shrink(
            x.contiguous(),
            [w.detach()[None, None].contiguous() for w in weights],
            out,
            *metadata(x),
            1.0,
        )
    return out


def _launch(owner, x, scores, ids, packed, buffer, max_tokens):
    from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import bf16_fixed_mm
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe_lora import (
        _add_delta_kernel,
        _down_inputs_kernel,
        _down_shrink_return_kernel,
        _stage_xa_kernel,
    )
    from vllm.triton_utils import triton
    from vllm.utils.deep_gemm import _import_deep_gemm

    dg = _import_deep_gemm()
    gu, w1, w3, w2 = factors(owner)
    t, rank = x.shape[0], gu.rank
    on = torch.ones(t, dtype=torch.bool, device=x.device)
    # Same batched Punica shrink and FP32 -> BF16 stage kernel as serving.
    xa = shrink_slices(x, (gu.fc1_lora_a, w1.lora_a, w3.lora_a))
    if t:
        _stage_xa_kernel[(t,)](xa, on, buffer.lora_xa, t, RANK=rank)
    bs = torch.stack(gu.per_expert(gu.fc1_lora_b))
    kwargs = dict(
        lora_b_gate_up=dg.transform_lora_b_for_mega_moe((bs * gu.scale).bfloat16()),
        lora_b_shared_gate_up=dg.transform_shared_lora_b_for_mega_moe(scaled_b(w1), scaled_b(w3)),
    )
    y = experts.launch(
        x,
        scores,
        ids,
        packed,
        buffer,
        owner.config.activation_func_clamp_value,
        owner.ep_group.size(),
        **kwargs,
    )
    down_a = torch.stack(gu.per_expert(gu.fc2_lora_a)).contiguous()
    grid = (
        triton.cdiv(owner.ep_group.size() * max_tokens * ids.shape[1], 16)
        + owner.num_local_experts,
    )
    _down_shrink_return_kernel[grid](
        buffer.lora_act,
        buffer.lora_expert_info,
        buffer.lora_pool_meta,
        down_a,
        down_a.stride(0),
        down_a.stride(1),
        buffer.lora_peer_bases,
        buffer.lora_ret_offset,
        buffer.num_max_tokens_per_rank,
        INTER=gu.fc2_lora_a.weight0.shape[1],
        RANK=rank,
        NUM_EXPERTS=owner.num_local_experts,
        BLOCK_M=16,
        BLOCK_K=64,
    )
    shared_down = shrink(buffer.lora_shared_act[:t], w2.lora_a)
    if owner.ep_group.size() > 1:
        buffer.handle.barrier()
    inputs = x.new_empty((t, 128))
    if t:
        _down_inputs_kernel[(t,)](
            buffer.lora_ret,
            ids,
            on,
            shared_down,
            inputs,
            buffer.num_max_tokens_per_rank,
            TOPK=ids.shape[1],
            RANK=rank,
            K_OUT=128,
        )
    b = x.new_zeros((x.shape[1], 128))
    b[:, :rank].copy_((gu.fc2_lora_b * gu.scale).bfloat16())
    b[:, rank : 2 * rank].copy_(scaled_b(w2))
    delta = bf16_fixed_mm(inputs, b)
    if y.numel():
        _add_delta_kernel[(triton.cdiv(y.numel(), 1024),)](y, delta, y.numel(), BLOCK=1024)
    return y


class _Hooks(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scores, ids, owner, *parameters):
        ctx.owner = owner
        ctx.save_for_backward(x, scores, ids, *parameters)
        ctx.base_only = not parameters
        rank = 0 if ctx.base_only else factors(owner)[0].rank
        capacity = 8192
        buffer = experts.symm_buffer(
            owner.ep_group,
            owner.num_local_experts * owner.ep_group.size(),
            capacity,
            ids.shape[1],
            x.shape[1],
            owner.experts.linear_fc1.weight0.shape[0] // 2,
            lora_rank=rank,
        )
        max_rows = torch.tensor(x.shape[0], device=x.device)
        torch.distributed.all_reduce(max_rows, torch.distributed.ReduceOp.MAX, group=owner.ep_group)
        max_rows = int(max_rows)
        packed = frozen_weights(owner)
        outputs = []
        for i in range(max(1, (max_rows + capacity - 1) // capacity)):
            sl = slice(i * capacity, (i + 1) * capacity)
            # Empty input ranks can own remote expert rows. Bound the down grid
            # by the largest source, never this rank's local token count.
            bound = min(capacity, max_rows - i * capacity)
            if ctx.base_only:
                outputs.append(
                    experts.launch(
                        x[sl],
                        scores[sl],
                        ids[sl],
                        packed,
                        buffer,
                        owner.config.activation_func_clamp_value,
                        owner.ep_group.size(),
                    )
                )
            else:
                outputs.append(_launch(owner, x[sl], scores[sl], ids[sl], packed, buffer, bound))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    @staticmethod
    def backward(ctx, grad):
        x, scores, ids, *parameters = ctx.saved_tensors
        x = x.detach().requires_grad_(ctx.needs_input_grad[0])
        scores = scores.detach().requires_grad_(ctx.needs_input_grad[1])
        inputs = (x, scores, *parameters)
        active = [p for p in inputs if p.requires_grad]
        # Native hooks expose no pre-SwiGLU L1. Recompute the old BF16 graph,
        # retaining its grouped TE backward, explicitly an approximate adjoint.
        # A disabled forward must stay disabled during backward even when the
        # caller has already exited its temporary disable context.
        replay = disable_lora_adapters(ctx.owner) if ctx.base_only else nullcontext()
        with torch.enable_grad(), replay:
            result = ctx.owner._lora_rows(x, scores, ids)
        values = iter(torch.autograd.grad(result, active, grad, allow_unused=True))
        grads = tuple(next(values) if p.requires_grad else None for p in inputs)
        return grads[0], grads[1], None, None, *grads[2:]


def forward(owner, x, scores, ids):
    gu, *shared = factors(owner)
    parameters = tuple(gu.parameters()) + tuple(p for m in shared for p in m.parameters())
    return _Hooks.apply(x, scores, ids, owner, *parameters)


def base_forward(owner, x, scores, ids):
    """Frozen base uses the same native fused MegaMoE launch as the baseline."""
    return _Hooks.apply(x, scores, ids, owner)
