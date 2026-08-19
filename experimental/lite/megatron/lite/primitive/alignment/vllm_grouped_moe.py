"""vLLM grouped-DeepGEMM forward with a BF16-master training backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.lite.primitive.modules.experts import swiglu_with_probs

_BACKWARD_CHUNK_ROWS = 1024


def _vllm_grouped_forward(
    hidden_states: torch.Tensor,
    counts: tuple[int, ...],
    swiglu_limit: float,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, hidden_states.shape[1]))
    from megatron.lite.primitive.alignment.vllm_batched_prepare import (
        _quantize_batched_input,
    )
    from megatron.lite.primitive.kernels.vllm_ds4 import (
        GroupedDeepGemmExpertsAdapter,
        GroupedMoEKernelBuilderAdapter,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        ExpertTokensMetadata,
    )

    num_experts = len(counts)
    max_count = max(counts)
    batched = hidden_states.new_zeros(
        (num_experts, max_count, hidden_states.shape[1])
    )
    token_offset = 0
    for expert, count in enumerate(counts):
        if count:
            batched[expert, :count].copy_(
                hidden_states.narrow(0, token_offset, count)
            )
            token_offset += count
    if token_offset != hidden_states.shape[0]:
        raise RuntimeError(
            "grouped MoE expert counts do not cover all expert-major rows: "
            f"{token_offset} != {hidden_states.shape[0]}"
        )

    adapter = GroupedDeepGemmExpertsAdapter()
    packed = adapter.pack(w13, w2)
    builder = GroupedMoEKernelBuilderAdapter(
        None,
        device=hidden_states.device,
        num_experts=num_experts,
        num_local_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden_states.shape[1],
        intermediate_size=w2[0].shape[1],
        max_tokens_per_rank=max_count,
        num_dispatchers=1,
        max_tokens_per_dispatcher_expert=max_count,
    )
    kernel = builder(packed, dispatcher=object())
    experts = kernel.fused_experts
    tokens_per_expert = torch.tensor(
        counts, dtype=torch.int32, device=hidden_states.device
    )
    quantized, scales = _quantize_batched_input(
        batched,
        experts.quant_config,
        tokens_per_expert,
    )
    workspace_dtype = experts.workspace_dtype(torch.bfloat16)
    workspace13 = torch.empty(
        (num_experts, max_count, w13[0].shape[0]),
        dtype=workspace_dtype,
        device=hidden_states.device,
    )
    workspace2 = torch.empty(
        (num_experts, max_count, w2[0].shape[1]),
        dtype=workspace_dtype,
        device=hidden_states.device,
    )
    output = hidden_states.new_empty(
        (num_experts, max_count, hidden_states.shape[1])
    )
    metadata = ExpertTokensMetadata(
        expert_num_tokens=tokens_per_expert,
        expert_num_tokens_cpu=None,
    )
    dummy_weights = torch.ones(
        (1, 1), dtype=torch.float32, device=hidden_states.device
    )
    dummy_ids = torch.zeros(
        (1, 1), dtype=torch.int64, device=hidden_states.device
    )
    experts.apply(
        output=output,
        hidden_states=quantized,
        w1=packed.w13,
        w2=packed.w2,
        topk_weights=dummy_weights,
        topk_ids=dummy_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=torch.arange(
            num_experts, dtype=torch.int32, device=hidden_states.device
        ),
        a1q_scale=scales,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=metadata,
        apply_router_weight_on_input=False,
    )
    compact = [
        output[expert, :count]
        for expert, count in enumerate(counts)
        if count
    ]
    if compact:
        return torch.cat(compact, dim=0)
    return hidden_states.new_empty(
        (0, hidden_states.shape[1])
    )


class VLLMGroupedMoEWithBF16Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        swiglu_limit: float,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = tokens_per_expert.numel()
        if len(weights) != 2 * num_experts:
            raise ValueError("grouped MoE weight count does not match local experts")
        counts = tuple(int(value) for value in tokens_per_expert.detach().cpu().tolist())
        if sum(counts) != hidden_states.shape[0]:
            raise ValueError("tokens_per_expert does not match expert-major rows")
        w13 = tuple(weights[:num_experts])
        w2 = tuple(weights[num_experts:])
        output = _vllm_grouped_forward(hidden_states, counts, float(swiglu_limit), w13, w2)
        ctx.counts = counts
        ctx.swiglu_limit = float(swiglu_limit)
        ctx.save_for_backward(hidden_states, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, *weights = ctx.saved_tensors
        num_experts = len(ctx.counts)
        w13 = weights[:num_experts]
        w2 = weights[num_experts:]
        grad_hidden = torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        grad_w13 = [torch.zeros_like(weight) if ctx.needs_input_grad[4 + i] else None for i, weight in enumerate(w13)]
        grad_w2 = [torch.zeros_like(weight) if ctx.needs_input_grad[4 + num_experts + i] else None for i, weight in enumerate(w2)]
        offset = 0
        for expert, count in enumerate(ctx.counts):
            for start in range(0, count, _BACKWARD_CHUNK_ROWS):
                end = min(start + _BACKWARD_CHUNK_ROWS, count)
                row_slice = slice(offset + start, offset + end)
                with torch.enable_grad():
                    hidden = hidden_states[row_slice].detach().requires_grad_(True)
                    fc1 = w13[expert].detach().requires_grad_(True)
                    fc2 = w2[expert].detach().requires_grad_(True)
                    gate_up = F.linear(hidden, fc1)
                    activated = swiglu_with_probs(gate_up, None, ctx.swiglu_limit)
                    recomputed = F.linear(activated, fc2)
                    grad_h, grad_fc1, grad_fc2 = torch.autograd.grad(
                        recomputed,
                        (hidden, fc1, fc2),
                        grad_output[row_slice],
                    )
                if grad_hidden is not None:
                    grad_hidden[row_slice].copy_(grad_h)
                if grad_w13[expert] is not None:
                    grad_w13[expert].add_(grad_fc1)
                if grad_w2[expert] is not None:
                    grad_w2[expert].add_(grad_fc2)
            offset += count
        return grad_hidden, None, None, None, *grad_w13, *grad_w2
