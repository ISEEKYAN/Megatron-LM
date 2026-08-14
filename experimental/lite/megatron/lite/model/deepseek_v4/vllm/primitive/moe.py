"""Whole-DeepEP visible owner with deterministic BF16 expert adjoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ._contract import check_parameter_versions, own_visible_tensor, parameter_versions


@dataclass(frozen=True)
class TrainingRouteTape:
    """Read-only discrete routing state captured from one visible forward."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    local_token_count: int
    global_expert_start: int


def _gather_variable(value: torch.Tensor, group):
    if not dist.is_available() or not dist.is_initialized():
        return (value,), (value.shape[0],), 0
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    count = torch.tensor([value.shape[0]], device=value.device, dtype=torch.int64)
    count_tensors = [torch.empty_like(count) for _ in range(world)]
    dist.all_gather(count_tensors, count, group=group)
    counts = tuple(int(item.item()) for item in count_tensors)
    maximum = max(counts)
    padded = torch.zeros((maximum, *value.shape[1:]), dtype=value.dtype, device=value.device)
    padded[: value.shape[0]].copy_(value)
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    return tuple(item[:count] for item, count in zip(gathered, counts, strict=True)), counts, rank


class _VLLMDeepEPMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, visible_op, group, global_start, num_w13, hidden, probs, ids, *weights):
        output = own_visible_tensor(visible_op(hidden, probs, ids, *weights))
        ctx.group = group
        ctx.global_start = global_start
        ctx.num_w13 = num_w13
        ctx.hidden = hidden
        ctx.probs = probs
        ctx.ids = ids
        ctx.weights = weights
        ctx.versions = parameter_versions(weights)
        ctx.tape = TrainingRouteTape(ids, probs, hidden.shape[0], global_start)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output):
        check_parameter_versions(ctx.weights, ctx.versions)
        hidden_parts, counts, rank = _gather_variable(ctx.hidden.detach(), ctx.group)
        prob_parts, _, _ = _gather_variable(ctx.probs.detach(), ctx.group)
        id_parts, _, _ = _gather_variable(ctx.ids.detach(), ctx.group)
        grad_parts, _, _ = _gather_variable(grad_output.detach(), ctx.group)
        with torch.enable_grad():
            hidden_all = torch.cat(hidden_parts).requires_grad_(True)
            probs_all = torch.cat(prob_parts).float().requires_grad_(True)
            ids_all = torch.cat(id_parts).long()
            grad_all = torch.cat(grad_parts).float()

            weight_copies = tuple(
                weight.detach().requires_grad_(True) for weight in ctx.weights
            )
            w13 = weight_copies[: ctx.num_w13]
            w2 = weight_copies[ctx.num_w13 :]
            objective = hidden_all.float().sum() * 0 + probs_all.sum() * 0
            objective = objective + sum(
                weight.float().sum() * 0 for weight in weight_copies
            )
            for local_expert, (fc1, fc2) in enumerate(zip(w13, w2, strict=True)):
                global_expert = ctx.global_start + local_expert
                token_index, slot_index = torch.where(ids_all == global_expert)
                if token_index.numel() == 0:
                    continue
                selected = hidden_all.index_select(0, token_index).float()
                gate_up = F.linear(selected, fc1.float())
                gate, up = gate_up.chunk(2, dim=-1)
                expert_output = F.linear(F.silu(gate) * up, fc2.float())
                route_prob = probs_all[token_index, slot_index]
                objective = objective + (
                    expert_output
                    * grad_all.index_select(0, token_index)
                    * route_prob.unsqueeze(-1)
                ).sum()

            grads = torch.autograd.grad(
                objective, (hidden_all, probs_all, *weight_copies), allow_unused=False
            )
        grad_hidden_all, grad_probs_all, *grad_weights = grads
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_hidden_all, group=ctx.group)
            dist.all_reduce(grad_probs_all, group=ctx.group)
        offset = sum(counts[:rank])
        local_count = counts[rank]
        grad_hidden = grad_hidden_all[offset : offset + local_count].to(ctx.hidden.dtype)
        grad_probs = grad_probs_all[offset : offset + local_count].to(ctx.probs.dtype)
        grad_weights = tuple(
            grad.to(weight.dtype) for grad, weight in zip(grad_weights, ctx.weights, strict=True)
        )
        return None, None, None, None, grad_hidden, grad_probs, None, *grad_weights


def deep_ep_moe(
    visible_op,
    hidden,
    topk_weights,
    topk_ids,
    w13,
    w2,
    *,
    group=None,
    global_expert_start: int,
):
    w13, w2 = tuple(w13), tuple(w2)
    return _VLLMDeepEPMoEFunction.apply(
        visible_op,
        group,
        global_expert_start,
        len(w13),
        hidden,
        topk_weights,
        topk_ids,
        *w13,
        *w2,
    )


__all__ = ["TrainingRouteTape", "deep_ep_moe"]
