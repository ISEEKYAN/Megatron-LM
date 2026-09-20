# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Expert computation through shared dispatch and modality statistics."""

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModalityLoad:
    """Detached replica-scope counts [text/image, expert] and token denominators."""

    counts: torch.Tensor
    total_tokens: torch.Tensor


def reduce_modality_load(indices, image_mask, num_experts, group=None):
    # Replica scope: int64 counts summed over TP; denominator is LOCAL tokens * TP.
    # Both modalities participate in the collective even when locally empty.
    counts = torch.zeros(2, num_experts, device=indices.device, dtype=torch.int64)
    totals = torch.zeros(2, device=indices.device, dtype=torch.int64)
    for modality in range(2):
        selected = image_mask == bool(modality)
        counts[modality] = torch.bincount(
            indices[selected].flatten(), minlength=num_experts
        )
        totals[modality] = selected.sum()
    if group is not None:
        dist.all_reduce(counts, group=group)
        totals = totals * dist.get_world_size(group=group)
    return ModalityLoad(counts, totals)


class ModalityRouter(nn.Module):
    def __init__(self, config, ps, *, gate_temperature=1.0, bias_rate=0.001):
        super().__init__()
        if gate_temperature <= 0 or bias_rate < 0:
            raise ValueError("Require positive temperature and nonnegative bias rate")
        self.router = SigmoidTopKRouter(config, ps, compute_aux_loss=False)
        self.gate_temperature = gate_temperature
        self.bias_rate = bias_rate
        self.world_size = max(
            math.prod(
                getattr(ps, name, 1)
                for name in ("tp_size", "cp_size", "pp_size", "dp_size")
            ),
            math.prod(
                getattr(ps, name, 1)
                for name in ("etp_size", "ep_size", "pp_size", "expert_dp_size")
            ),
            int(os.environ.get("WORLD_SIZE", "1")),
        )
        self.register_buffer("bias", torch.zeros(config.n_routed_experts))
        self.register_buffer("bias_vl", torch.zeros(config.n_routed_experts))

    def _apply(self, fn, recurse=True):
        def preserve_bias(tensor):
            if tensor is self.bias or tensor is self.bias_vl:
                destination = fn(tensor.new_empty(0))
                return tensor.to(device=destination.device, dtype=torch.float32)
            return fn(tensor)

        return super()._apply(preserve_bias, recurse=recurse)

    def forward(self, x, image_mask=None):
        x = x.reshape(-1, self.router.gate.in_features)
        if image_mask is None:
            image_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if image_mask.dtype != torch.bool or image_mask.numel() != x.shape[0]:
            raise ValueError("Expected one boolean image-mask entry per token")
        image_mask = image_mask.reshape(-1)
        bias = torch.where(image_mask[:, None], self.bias_vl, self.bias)
        logits = (
            F.linear(x.float(), self.router.gate.weight.float()) / self.gate_temperature
        )
        weights, indices = self.router.route_logits(logits, expert_bias=bias)
        stats = reduce_modality_load(
            indices, image_mask, self.router.num_experts, self.router._aux_loss_group
        )
        return weights, indices.contiguous(), stats

    @torch.no_grad()
    def update_bias(self, stats):
        """Explicit step-time update; forward/recompute never mutates the biases.

        Caller owns accumulation and the optimizer's skip-step policy.
        No aux objective is attached, matching the post-training assembly.
        """
        if stats.counts.shape != (2, self.router.num_experts):
            raise ValueError("Expected text/image expert counts")
        world_size = max(
            self.world_size, dist.get_world_size() if dist.is_initialized() else 1
        )
        if world_size > 1 and not dist.is_initialized():
            raise RuntimeError("Multi-rank router bias updates require a process group")
        for modality, bias in enumerate((self.bias, self.bias_vl)):
            counts = stats.counts[modality].float()
            if counts.sum() > 0:
                if world_size == 1:
                    bias.add_(torch.sign(counts.mean() - counts) * self.bias_rate)
                    continue
                bias.copy_(
                    get_updated_expert_bias(
                        counts, bias, self.bias_rate, dist.group.WORLD
                    )
                )


class DeepseekV41MoE(nn.Module):
    """Global expert slots with local owners and primitive token transport."""

    def __init__(
        self, router, experts, shared_experts=None, *, ps=None, use_deepep=False
    ):
        super().__init__()
        self.gate = router
        self.experts = Experts.from_modules(experts)
        self.shared_experts = shared_experts
        from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
        from megatron.lite.primitive.parallel.state import ParallelState

        self.dispatcher = TokenDispatcher(
            router.router.num_experts,
            router.router.gate.in_features,
            ps or ParallelState(),
            use_deepep=use_deepep,
        )
        if (
            use_deepep
            and self.dispatcher.ep_size > 1
            and not self.dispatcher.use_deepep
        ):
            raise RuntimeError(
                'V4.1_DEEPEP_UNAVAILABLE: requested DeepEP is not installed'
            )
        if len(self.experts) != router.router.num_experts:
            raise ValueError("Provide one expert module per routed expert")

    def forward(self, x, *, image_mask=None, load_sink=None):
        flat = x.reshape(-1, x.shape[-1])
        weights, indices, stats = self.gate(flat, image_mask)
        dispatched, counts, scores = self.dispatcher.dispatch(flat, weights, indices)
        output = self.dispatcher.combine(self.experts(dispatched, counts, scores))
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat)
        output = output.reshape_as(x)
        if load_sink is not None:
            load_sink.append(stats)
        return output
