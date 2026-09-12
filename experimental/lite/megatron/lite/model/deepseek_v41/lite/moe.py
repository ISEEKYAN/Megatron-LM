# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-rank V4.1 expert computation and explicit modality bias statistics."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModalityLoad:
    """Detached DS4-scope counts [text/image, expert] and token denominators."""

    counts: torch.Tensor
    total_tokens: torch.Tensor


def reduce_modality_load(indices, image_mask, num_experts, group=None):
    # DS4 scope: int64 counts summed over TP; denominator is LOCAL tokens * TP.
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
        return weights, indices, stats

    @torch.no_grad()
    def update_bias(self, stats):
        """Explicit step-time update; forward/recompute never mutates the biases.

        Caller owns accumulation and the DS4 optimizer's skip-step policy.
        No aux objective is attached, matching the DS4 post-training assembly.
        """
        if stats.counts.shape != (2, self.router.num_experts):
            raise ValueError("Expected text/image expert counts")
        for modality, bias in enumerate((self.bias, self.bias_vl)):
            counts = stats.counts[modality].float()
            if counts.sum() > 0:
                bias.add_(torch.sign(counts.mean() - counts), alpha=self.bias_rate)


class SwiGLUExpert(nn.Module):
    """Published expert ordering over injected FP4/FP8/floating projections."""

    def __init__(self, w1, w2, w3, *, swiglu_limit=0.0):
        super().__init__()
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.swiglu_limit = swiglu_limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        activation = F.silu(gate) * up
        if weights is not None:
            activation = activation * weights
        return self.w2(activation.to(x.dtype))


class DeepseekV41MoE(nn.Module):
    """Local dispatch over injected expert providers, including shared experts.

    EP dispatch and optimizer-step publication belong to the distributed layer.
    """

    def __init__(self, router, experts, shared_experts=None):
        super().__init__()
        self.gate = router
        self.experts = nn.ModuleList(experts)
        self.shared_experts = shared_experts
        if len(self.experts) != router.router.num_experts:
            raise ValueError("Provide one expert module per routed expert")

    def forward(self, x, *, image_mask=None, load_stats=None):
        output, stats = self.forward_with_stats(x, image_mask=image_mask)
        if load_stats is not None:
            load_stats.append((self.gate, stats))
        return output

    def forward_with_stats(self, x, *, image_mask=None):
        flat = x.reshape(-1, x.shape[-1])
        weights, indices, stats = self.gate(flat, image_mask)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == index)
            if rows.numel():
                # Weight before down projection/cast, matching official Expert.
                values = expert(flat[rows], weights=weights[rows, slots, None])
                output = output.index_add(0, rows, values)
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat)
        return output.reshape_as(x), stats
