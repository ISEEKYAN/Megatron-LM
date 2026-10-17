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
    def __init__(
        self, config, ps, *, gate_temperature=1.0, bias_rate=0.001, bias_group=None
    ):
        super().__init__()
        if gate_temperature <= 0 or bias_rate < 0:
            raise ValueError("Require positive temperature and nonnegative bias rate")
        self.router = SigmoidTopKRouter(config, ps, compute_aux_loss=False)
        self.deployment_math = False
        self.gate_temperature = gate_temperature
        self.bias_rate = bias_rate
        self.bias_group = bias_group
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
        if self.deployment_math and x.is_cuda:
            from megatron.lite.primitive.modules import deployment_math

            logits = (
                deployment_math.bf16_fp32_linear(x, self.router.gate.weight)
                / self.gate_temperature
            )
        else:
            logits = (
                F.linear(x.float(), self.router.gate.weight.float())
                / self.gate_temperature
            )
        if self.deployment_math and x.is_cuda:
            # Native top-k is row-independent; partition by selection bias while
            # preserving original token order and the selected-score FP32 VJP.
            weights = logits.new_empty((len(logits), self.router.topk))
            indices = torch.empty_like(weights, dtype=torch.int64)
            for selected, bias in (
                (~image_mask, self.bias),
                (image_mask, self.bias_vl),
            ):
                if selected.any():
                    routed, slots = self.router.route_logits(
                        logits[selected], expert_bias=bias, deployment_math=True
                    )
                    weights[selected], indices[selected] = routed, slots
        else:
            bias = torch.where(image_mask[:, None], self.bias_vl, self.bias)
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
                        counts,
                        bias,
                        self.bias_rate,
                        (
                            self.bias_group
                            if self.bias_group is not None
                            else dist.group.WORLD
                        ),
                    )
                )


class DeepseekV41MoE(nn.Module):
    """Global expert slots with local owners and primitive token transport."""

    def __init__(
        self,
        router,
        experts,
        shared_experts=None,
        *,
        ps=None,
        use_deepep=False,
        w4a8=False,
        official_expert_contract=False,
        deepep_routed_backward=False,
    ):
        super().__init__()
        self.gate = router
        self.experts = Experts.from_modules(experts)
        self.shared_experts = shared_experts
        self.w4a8 = w4a8
        if official_expert_contract and not w4a8:
            raise ValueError('The official expert contract requires W4A8 experts')
        # Reference inference/model.py: group-32 activations and the routing
        # weight applied before FC2 quantization (rows combine unweighted).
        self.official_expert_contract = official_expert_contract
        from megatron.lite.primitive.modules import dispatcher
        from megatron.lite.primitive.parallel.state import ParallelState

        ps = ps or ParallelState()
        if use_deepep and ps.ep_size > 1 and dispatcher.deep_ep is None:
            raise RuntimeError(
                'V4.1_DEEPEP_UNAVAILABLE: requested DeepEP is not installed'
            )
        if deepep_routed_backward and not (w4a8 and use_deepep):
            raise ValueError('Routed DeepEP backward requires W4A8 experts over DeepEP')
        options = {}
        if w4a8 and use_deepep:
            # W4A8 reduces each token's top-k rows itself (ordered FMA), so
            # DeepEP must return the routed rows unreduced.
            from megatron.lite.primitive.modules.route_preserving_deepep import (
                RoutePreservingDeepEPDispatcher as Dispatcher,
            )

            options['routed_backward'] = deepep_routed_backward
        else:
            Dispatcher = dispatcher.TokenDispatcher
        self.dispatcher = Dispatcher(
            router.router.num_experts,
            router.router.gate.in_features,
            ps,
            use_deepep=use_deepep,
            **options,
        )
        if len(self.experts) != router.router.num_experts:
            raise ValueError("Provide one expert module per routed expert")
        if w4a8:
            if not use_deepep and self.dispatcher.moe_permute_fusion:
                raise ValueError('V4.1 W4A8 requires unfused dispatch')
            start = self.dispatcher.ps.ep_rank * self.dispatcher.num_local_experts
            for index, expert in enumerate(self.experts):
                owned = start <= index < start + self.dispatcher.num_local_experts
                if (expert is not None) != owned:
                    raise ValueError('V4.1 W4A8 requires contiguous EP expert owners')
                if not owned:
                    continue
                if any(
                    weight.shape[-1] % 128
                    for weight in (expert.w1.weight, expert.w2.weight, expert.w3.weight)
                ):
                    raise ValueError(
                        'V4.1 W4A8 expert K dimensions must be divisible by 128'
                    )

    def forward(self, x, *, image_mask=None, load_sink=None):
        flat = x.reshape(-1, x.shape[-1])
        weights, indices, stats = self.gate(flat, image_mask)
        dispatched, counts, scores = self.dispatcher.dispatch(flat, weights, indices)
        if self.w4a8:
            from megatron.lite.primitive.quantization import w4a8_experts

            # Preserve w1/w2/w3 leaf owners and checkpoint/optimizer bindings.
            # Concatenation is row-wise, so group32 W quantization is unchanged.
            # Dispatcher counts and rows are local expert-major. Keep global
            # module slots intact for mirrored load/export and optimizer owners.
            local = [expert for expert in self.experts if expert is not None]
            expert_rows = w4a8_experts.w4a8_expert_mlp(
                dispatched,
                [torch.cat((e.w1.weight, e.w3.weight)) for e in local],
                [e.w2.weight for e in local],
                counts.tolist(),
                swiglu_limit=local[0].swiglu_limit or None,
                row_weights=scores if self.official_expert_contract else None,
            )
            rows, row_token, row_expert = self.dispatcher.combine_unreduced(expert_rows)
            output = w4a8_experts.topk_fma_combine(
                rows,
                row_token,
                row_expert,
                indices,
                (
                    torch.ones_like(weights)
                    if self.official_expert_contract
                    else weights
                ),
            )
        else:
            output = self.dispatcher.combine(self.experts(dispatched, counts, scores))
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat)
        output = output.reshape_as(x)
        if load_sink is not None:
            load_sink.append(stats)
        return output
