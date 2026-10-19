# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core-owned experts, groups and autograd collectives; R1's slot arithmetic.

Core's dispatcher unpermutes by summing expert slots then adds the shared
expert. MegaMoE starts from the shared expert and adds slots in slot order.
Keep that arithmetic here, reusing Core AG/RS and A2A rather than another
collective implementation. Owner selection is a copy, including signed zero.
"""

import os

import torch

from megatron.core.tensor_parallel.mappings import all_to_all, gather_from_sequence_parallel_region
from megatron.core.transformer.moe.moe_layer import MoELayer

from .experts import combine  # noqa: F401  (re-exported)


def _gather(x, group):
    return gather_from_sequence_parallel_region(x, group=group)


def return_slots(slots, local_ids, local_experts, group):
    received = all_to_all(group, slots)
    parts = received.permute(1, 2, 0, 3)
    owner = local_ids.clamp_min(0).long() // local_experts
    return parts.gather(2, owner[:, :, None, None].expand(-1, -1, 1, slots.shape[-1])).squeeze(2)


class MegaMoE(MoELayer):
    """vLLM BI MegaMoE: ``bf16_mega_gate`` routing, then DeepGEMM ``fp8_fp4_mega_moe``
    over this layer's EP group (see :mod:`.megamoe`). Tokens beyond the symmetric
    buffer run in chunks; every EP rank launches the same number of chunks."""

    max_tokens = int(os.environ.get("DSV41_MEGAMOE_MAX_TOKENS", "8192"))

    def forward(self, hidden_states, input_ids=None, **kwargs):
        from .megamoe import mega_moe
        from .routing import mega_route

        if self.router.tid2eid is not None:
            raise NotImplementedError("DS4.1 MegaMoE hash-routed layers")
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        group, rank = self.ep_group, self.ep_group.rank()
        expected = list(range(rank * self.num_local_experts, (rank + 1) * self.num_local_experts))
        if self.local_expert_indices != expected:
            raise ValueError('MegaMoE requires the Core contiguous global expert ownership map')
        weights, indices = mega_route(
            x,
            self.router.weight,
            self.router.expert_bias.float(),
            self.router.topk,
            self.config.moe_router_topk_scaling_factor,
        )
        fc1 = [
            getattr(self.experts.linear_fc1, f'weight{i}') for i in range(self.num_local_experts)
        ]
        fc2 = [
            getattr(self.experts.linear_fc2, f'weight{i}') for i in range(self.num_local_experts)
        ]
        shared = self.shared_experts
        chunks = torch.tensor([-(-x.shape[0] // self.max_tokens)], device=x.device)
        if group.size() > 1:
            torch.distributed.all_reduce(chunks, torch.distributed.ReduceOp.MAX, group=group)
        outputs = []
        for c in range(int(chunks)):
            rows = slice(c * self.max_tokens, (c + 1) * self.max_tokens)
            outputs.append(
                mega_moe(
                    x[rows],
                    weights[rows],
                    indices[rows],
                    fc1,
                    fc2,
                    shared.linear_fc1.weight,
                    shared.linear_fc2.weight,
                    limit=self.config.activation_func_clamp_value,
                    group=group,
                    max_tokens=self.max_tokens,
                )
            )
        return torch.cat(outputs).reshape_as(hidden_states), None
