# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MegaMoE on the native Core MoE parameter owners (router, grouped experts, shared)."""

import os

from megatron.core.transformer.moe.moe_layer import MoELayer

from .experts import combine  # noqa: F401  (re-exported)


class MegaMoE(MoELayer):
    """vLLM BI MegaMoE: ``bf16_mega_gate`` routing, then DeepGEMM ``fp8_fp4_mega_moe``
    over this layer's EP group (see :mod:`.megamoe`)."""

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
        y = mega_moe(
            x,
            weights,
            indices,
            fc1,
            fc2,
            shared.linear_fc1.weight,
            shared.linear_fc2.weight,
            limit=self.config.activation_func_clamp_value,
            group=group,
            max_tokens=self.max_tokens,
        )
        return y.reshape_as(hidden_states), None
