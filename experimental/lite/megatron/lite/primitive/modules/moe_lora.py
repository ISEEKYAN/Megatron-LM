# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared-outer LoRA for Core grouped experts.

Structure of verl-project/verl#8257 ``MuseSparkExpertsLoRA``: fc1 (gate_up) A and fc2
(down) B are shared by all experts; fc1 B and fc2 A are per expert. The per-expert
factors are Core TE grouped linears (``is_expert=True``): expert-parallel gradient
reduction and EP-sharded checkpoints come from Core, one grouped GEMM per factor.
"""
import math

import torch
from torch import nn

from megatron.core.transformer.module import MegatronModule


class SharedOuterExpertLoRA(MegatronModule):
    def __init__(self, num_local_experts, hidden, intermediate, rank, alpha, *, config, groups):
        from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear

        super().__init__(config=config)
        if config.use_transformer_engine_op_fuser:
            raise ValueError("Expert LoRA hooks the unfused Core grouped linears")
        self.scale, self.rank = alpha / rank, rank
        self.offset = groups.ep.rank() * num_local_experts
        dtype = config.params_dtype
        self.fc1_lora_a = nn.Parameter(torch.empty(rank, hidden, dtype=dtype))
        self.fc2_lora_b = nn.Parameter(torch.zeros(hidden, rank, dtype=dtype))
        grouped = lambda i, o: TEColumnParallelGroupedLinear(
            num_local_experts,
            i,
            o,
            config=config,
            init_method=nn.init.zeros_,
            bias=False,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name=None,
            pg_collection=groups,
        )
        self.fc1_lora_b, self.fc2_lora_a = grouped(rank, 2 * intermediate), grouped(
            intermediate, rank
        )
        with torch.no_grad():  # explicit (whatever perform_initialization): A kaiming, B zero
            for a in (self.fc1_lora_a, *self.per_expert(self.fc2_lora_a)):
                value = torch.empty(a.shape, dtype=torch.float32)
                a.copy_(nn.init.kaiming_uniform_(value, a=math.sqrt(5)))
            for b in self.per_expert(self.fc1_lora_b):
                b.zero_()

    @staticmethod
    def per_expert(grouped):
        return [getattr(grouped, f"weight{i}") for i in range(grouped.num_gemms)]

    def attach(self, experts):
        """Add the deltas to a Core TEGroupedMLP's fc1(x, splits) and fc2(x, splits) outputs."""
        experts.lora = self
        experts.linear_fc1.register_forward_hook(
            lambda m, args, out: (out[0] + self.fc1_delta(*args[:2]), *out[1:])
        )
        experts.linear_fc2.register_forward_hook(
            lambda m, args, out: (out[0] + self.fc2_delta(*args[:2]), *out[1:])
        )

    def fc1_delta(self, x, splits):
        return self.fc1_lora_b(x @ self.fc1_lora_a.t(), splits)[0] * self.scale

    def fc2_delta(self, x, splits):
        return self.fc2_lora_a(x, splits)[0] @ self.fc2_lora_b.t() * self.scale

    def factors(self):
        """Canonical tensors; per-expert names carry the global expert number."""
        state = {"lora_gate_up_A.weight": self.fc1_lora_a, "lora_down_B.weight": self.fc2_lora_b}
        for i, (b, a) in enumerate(
            zip(self.per_expert(self.fc1_lora_b), self.per_expert(self.fc2_lora_a))
        ):
            state[f"{self.offset + i}.lora_gate_up_B.weight"] = b
            state[f"{self.offset + i}.lora_down_A.weight"] = a
        return state


def expand_shared_outer(gate_up_a, gate_up_b, down_a, down_b, scale):
    """Per-expert serving layout (verl#8257 ``export_serving_lora``), ``scale`` folded into B.

    gate_up_a [r, H], gate_up_b [E, 2I, r], down_a [E, r, I], down_b [H, r] ->
    A rows expert-major ([E*r, H], [E*r, I]); B columns ``rank * E + expert``.
    """
    e = gate_up_b.shape[0]
    return {
        "gate_up.lora_A": gate_up_a[None]
        .expand(e, *gate_up_a.shape)
        .reshape(-1, gate_up_a.shape[-1]),
        "gate_up.lora_B": (scale * gate_up_b).permute(1, 2, 0).reshape(gate_up_b.shape[1], -1),
        "down.lora_A": down_a.reshape(-1, down_a.shape[-1]),
        "down.lora_B": (scale * down_b)[:, :, None]
        .expand(*down_b.shape, e)
        .reshape(down_b.shape[0], -1),
    }
