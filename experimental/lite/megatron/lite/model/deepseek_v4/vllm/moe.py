"""DS4 MoE with vLLM-visible compute and training DeepEP communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.primitive import (
    block_fp8_linear,
    deep_ep_moe,
    fixed_route_vjp,
    gate_linear as training_gate_linear,
)
from megatron.lite.primitive.kernels.vllm_ds4 import (
    DS4TopKAdapter,
    GateLinearAdapter,
    GroupedDeepGemmExpertsAdapter,
    HashRouteAdapter,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    DeploymentBlockFP8Adapter,
)


@dataclass
class MoEKernelMetadata:
    """Caller-owned vLLM router and exact LL grouped-MoE lifecycle."""

    gate_linear: Callable[[torch.Tensor], Any] | None
    build_grouped_moe: Callable[[Any], Any] | None = None


class _RouterState(nn.Module):
    def __init__(self, config: DeepseekV4Config, *, hash_layer: bool):
        super().__init__()
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            dtype=torch.bfloat16,
        )
        if hash_layer:
            self.register_buffer(
                "tid2eid",
                torch.zeros(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        else:
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.n_routed_experts, dtype=torch.float32),
            )


class _SharedExpertsState(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        width = config.n_shared_experts * config.moe_intermediate_size
        self.gate_up = nn.Linear(
            config.hidden_size, 2 * width, bias=False, dtype=torch.bfloat16
        )
        self.down = nn.Linear(
            width, config.hidden_size, bias=False, dtype=torch.bfloat16
        )
        self.gate_up_fp8 = DeploymentBlockFP8Adapter(cache_weight=True)
        self.down_fp8 = DeploymentBlockFP8Adapter(cache_weight=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = block_fp8_linear(
            self.gate_up_fp8, hidden_states, self.gate_up.weight
        )
        gate, up = gate_up.chunk(2, dim=-1)
        return block_fp8_linear(
            self.down_fp8,
            F.silu(gate) * up,
            self.down.weight,
        )


class _LocalExpertsState(nn.Module):
    def __init__(self, config: DeepseekV4Config, ps: ParallelState):
        super().__init__()
        self.ps = ps
        if config.n_routed_experts % ps.ep_size:
            raise ValueError("routed experts must divide evenly across EP ranks")
        self.ps = ps
        self.global_start = ps.ep_rank * (config.n_routed_experts // ps.ep_size)
        self.local_count = config.n_routed_experts // ps.ep_size
        self.w13 = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        2 * config.moe_intermediate_size,
                        config.hidden_size,
                        dtype=torch.bfloat16,
                    )
                )
                for _ in range(self.local_count)
            ]
        )
        self.w2 = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        config.hidden_size,
                        config.moe_intermediate_size,
                        dtype=torch.bfloat16,
                    )
                )
                for _ in range(self.local_count)
            ]
        )
        expert_map = torch.full(
            (config.n_routed_experts,), -1, dtype=torch.int32
        )
        expert_map[self.global_start : self.global_start + self.local_count] = (
            torch.arange(self.local_count, dtype=torch.int32)
        )
        self.register_buffer("expert_map", expert_map, persistent=False)
        self.grouped_adapter = GroupedDeepGemmExpertsAdapter()

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        build_kernel: Callable[[Any], Any],
        global_num_experts: int,
    ) -> torch.Tensor:
        def visible(hidden, probs, ids, *weights):
            split = len(weights) // 2
            return self.grouped_adapter(
                hidden,
                weights[:split],
                weights[split:],
                probs,
                ids,
                build_kernel=build_kernel,
                global_num_experts=global_num_experts,
                expert_map=self.expert_map,
            )

        return deep_ep_moe(
            visible,
            hidden_states,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            group=self.ps.ep_group,
            global_expert_start=self.global_start,
        )


class DeepseekV4MoE(nn.Module):

    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        *,
        layer_idx: int,
        selected_stages: frozenset[str] = frozenset(),
        use_deepep: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps or ParallelState()
        self.selected_stages = selected_stages
        self.use_deepep = use_deepep
        self.is_hash_layer = layer_idx < config.num_hash_layers
        self.gate = _RouterState(config, hash_layer=self.is_hash_layer)
        self.experts = _LocalExpertsState(config, self.ps)
        self.shared_experts = (
            _SharedExpertsState(config) if config.n_shared_experts > 0 else None
        )
        self.gate_adapter = GateLinearAdapter()
        self.hash_route_adapter = HashRouteAdapter()
        self.learned_route_adapter = DS4TopKAdapter()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        metadata: MoEKernelMetadata | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("MoE requires flat [tokens, hidden]")
        if "router_moe" not in self.selected_stages:
            raise NotImplementedError("stage 'router_moe' was not executed")
        if self.is_hash_layer and input_ids is None:
            raise NotImplementedError("hash MoE requires explicit input_ids")
        if metadata is None or metadata.gate_linear is None:
            raise NotImplementedError("MoE requires an explicit vLLM GateLinear")
        gate_linear = metadata.gate_linear
        if isinstance(gate_linear, nn.Module) and hasattr(gate_linear, "weight"):
            # Bind the caller-constructed vLLM GateLinear to this model's BF16
            # master rather than retaining a second persistent parameter.
            gate_linear._parameters["weight"] = self.gate.gate.weight
        logits = training_gate_linear(
            lambda value: self.gate_adapter(gate_linear, value),
            hidden_states,
            self.gate.gate.weight,
        ).float().contiguous()
        if self.is_hash_layer:
            token_ids = input_ids.reshape(-1).to(dtype=torch.int32)
            tid2eid = self.gate.tid2eid.to(dtype=torch.int32).contiguous()
            topk_weights, topk_ids = fixed_route_vjp(
                lambda value: self.hash_route_adapter(
                    value,
                    token_ids,
                    tid2eid,
                    topk=self.config.num_experts_per_tok,
                    renormalize=self.config.norm_topk_prob,
                    routed_scaling_factor=self.config.routed_scaling_factor,
                    indices_dtype=torch.int64,
                ),
                logits,
                renormalize=self.config.norm_topk_prob,
                route_scale=self.config.routed_scaling_factor,
            )
        else:
            correction_bias = self.gate.expert_bias.float().contiguous()
            topk_weights, topk_ids = fixed_route_vjp(
                lambda value: self.learned_route_adapter(
                    value,
                    correction_bias,
                    indices_dtype=torch.int64,
                    routed_scaling_factor=self.config.routed_scaling_factor,
                ),
                logits,
                renormalize=True,
                route_scale=self.config.routed_scaling_factor,
            )

        if "deepep" not in self.selected_stages:
            raise NotImplementedError("stage 'deepep' was not executed")
        if not self.use_deepep or metadata.build_grouped_moe is None:
            raise NotImplementedError(
                "MoE requires an official DeepEPLLPrepareAndFinalize + "
                "BatchedDeepGemmExperts kernel builder"
            )
        output = self.experts(
            hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(dtype=torch.int64),
            build_kernel=metadata.build_grouped_moe,
            global_num_experts=self.config.n_routed_experts,
        )
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output


__all__ = ["DeepseekV4MoE", "MoEKernelMetadata"]
