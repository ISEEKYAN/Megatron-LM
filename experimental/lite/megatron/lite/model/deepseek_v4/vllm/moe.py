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
    fixed_route_vjp,
    gate_linear as training_gate_linear,
)
from megatron.lite.primitive.alignment.vllm_grouped_moe import (
    VLLMGroupedMoEWithBF16Backward,
)
from megatron.lite.primitive.kernels.vllm_ds4 import (
    DS4TopKAdapter,
    GateLinearAdapter,
    HashRouteAdapter,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.router_replay import RouterReplay
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    DeploymentBlockFP8Adapter,
)


def _kernel_topk_weights(weights: torch.Tensor) -> torch.Tensor:
    """Restore the FP32 grouped-kernel boundary after FSDP input casting."""
    return weights if weights.dtype == torch.float32 else weights.float()


@dataclass
class MoEKernelMetadata:
    """Caller-owned vLLM router metadata."""

    gate_linear: Callable[[torch.Tensor], Any] | None


class _RouterState(nn.Module):
    def __init__(self, config: DeepseekV4Config, *, hash_layer: bool):
        super().__init__()
        self.router_replay: RouterReplay | None = None
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
        # The rollout BatchedDeepGemm path currently exposes plain SiLU*up for
        # routed experts.  The compute-only VJP must differentiate that same
        # visible function, not the clamped Lite expert fallback.
        self.visible_swiglu_limit = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None,
        *,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        if tokens_per_expert_list is not None:
            tokens_per_expert = torch.tensor(
                tokens_per_expert_list,
                device=tokens_per_expert.device,
                dtype=tokens_per_expert.dtype,
            )
        if permuted_probs is None:
            permuted_probs = hidden_states.new_zeros(
                (hidden_states.shape[0],), dtype=torch.float32
            )
        return VLLMGroupedMoEWithBF16Backward.apply(
            hidden_states,
            tokens_per_expert,
            permuted_probs,
            self.visible_swiglu_limit,
            *self.w13,
            *self.w2,
        )


class DeepseekV4MoE(nn.Module):

    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        *,
        layer_idx: int,
        selected_stages: frozenset[str] = frozenset(),
        moe_token_dispatcher_type: str = "deepep",
    ):
        super().__init__()
        self.config = config
        self.ps = ps or ParallelState()
        self.selected_stages = selected_stages
        self.is_hash_layer = layer_idx < config.num_hash_layers
        self.gate = _RouterState(config, hash_layer=self.is_hash_layer)
        self.experts = _LocalExpertsState(config, self.ps)
        self.dispatcher = TokenDispatcher(
            config.n_routed_experts,
            config.hidden_size,
            self.ps,
            deepep_align_to_low_latency=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
        )
        self.shared_experts = (
            _SharedExpertsState(config) if config.n_shared_experts > 0 else None
        )
        self.gate_adapter = GateLinearAdapter()
        self.hash_route_adapter = HashRouteAdapter()
        self.learned_route_adapter = DS4TopKAdapter()

    def forward_with_fixed_routes(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the production training MoE body with caller-owned routes."""

        if hidden_states.ndim != 2:
            raise ValueError("MoE requires flat [tokens, hidden]")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("fixed-route weights and IDs must have identical shapes")
        if topk_ids.ndim != 2 or topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("fixed routes must describe every input token")
        if topk_ids.shape[1] != self.config.num_experts_per_tok:
            raise ValueError("fixed routes use the wrong top-k width")

        dispatched, tokens_per_expert, permuted_probs = self.dispatcher.dispatch(
            hidden_states,
            _kernel_topk_weights(topk_weights),
            topk_ids.to(dtype=torch.int64),
        )
        self.dispatcher.wait_dispatch_event()
        output = self.experts(
            dispatched,
            tokens_per_expert,
            permuted_probs,
            tokens_per_expert_list=getattr(
                self.dispatcher, "_local_tpe_list", None
            ),
        )
        return self.dispatcher.combine(output)

    def _replay_route(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay IDs while retaining weights from this actor's live router."""

        replay = self.gate.router_replay
        if replay is None:
            return weights, ids
        selected = replay.select_indices(ids)
        if selected is ids:
            return weights, ids
        dense = torch.sqrt(F.softplus(logits.float()))
        weights = dense.gather(-1, selected.long())
        if self.config.norm_topk_prob and selected.size(-1) > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.config.routed_scaling_factor
        return weights.to(dtype=logits.dtype), selected

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
            def hash_route(value):
                weights, ids = self.hash_route_adapter(
                    value,
                    token_ids,
                    tid2eid,
                    topk=self.config.num_experts_per_tok,
                    renormalize=self.config.norm_topk_prob,
                    routed_scaling_factor=self.config.routed_scaling_factor,
                    indices_dtype=torch.int64,
                )
                return self._replay_route(value, weights, ids)

            topk_weights, topk_ids = fixed_route_vjp(
                hash_route,
                logits,
                renormalize=self.config.norm_topk_prob,
                route_scale=self.config.routed_scaling_factor,
            )
        else:
            correction_bias = self.gate.expert_bias.float().contiguous()

            def learned_route(value):
                weights, ids = self.learned_route_adapter(
                    value,
                    correction_bias,
                    indices_dtype=torch.int64,
                    routed_scaling_factor=self.config.routed_scaling_factor,
                )
                return self._replay_route(value, weights, ids)

            topk_weights, topk_ids = fixed_route_vjp(
                learned_route,
                logits,
                renormalize=True,
                route_scale=self.config.routed_scaling_factor,
            )

        if "deepep" not in self.selected_stages:
            raise NotImplementedError("stage 'deepep' was not executed")
        output = self.forward_with_fixed_routes(
            hidden_states,
            topk_weights,
            topk_ids,
        )
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output


__all__ = ["DeepseekV4MoE", "MoEKernelMetadata"]
