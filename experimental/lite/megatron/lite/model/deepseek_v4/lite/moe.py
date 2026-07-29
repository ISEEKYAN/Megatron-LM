# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.mlp import SwiGLUMLP
from megatron.lite.primitive.modules.moe_ep_chunk_overlap import EPChunkOverlapOperator
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.parallel.state import ParallelState


class DeepseekV4MoE(nn.Module):
    """Model-specific assembly over shared router, Experts, dispatcher, and shared MLP.

    Allowlist reason: this owns DS4 hash routing wiring, while expert compute stays shared.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        *,
        layer_idx: int,
        use_deepep: bool = False,
        num_chunks_ep_a2a_overlap: int = 1,
    ):
        super().__init__()
        self.gate = SigmoidTopKRouter(
            config,
            ps,
            compute_aux_loss=False,
            moe_router_fusion=num_chunks_ep_a2a_overlap > 1,
        )
        is_hash_layer = layer_idx < config.num_hash_layers
        if is_hash_layer:
            self.gate.register_buffer(
                "tid2eid",
                torch.zeros(
                    config.vocab_size, config.num_experts_per_tok, dtype=torch.int64
                ),
                persistent=True,
            )
        else:
            self.gate._non_persistent_buffers_set.discard("expert_bias")
        self.experts = Experts(
            config,
            ps,
            delay_wgrad_compute=num_chunks_ep_a2a_overlap > 1,
        )
        self.hidden_size = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.is_hash_layer = is_hash_layer
        self.dispatcher = TokenDispatcher(
            config.n_routed_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )
        self.ep_chunk_dispatchers = ()
        self.ep_chunk_overlap = None
        if num_chunks_ep_a2a_overlap > 1:
            layer_slot = int(layer_idx) % 8
            self.ep_chunk_dispatchers = tuple(
                TokenDispatcher(
                    config.n_routed_experts,
                    config.hidden_size,
                    ps,
                    use_deepep=use_deepep,
                    buffer_slot=("ep_chunk_overlap", "forward", layer_slot, idx),
                )
                for idx in range(num_chunks_ep_a2a_overlap)
            )
            self.ep_chunk_overlap = EPChunkOverlapOperator(
                router=self.gate,
                experts=self.experts,
                dispatcher=self.dispatcher,
                forward_dispatchers=self.ep_chunk_dispatchers,
                router_forward=self._route_for_overlap,
            )
        shared_intermediate = config.n_shared_experts * config.moe_intermediate_size
        self.shared_experts = (
            SwiGLUMLP(
                config.hidden_size,
                shared_intermediate,
                swiglu_limit=config.swiglu_limit,
            )
            if config.n_shared_experts > 0
            else None
        )

    def _route_for_overlap(
        self, router: nn.Module, x: torch.Tensor, input_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del router
        if self.is_hash_layer and input_ids is not None:
            return self._hash_route(x, input_ids)
        return self.gate(x)

    def _hash_route(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate.gate(x).view(-1, self.gate.num_experts)
        if self.gate.score_function == "sqrtsoftplus":
            scores = F.softplus(logits.float()).sqrt()
        else:
            scores = logits.float().sigmoid()
        indices = self.gate.tid2eid[input_ids.reshape(-1).to(torch.int64)]
        weights = scores.gather(1, indices)
        # R3's rollout tensor contains one column for every DS4 MoE layer,
        # including the input-deterministic hash layers.  Consume/replay those
        # columns here so later learned-router layers retain the same global
        # layer index as vLLM.  Replay changes only indices; normalization and
        # route scaling below still use this actor's live scores.
        if self.gate.router_replay is not None:
            weights, indices = self.gate.router_replay.apply(scores, weights, indices)
        if self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (weights * self.route_scale).to(dtype=x.dtype), indices

    def forward(self, x: torch.Tensor, *, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        if self.ep_chunk_overlap is not None:
            out = self.ep_chunk_overlap(x_flat, routing_input=input_ids)
        else:
            scores, indices = self._route_for_overlap(self.gate, x_flat, input_ids)
            dispatched, tpe, probs = self.dispatcher.dispatch(
                x_flat, scores, indices
            )
            self.dispatcher.wait_dispatch_event()
            expert_out = self.experts(
                dispatched,
                tpe,
                probs,
                tokens_per_expert_list=getattr(
                    self.dispatcher, "_local_tpe_list", None
                ),
            )
            out = self.dispatcher.combine(expert_out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x_flat)
        return out.view(shape)
