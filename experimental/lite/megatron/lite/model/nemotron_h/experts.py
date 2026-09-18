"""ReLU² experts using mlite token dispatch and inference-visible arithmetic."""

import torch
from torch import nn

from .functional import linear, visible_forward


def routed_experts(x, up, down, weights, ids):
    """Evaluate local expert routes; IDs index the locally owned weight tensors."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    def visible(x, up, down, weights):
        if x.shape[0] == 0:
            return x.new_empty((0, down.shape[1]))
        return fused_experts(
            x, up, down, weights, ids, activation=MoEActivation.RELU2_NO_MUL
        )

    def native(x, up, down, weights):
        slots = x.new_zeros(x.shape[0] * ids.shape[1], down.shape[1])
        for expert in range(up.shape[0]):
            token, slot = torch.where(ids == expert)
            hidden = torch.nn.functional.linear(x[token], up[expert])
            hidden = torch.nn.functional.relu(hidden).square()
            hidden = torch.nn.functional.linear(hidden, down[expert])
            hidden = (hidden * weights[token, slot, None]).to(x.dtype)
            slots = slots.index_copy(0, token * ids.shape[1] + slot, hidden)
        return slots.view(x.shape[0], ids.shape[1], down.shape[1]).sum(1)

    return visible_forward(visible, native, x, up, down, weights)


class RoutedExperts(nn.Module):
    """TP1/ETP1 owned experts with native all-to-all and ordered top-k combine.

    Each slot is dispatched separately to avoid expert-order reduction changing
    inference rounding. This correctness-first transport is not a performance
    claim; replacing it requires the same route-order and backward gates.
    """

    def __init__(self, config, ps, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

        if ps.tp_size != 1 or ps.etp_size != 1:
            raise ValueError("Nemotron experts currently require TP1/ETP1")
        if config.n_routed_experts % ps.ep_size:
            raise ValueError("Expert count must divide EP size")
        if config.mlp_bias or config.mlp_hidden_act != "relu2":
            raise ValueError("Expected bias-free ReLU2 experts")
        local = config.n_routed_experts // ps.ep_size
        self.up_proj = nn.Parameter(
            torch.empty(
                local,
                config.moe_intermediate_size,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                local,
                config.hidden_size,
                config.moe_intermediate_size,
                device=device,
                dtype=dtype,
            )
        )
        self.dispatcher = TokenDispatcher(
            config.n_routed_experts,
            config.hidden_size,
            ps,
            use_deepep=False,
            moe_permute_fusion=False,
        )

    def forward(self, x, ids, weights):
        from vllm import _custom_ops as ops

        if ids.shape != weights.shape or ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("Expected matching nonempty [tokens, topk] routes")
        slots = []
        for slot in range(ids.shape[1]):
            rows, counts, probs = self.dispatcher.dispatch(
                x, weights[:, slot : slot + 1], ids[:, slot : slot + 1].long()
            )
            local_ids = torch.repeat_interleave(
                torch.arange(self.up_proj.shape[0], device=x.device),
                counts.to(device=x.device),
                output_size=rows.shape[0],
            )[:, None].int()
            output = routed_experts(
                rows,
                self.up_proj,
                self.down_proj,
                probs[:, None].contiguous(),
                local_ids,
            )
            slots.append(self.dispatcher.combine(output))
        routes = torch.stack(slots, dim=1)

        def combine(routes):
            output = torch.empty_like(x)
            ops.moe_sum(routes, output)
            return output

        return visible_forward(combine, lambda routes: routes.sum(1), routes)


class Router(nn.Module):
    def __init__(self, config, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, device=device, dtype=dtype
            )
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.n_routed_experts, device=device, dtype=torch.float32),
        )

    def forward(self, x):
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            grouped_topk,
        )

        config = self.config
        logits = visible_forward(
            lambda x, weight: torch.mm(x, weight.T, out_dtype=torch.float32),
            lambda x, weight: torch.nn.functional.linear(x.float(), weight.float()),
            x,
            self.weight,
        )
        with torch.no_grad():
            weights, ids = grouped_topk(
                x,
                logits,
                config.num_experts_per_tok,
                config.norm_topk_prob,
                config.n_group,
                config.topk_group,
                "sigmoid",
                1.0,
                self.e_score_correction_bias.float(),
            )

        def native(logits):
            selected = logits.sigmoid().gather(1, ids.long())
            if config.norm_topk_prob:
                selected = selected / selected.sum(-1, keepdim=True)
            return selected

        return ids, visible_forward(lambda logits: weights, native, logits)


class SharedExperts(nn.Module):
    def __init__(self, config, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.moe_shared_expert_intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            config.moe_shared_expert_intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = torch.nn.functional.relu(linear(x, self.up_proj.weight)).square()
        return linear(x, self.down_proj.weight)


class MoE(nn.Module):
    def __init__(self, config, ps, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        if config.n_shared_experts != 1:
            raise ValueError("Nemotron MoE requires the single shared expert contract")
        self.gate = Router(config, device=device, dtype=dtype)
        self.experts = RoutedExperts(config, ps, device=device, dtype=dtype)
        self.shared_experts = SharedExperts(config, device=device, dtype=dtype)
        scale = config.routed_scaling_factor

        def combine(shared, routed):
            return shared + routed * scale

        self._native_combine = combine
        self._visible_combine = torch.compile(combine, fullgraph=True, dynamic=True)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        ids, weights = self.gate(x)
        routed = self.experts(x, ids, weights)
        shared = self.shared_experts(x)
        return visible_forward(
            self._visible_combine, self._native_combine, shared, routed
        ).view(shape)
