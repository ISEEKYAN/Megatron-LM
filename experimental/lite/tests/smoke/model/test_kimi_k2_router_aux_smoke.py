# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi K2 router auxiliary-loss gradient characterization."""

from dataclasses import replace

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]


def _router_gradients(aux_loss_alpha: float, weight: torch.Tensor, hidden: torch.Tensor):
    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite.model import KimiK2SigmoidTopKRouter
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
    from megatron.lite.primitive.parallel import ParallelState

    base_config = KimiK2Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        intermediate_size=16,
        moe_intermediate_size=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=None,
        topk_group=None,
        first_k_dense_replace=0,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    config = replace(base_config, aux_loss_alpha=aux_loss_alpha)
    router = KimiK2SigmoidTopKRouter(config, ParallelState()).cuda().train()
    router.gate.weight.data.copy_(weight)
    router_input = hidden.detach().clone().requires_grad_(True)

    MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device="cuda"))
    scores, _ = router(router_input)
    scores.square().sum().backward()
    gate_grad = router.gate.weight.grad.detach().clone()
    input_grad = router_input.grad.detach().clone()
    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    return gate_grad, input_grad


def test_kimi_k2_nonzero_aux_loss_changes_router_gradients():
    if not torch.cuda.is_available():
        pytest.skip("Kimi K2 router gradient characterization requires CUDA")

    torch.manual_seed(2026)
    weight = torch.randn(4, 8, device="cuda")
    hidden = torch.randn(64, 8, device="cuda")

    gate_off, input_off = _router_gradients(0.0, weight, hidden)
    gate_on, input_on = _router_gradients(0.001, weight, hidden)

    gate_delta = torch.linalg.vector_norm(gate_on - gate_off)
    input_delta = torch.linalg.vector_norm(input_on - input_off)
    print(
        "Kimi K2 router aux gradient A/B: "
        f"gate_off={torch.linalg.vector_norm(gate_off).item():.8e}, "
        f"gate_on={torch.linalg.vector_norm(gate_on).item():.8e}, "
        f"gate_delta={gate_delta.item():.8e}, "
        f"input_delta={input_delta.item():.8e}"
    )
    assert gate_delta.item() > 0.0
    assert input_delta.item() > 0.0
