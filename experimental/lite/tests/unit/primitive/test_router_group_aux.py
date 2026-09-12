# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The balancing objective uses unbiased, unrestricted expert affinity."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("groups,bias", [(1, False), (2, False), (2, True)])
def test_sigmoid_aux_gradient_matches_unrestricted_reference(
    transformer_engine_import_stub, monkeypatch, groups, bias, fused
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.utils import moe as moe_utils

    # Exercise the real fused dispatch/aux branches on CPU, replacing only
    # their TE entrypoints. This is wiring coverage, not CUDA kernel validation.
    fused_calls = []

    def fused_topk(**kwargs):
        fused_calls.append("dispatch")
        return moe_utils.topk_routing_with_score_function(**kwargs, fused=False)

    def fused_aux(**kwargs):
        fused_calls.append("aux")
        return moe_utils.compute_routing_scores_for_aux_loss(**kwargs, fused=False)

    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", fused_topk)
    monkeypatch.setattr(moe_utils, "fused_compute_score_for_moe_aux_loss", fused_aux)
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    config = SimpleNamespace(
        hidden_size=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        aux_loss_alpha=0.1,
        routed_scaling_factor=1.0,
        n_group=groups,
        topk_group=1,
    )
    router = SigmoidTopKRouter(
        config, SimpleNamespace(tp_size=1), moe_router_fusion=fused
    )
    with torch.no_grad():
        router.gate.weight.copy_(torch.eye(4))
        if bias:
            router.expert_bias.copy_(torch.tensor([0.0, 0.0, 2.0, 2.0]))
    x = torch.tensor([[3.0, -2.0, 2.0, 1.0], [4.0, -3.0, 1.5, 0.5]], requires_grad=True)
    scores, indices = router(x)
    assert fused_calls == (["dispatch", "aux"] if fused else [])
    # Zero task gradient isolates the actual autoscaler-attached aux gradient.
    (scores.sum() * 0).backward()

    ref_x = x.detach().clone().requires_grad_()
    ref_weight = router.gate.weight.detach().clone().requires_grad_()
    affinity = (ref_x @ ref_weight.T).sigmoid()
    probabilities = affinity / affinity.sum(-1, keepdim=True)
    chosen = affinity.topk(2, dim=-1).indices
    counts = torch.bincount(chosen.flatten(), minlength=4)
    reference_loss = (probabilities.mean(0) * (counts / 4)).sum() * 4 * 0.1
    reference_loss.backward()
    if groups > 1:
        assert not torch.equal(indices.sort(-1).values, chosen.sort(-1).values)
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(router.gate.weight.grad, ref_weight.grad)
