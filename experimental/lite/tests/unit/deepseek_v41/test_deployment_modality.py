# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generated image sentinels have separate native routing and Engram semantics."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("deployment", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_sequence_routes_generated_sentinels_without_masking_engram_placeholders(
    v41_core_te, monkeypatch, deployment, explicit
):
    from megatron.lite.model.deepseek_v41.lite import model as owner

    ids = torch.tensor([[7, 129263, 129264, 129265, 129266, 129267, 129268, 129269]])
    image_mask = (
        torch.tensor([[False, False, True, False, False, False, False, False]])
        if explicit
        else None
    )
    seen = {}

    def hashes(memory, inputs, mask, context):
        seen["hash_mask"] = mask
        return torch.zeros(1, ids.size(1), 1, 2, dtype=torch.int64)

    def engram(hidden, hashed, mask):
        seen["gate_mask"] = mask
        return hidden

    class Layer:
        def __init__(self):
            self.engram = engram

        def forward(self, hidden, pre, state, *, attention_kwargs, ffn_kwargs):
            seen["routing_mask"] = ffn_kwargs["image_mask"]
            return hidden, pre, state

    monkeypatch.setattr(owner, "sequence_hashes", hashes)
    model = SimpleNamespace(
        local_layer_range=(0, 1),
        deployment_math=deployment,
        engram_hash=object(),
        engram_layer_ids=[0],
        layers=[Layer()],
        topology=[SimpleNamespace(engram_slot=0)],
    )
    hidden = torch.randn(1, ids.size(1), 4, 8, requires_grad=True)
    pre = torch.randn(1, ids.size(1), 4, requires_grad=True)
    out, shift = owner.DeepseekV41Model._sequence(
        model, hidden, pre, input_ids=ids, image_mask=image_mask
    )
    assert out is hidden and shift is pre
    if deployment:
        # Native router range [129264,129269); native Engram masks only 129264.
        expected_route = torch.tensor(
            [[False, False, True, True, True, True, True, False]]
        )
        expected_keep = torch.tensor(
            [[True, True, False, True, True, True, True, True]]
        )
        assert torch.equal(seen["routing_mask"], expected_route)
        assert torch.equal(seen["hash_mask"], expected_keep)
        assert torch.equal(seen["gate_mask"], expected_keep)
    else:
        assert seen["routing_mask"] is image_mask
        if image_mask is None:
            assert seen["hash_mask"] is None and seen["gate_mask"] is None
        else:
            assert torch.equal(seen["hash_mask"], ~image_mask)
            assert torch.equal(seen["gate_mask"], ~image_mask)
    out.sum().backward()
    assert torch.equal(hidden.grad, torch.ones_like(hidden))


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA native mixed-modality router'
)
@pytest.mark.parametrize('experts', [256, 384])
def test_mixed_modality_router_native_slots_weights_and_fp32_vjp(v41_core_te, experts):
    from megatron.lite.model.deepseek_v41.lite.moe import ModalityRouter
    from megatron.lite.primitive.parallel.state import ParallelState
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    torch.manual_seed(1837)
    config = SimpleNamespace(
        n_routed_experts=experts,
        num_experts_per_tok=6,
        hidden_size=512,
        routed_scaling_factor=1.5,
        scoring_func='sqrtsoftplus',
    )
    router = ModalityRouter(config, ParallelState()).cuda()
    router.deployment_math = True
    with torch.no_grad():
        router.bias.copy_(torch.linspace(-0.4, 0.2, experts, device='cuda'))
        router.bias_vl.copy_(torch.linspace(0.3, -0.5, experts, device='cuda'))
    ids = torch.tensor(
        [7, 129263, 129264, 129265, 129266, 129267, 129268, 129269], device='cuda'
    )
    mask = torch.tensor(
        [False, False, True, True, True, True, True, False], device='cuda'
    )
    x = torch.randn(len(ids), 512, device='cuda').bfloat16().requires_grad_()
    from megatron.lite.primitive.modules import deployment_math as dm

    logits = dm.bf16_fp32_linear(x, router.router.gate.weight, persistent=True)
    # Observe the actual projected logits without replacing its provider.
    captured = []
    original = router.router.route_logits

    def observe(projected, **kwargs):
        projected.retain_grad()
        captured.append((mask if len(captured) else ~mask, projected))
        return original(projected, **kwargs)

    router.router.route_logits = observe
    weights, slots, stats = router(x, image_mask=mask)
    expected, expected_slots = dsv4_topk(
        logits.detach(),
        router.bias,
        torch.int64,
        1.5,
        input_ids=ids,
        bias_vl=router.bias_vl,
        image_sentinel_lo=129264,
    )
    assert torch.equal(weights, expected) and torch.equal(slots, expected_slots)
    for selected, projected in captured:
        assert torch.equal(projected, logits[selected])
    assert stats.total_tokens.tolist() == [3, 5]
    incoming = torch.randn_like(weights)
    weights.backward(incoming)
    # Independent smooth score VJP at the unchanged visible logits/slots.
    leaf = logits.detach().clone().requires_grad_()
    selected = torch.nn.functional.softplus(leaf).sqrt().gather(1, slots)
    (selected / selected.sum(-1, keepdim=True) * 1.5).backward(incoming)
    for selected, projected in captured:
        assert torch.equal(projected.grad, leaf.grad[selected])
    expected_dx = (
        leaf.grad @ router.router.gate.weight.detach().bfloat16().float()
    ).to(x.dtype)
    expected_dw = leaf.grad.T @ x.detach().float()
    assert torch.equal(x.grad, expected_dx)
    assert torch.equal(router.router.gate.weight.grad, expected_dw)
    assert router.router.gate.weight.grad.dtype == torch.float32
    assert torch.isfinite(router.router.gate.weight.grad).all()
    assert (router.router.gate.weight.grad != 0).any()
