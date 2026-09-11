from types import SimpleNamespace

import torch
from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE, ModalityRouter
from torch import nn


def make_router():
    config = SimpleNamespace(
        hidden_size=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        scoring_func='sqrtsoftplus',
        aux_loss_alpha=0.0001,
    )
    router = ModalityRouter(config, SimpleNamespace(tp_size=1), gate_temperature=0.7)
    with torch.no_grad():
        router.router.gate.weight.copy_(torch.eye(3))
        router.bias.copy_(torch.tensor([0.0, 0.0, 4.0]))
        router.bias_vl.copy_(torch.tensor([4.0, 0.0, 0.0]))
    return router


def test_modality_selection_unbiased_weights_gradients_and_two_updates():
    router = make_router()
    x = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 1.0, 2.0]], requires_grad=True)
    image = torch.tensor([False, True])
    weights, indices, stats = router(x, image)
    assert indices.tolist() == [[0, 2], [0, 2]]
    raw = torch.nn.functional.softplus(x / 0.7).sqrt()
    picked = raw[:, [0, 2]]
    expected = picked / picked.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected)
    cotangent = torch.tensor([[1.0, -2.0], [3.0, -1.0]])
    ga = torch.autograd.grad((weights * cotangent).sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad((expected * cotangent).sum(), x)[0]
    torch.testing.assert_close(ga, gb)
    assert stats.counts.dtype == torch.int64
    assert stats.counts.tolist() == [[1, 0, 1], [1, 0, 1]]
    assert stats.total_tokens.tolist() == [1, 1]
    assert not router.router.compute_aux_loss
    old = torch.stack([router.bias.clone(), router.bias_vl.clone()])
    router.update_bias(stats)
    delta = torch.tensor([[-0.001, 0.001, -0.001], [-0.001, 0.001, -0.001]])
    torch.testing.assert_close(torch.stack([router.bias, router.bias_vl]), old + delta)
    router.update_bias(stats)
    torch.testing.assert_close(
        torch.stack([router.bias, router.bias_vl]), old + 2 * delta
    )


def test_empty_modality_and_forward_has_no_bias_side_effects():
    router = make_router()
    old = router.bias_vl.clone()
    _, _, stats = router(torch.ones(2, 3), torch.zeros(2, dtype=torch.bool))
    assert stats.total_tokens.tolist() == [2, 0]
    assert not stats.counts[1].any()
    torch.testing.assert_close(router.bias_vl, old, atol=0, rtol=0)
    router.update_bias(stats)
    torch.testing.assert_close(router.bias_vl, old, atol=0, rtol=0)
    w, ids, empty = router(torch.empty(0, 3))
    assert w.shape == ids.shape == (0, 2)
    router.update_bias(empty)
    assert torch.isfinite(router.bias).all()


def test_real_moe_dispatch_matches_independent_weighted_experts_and_vjps():
    router = make_router()

    class WeightedLinear(nn.Linear):
        def forward(self, x, weights=None):
            return super().forward(x if weights is None else x * weights)

    experts = nn.ModuleList([WeightedLinear(3, 3, bias=False) for _ in range(3)])
    with torch.no_grad():
        for i, expert in enumerate(experts):
            expert.weight.copy_(torch.eye(3) * (i + 1))
    moe = DeepseekV41MoE(router, experts, nn.Identity())
    x = torch.tensor([[[2.0, 1.0, -1.0], [-1.0, 1.0, 2.0]]], requires_grad=True)
    mask = torch.tensor([[False, True]])
    actual = moe(x, image_mask=mask)
    raw = torch.nn.functional.softplus(x / 0.7).sqrt()[..., [0, 2]]
    weights = raw / raw.sum(-1, keepdim=True) * 1.5
    expected = x + weights[..., :1] * experts[0](x) + weights[..., 1:] * experts[2](x)
    torch.testing.assert_close(actual, expected)
    params = [x, experts[0].weight, experts[2].weight]
    a = torch.autograd.grad(actual.square().sum(), params)
    b = torch.autograd.grad(expected.square().sum(), params)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


def test_ds4_tp_count_scope_keeps_both_modality_reductions(monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import moe

    group = object()
    seen = []

    def reduce(counts, *, group):
        assert counts.dtype == torch.int64
        seen.append(counts.clone())
        counts.add_(torch.tensor([[0, 2, 0], [1, 0, 1]]))

    monkeypatch.setattr(moe.dist, 'all_reduce', reduce)
    monkeypatch.setattr(moe.dist, 'get_world_size', lambda *, group: 2)
    stats = moe.reduce_modality_load(
        torch.tensor([[0, 2]]), torch.tensor([False]), 3, group
    )
    assert len(seen) == 1 and seen[0].tolist() == [[1, 0, 1], [0, 0, 0]]
    assert stats.counts.tolist() == [[1, 2, 1], [1, 0, 1]]
    # DS4 denominator deliberately uses local count * TP, not a global recount.
    assert stats.total_tokens.tolist() == [2, 0]


def test_parent_dtype_conversion_keeps_modality_bias_fp32():
    router = make_router()
    with torch.no_grad():
        router.bias.add_(0.0012345)
        router.bias_vl.add_(0.0012345)
    before = [router.bias.clone(), router.bias_vl.clone()]
    router.bfloat16()
    for bias, expected in zip((router.bias, router.bias_vl), before):
        assert bias.dtype == torch.float32
        torch.testing.assert_close(bias, expected, atol=0, rtol=0)
