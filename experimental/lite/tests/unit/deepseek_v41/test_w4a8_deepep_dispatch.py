# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W4A8 over the route-preserving DeepEP dispatcher against unfused dispatch."""
import copy
from types import SimpleNamespace as NS

import pytest
import torch


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(128, 128, bias=False)
        self.w2 = torch.nn.Linear(128, 128, bias=False)
        self.w3 = torch.nn.Linear(128, 128, bias=False)
        self.swiglu_limit = 10.0


def _router():
    from megatron.lite.model.deepseek_v41.lite.moe import ModalityRouter
    from megatron.lite.primitive.parallel.state import ParallelState

    cfg = NS(
        n_routed_experts=8,
        num_experts_per_tok=3,
        hidden_size=128,
        norm_topk_prob=True,
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    router = ModalityRouter(cfg, ParallelState())
    # Experts 5..7 receive no tokens.
    router.router.gate.weight.data.zero_()
    router.router.gate.weight.data[:5].normal_(std=0.1)
    return router


@pytest.mark.parametrize('routed', [False, True], ids=['summed_bwd', 'routed_bwd'])
@pytest.mark.parametrize('official', [False, True])
def test_w4a8_deepep_matches_unfused_dispatch(
    v41_core_te, monkeypatch, official, routed
):
    from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE
    from megatron.lite.primitive.modules.route_preserving_deepep import (
        RoutePreservingDeepEPDispatcher,
    )

    monkeypatch.setenv('MEGATRON_LITE_MOE_PERMUTE_FUSION', '0')
    torch.manual_seed(905)
    router = _router()
    bank = [_Expert() for _ in range(8)]
    models = [
        DeepseekV41MoE(
            copy.deepcopy(router),
            copy.deepcopy(bank),
            use_deepep=use_deepep,
            w4a8=True,
            official_expert_contract=official,
            deepep_routed_backward=use_deepep and routed,
        )
        for use_deepep in (False, True)
    ]
    assert isinstance(models[1].dispatcher, RoutePreservingDeepEPDispatcher)
    assert models[1].dispatcher.routed_backward == routed
    data = torch.randn(2, 9, 128).bfloat16()
    incoming = torch.randn_like(data)
    outputs, inputs = [], []
    for model in models:
        x = data.clone().requires_grad_()
        out = model(x)
        out.backward(incoming)
        outputs.append(out)
        inputs.append(x.grad)
    assert torch.equal(outputs[0], outputs[1])
    if routed:
        # Route gradients reduced by the unfused dispatcher's own VJP.
        assert torch.equal(inputs[0], inputs[1])
    else:
        # The route scatter VJP sums a token's slot gradients in BF16 slot
        # order; the unfused unpermute sums them in another order (and DeepEP
        # combine owns that sum across ranks): dX agrees to BF16 rounding.
        torch.testing.assert_close(inputs[0], inputs[1], rtol=2**-6, atol=2**-9)
    assert torch.equal(
        models[0].gate.router.gate.weight.grad, models[1].gate.router.gate.weight.grad
    )
    for reference, actual in zip(models[0].experts, models[1].experts):
        for name in ('w1', 'w2', 'w3'):
            expected = getattr(reference, name).weight.grad
            grad = getattr(actual, name).weight.grad
            if expected is None:
                assert grad is None
            else:
                assert torch.equal(grad, expected)


def test_unreduced_rows_keep_every_route(v41_core_te):
    from megatron.lite.primitive.modules.route_preserving_deepep import (
        RoutePreservingDeepEPDispatcher,
    )
    from megatron.lite.primitive.parallel.state import ParallelState

    dispatcher = RoutePreservingDeepEPDispatcher(
        num_experts=3, hidden_size=16, ps=ParallelState(), use_deepep=True
    )
    hidden = torch.arange(3, dtype=torch.bfloat16)[:, None].expand(3, 16) + 1
    # Token 0 routes twice to expert 0; expert 2 gets no tokens.
    indices = torch.tensor([[0, 0], [1, 0], [1, 0]])
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.5, 0.5]])

    rows, counts, probs = dispatcher.dispatch(hidden, weights, indices)
    assert counts.device.type == 'cpu' and counts.tolist() == [4, 2, 0]
    assert rows[:, 0].tolist() == [1, 1, 2, 3, 2, 3]
    assert torch.equal(probs, torch.tensor([0.25, 0.75, 0.6, 0.5, 0.4, 0.5]))
    with pytest.raises(NotImplementedError, match='combine_unreduced'):
        dispatcher.combine(rows)

    out, row_token, row_expert = dispatcher.combine_unreduced(rows * 10)
    assert row_token.tolist() == [0, 0, 1, 1, 2, 2]
    assert row_expert.tolist() == [0, 0, 1, 0, 1, 0]
    assert out[:, 0].tolist() == [10, 10, 20, 20, 30, 30]


def test_routed_backward_requires_w4a8_deepep(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE

    with pytest.raises(ValueError, match='Routed DeepEP backward requires'):
        DeepseekV41MoE(
            _router(),
            [_Expert() for _ in range(8)],
            w4a8=True,
            deepep_routed_backward=True,
        )


def test_routed_backward_rejects_repeated_experts(v41_core_te):
    from megatron.lite.primitive.modules.route_preserving_deepep import (
        RoutePreservingDeepEPDispatcher,
    )
    from megatron.lite.primitive.parallel.state import ParallelState

    dispatcher = RoutePreservingDeepEPDispatcher(
        num_experts=3,
        hidden_size=16,
        ps=ParallelState(),
        use_deepep=True,
        routed_backward=True,
    )
    hidden = torch.ones(2, 16, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.full((2, 2), 0.5, requires_grad=True)
    with pytest.raises(RuntimeError, match='distinct experts per token'):
        dispatcher.dispatch(hidden, weights, torch.tensor([[0, 0], [1, 2]]))
