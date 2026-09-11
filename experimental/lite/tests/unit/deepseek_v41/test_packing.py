import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.attention import (
    AttentionState,
    CSA2Attention,
)
from megatron.lite.model.deepseek_v41.lite.block import DeepseekV41Block, contract_hc
from megatron.lite.model.deepseek_v41.lite.engram import Engram, NgramHash
from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE, SwiGLUExpert
from megatron.lite.model.deepseek_v41.lite.packing import packed_forward
from test_attention import config
from torch import nn


class Sequence(nn.Module):
    def __init__(self):
        super().__init__()
        self.hash = NgramHash(
            list(range(8)), 0, torch.tensor([[3, 5, 7]]), torch.tensor([[[11], [13]]])
        )
        self.engram = Engram(32, 2, nn.Embedding(24, 4), nn.Linear(8, 96, bias=False))
        from types import SimpleNamespace

        from megatron.lite.model.deepseek_v41.lite.moe import ModalityRouter

        cfg = SimpleNamespace(
            hidden_size=32,
            n_routed_experts=3,
            num_experts_per_tok=2,
            routed_scaling_factor=1.5,
            scoring_func='sqrtsoftplus',
        )
        self.layers = nn.ModuleList()
        for i in (2, 3, 8, 14, 20, 21, 24):
            router = ModalityRouter(cfg, SimpleNamespace(tp_size=1))
            experts = [
                SwiGLUExpert(nn.Linear(32, 16), nn.Linear(16, 32), nn.Linear(32, 16))
                for _ in range(3)
            ]
            self.layers.append(
                DeepseekV41Block(
                    32,
                    2,
                    CSA2Attention(config(), i).float(),
                    DeepseekV41MoE(router, experts),
                )
            )

    def forward(self, h, p, *, input_ids, image_mask):
        hashed = self.hash(input_ids, ~image_mask)
        h = self.engram(h, hashed[:, :, 0], ~image_mask)
        state = AttentionState()
        for layer in self.layers:
            h, p, state = layer.forward_with_state(
                h, p, state, ffn_kwargs={'image_mask': image_mask}
            )
        return h, p


def test_packed_b_only_gradient_and_parameter_contributions_ignore_a():
    torch.manual_seed(84)
    model = Sequence()
    captured = []
    handles = [
        layer.ffn.gate.register_forward_hook(
            lambda module, inputs, output: captured.append(output[2])
        )
        for layer in model.layers
    ]
    original_bias = [
        torch.stack([layer.ffn.gate.bias, layer.ffn.gate.bias_vl]).clone()
        for layer in model.layers
    ]

    def latest_stats():
        return captured[-len(model.layers) :]

    h = torch.randn(1, 9, 2, 32, requires_grad=True)
    p = torch.randn(1, 9, 2, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1, 2]])
    mask = torch.tensor([[False, True, False, False, False, False, True, False, False]])
    bounds = torch.tensor([0, 4, 9], dtype=torch.int32)
    params = [q for q in model.parameters() if q.requires_grad]

    def run(h, p, ids):
        out, mix = packed_forward(model, h, p, bounds, input_ids=ids, image_mask=mask)
        return contract_hc(out, mix)[:, 4:]

    actual = run(h, p, ids)
    actual_stats = latest_stats()
    bh, bp = model(h[:, 4:], p[:, 4:], input_ids=ids[:, 4:], image_mask=mask[:, 4:])
    expected = contract_hc(bh, bp)
    expected_stats = latest_stats()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ga = torch.autograd.grad(actual.square().mean(), [h, p, *params], allow_unused=True)
    gb = torch.autograd.grad(
        expected.square().mean(), [h, p, *params], allow_unused=True
    )
    for a, b in zip(ga, gb):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not ga[0][:, :4].any() and not ga[1][:, :4].any()
    changed = h.detach().clone().requires_grad_()
    with torch.no_grad():
        changed[:, :4].mul_(10)
    ids2 = ids.clone()
    ids2[:, :4] = 7
    perturbed = run(changed, p, ids2)
    for a, b, c in zip(actual_stats, expected_stats, latest_stats()):
        torch.testing.assert_close(a.counts, b.counts, atol=0, rtol=0)
        torch.testing.assert_close(a.counts, c.counts, atol=0, rtol=0)
        torch.testing.assert_close(a.total_tokens, c.total_tokens, atol=0, rtol=0)
    for layer, old in zip(model.layers, original_bias):
        torch.testing.assert_close(
            torch.stack([layer.ffn.gate.bias, layer.ffn.gate.bias_vl]),
            old,
            atol=0,
            rtol=0,
        )
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(perturbed, expected, atol=0, rtol=0)
    gc = torch.autograd.grad(
        perturbed.square().mean(), [changed, p, *params], allow_unused=True
    )
    for a, c in zip(ga, gc):
        if a is None or c is None:
            assert a is c
        else:
            torch.testing.assert_close(a, c, atol=0, rtol=0)


@pytest.mark.parametrize(
    'bounds',
    [
        torch.tensor([0.0, 3.0]),
        torch.tensor([1, 3]),
        torch.tensor([0, 2, 2, 3]),
        torch.tensor([[0, 3]]),
    ],
)
def test_invalid_packing_is_rejected(bounds):
    with pytest.raises(ValueError):
        packed_forward(
            lambda *args, **kwargs: args,
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 3, 2),
            bounds,
        )
