import torch
from megatron.lite.model.deepseek_v41.lite import block as hc
from torch import nn


def test_initial_final_and_analytic_contraction_gradients():
    token = torch.tensor([[[2.0, 3.0]]], requires_grad=True)
    hidden, pre = hc.expand_hc(token, 2)
    assert pre.tolist() == [[[1.0, 0.0]]]
    torch.testing.assert_close(hc.contract_hc(hidden, pre), token)
    hidden = torch.tensor([[[[2.0, 3.0], [5.0, 7.0]]]], requires_grad=True)
    pre = torch.tensor([[[0.25, 0.75]]], requires_grad=True)
    result = hc.contract_hc(hidden, pre)
    torch.testing.assert_close(result, torch.tensor([[[4.25, 6.0]]]))
    (result * torch.tensor([[[2.0, -1.0]]])).sum().backward()
    torch.testing.assert_close(
        hidden.grad, torch.tensor([[[[0.5, -0.25], [1.5, -0.75]]]])
    )
    torch.testing.assert_close(pre.grad, torch.tensor([[[1.0, 3.0]]]))


def test_residual_combination_uses_source_then_destination_axes():
    residual = torch.tensor([[[[2.0], [7.0]]]])
    comb = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    actual = hc.mix_residual(
        torch.tensor([[[3.0]]]), residual, torch.tensor([[[2.0, 4.0]]]), comb
    )
    torch.testing.assert_close(actual, torch.tensor([[[[8.3], [15.2]]]]))


class Sublayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.inputs = []

    def forward(self, x, **kwargs):
        self.inputs.append(x)
        return x * self.weight


class FixedMix(nn.Module):
    def __init__(self, pre):
        super().__init__()
        self.pre = nn.Parameter(torch.tensor(pre))

    def forward(self, hidden):
        pre = self.pre.expand(*hidden.shape[:-2], -1)
        post = torch.ones_like(pre)
        comb = torch.eye(pre.shape[-1]).expand(*pre.shape[:-1], -1, -1)
        return pre, post, comb


def make_block():
    block = hc.DeepseekV41Block(2, 2, Sublayer(), Sublayer())
    block.attn_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.attn_mixes = FixedMix([0.2, 0.8])
    block.ffn_mixes = FixedMix([0.6, 0.4])
    return block


def test_shifted_coefficients_and_paired_recompute():
    from torch.utils.checkpoint import checkpoint

    block = make_block()
    hidden = torch.tensor([[[[2.0, 3.0], [5.0, 7.0]]]], requires_grad=True)
    pre = torch.tensor([[[0.9, 0.1]]], requires_grad=True)
    output, next_pre = block(hidden, pre)
    # Incoming pre, not [.2,.8], controls attention.
    torch.testing.assert_close(block.attn.inputs[-1], torch.tensor([[[2.3, 3.4]]]))
    # Attention residual copies are [6.6,9.8] and [9.6,13.8].
    torch.testing.assert_close(block.ffn.inputs[-1], torch.tensor([[[9.0, 13.0]]]))
    torch.testing.assert_close(next_pre, torch.tensor([[[0.6, 0.4]]]))
    parameters = [hidden, pre, *block.parameters()]
    expected = torch.autograd.grad(hc.contract_hc(output, next_pre).sum(), parameters)
    recomputed = checkpoint(block, hidden, pre, use_reentrant=False)
    actual = torch.autograd.grad(hc.contract_hc(*recomputed).sum(), parameters)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert pre.grad is None  # autograd.grad above does not mutate leaf state
    assert all(torch.count_nonzero(g) for g in expected)


def test_sinkhorn_zero_projection_card_and_gradients():
    from megatron.lite.model.deepseek_v41.lite.block import HCMixes

    mix = HCMixes(2, 2, hc_eps=0.01, iterations=1)
    with torch.no_grad():
        mix.fn.zero_()
    hidden = torch.tensor([[[[2.0, 3.0], [5.0, 7.0]]]], requires_grad=True)
    pre, post, comb = mix(hidden)
    torch.testing.assert_close(pre, torch.full((1, 1, 2), 0.51))
    torch.testing.assert_close(post, torch.ones(1, 1, 2))
    torch.testing.assert_close(comb, torch.full((1, 1, 2, 2), 0.51 / 1.03))
    pre.sum().backward()
    norm = (21.75 + 1e-6) ** -0.5
    torch.testing.assert_close(
        mix.fn.grad[:2],
        hidden.detach().flatten(2).reshape(1, 4).repeat(2, 1) * (0.25 * norm),
    )
    torch.testing.assert_close(mix.base.grad[:2], torch.full((2,), 0.25))
    assert torch.count_nonzero(mix.fn.grad[2:]) == 0
