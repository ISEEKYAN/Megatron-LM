# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from types import SimpleNamespace
from megatron.lite.model.deepseek_v41.lite import block as hc


@pytest.mark.parametrize('copies', [2, 4])
def test_v41_mhc_source_destination_orientation(copies):
    torch.manual_seed(9)
    residual = torch.randn(1, 2, copies, 3, requires_grad=True)
    output = torch.randn(1, 2, 3, requires_grad=True)
    post = torch.randn(1, 2, copies, requires_grad=True)
    comb = torch.randn(1, 2, copies, copies, requires_grad=True)
    expected = torch.stack(
        [
            post[..., j, None] * output
            + sum((comb[..., i, j, None] * residual[..., i, :] for i in range(copies)))
            for j in range(copies)
        ],
        -2,
    )
    actual = hc.mix_residual(output, residual, post, comb)
    torch.testing.assert_close(actual, expected)
    args = (output, residual, post, comb)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), args),
        torch.autograd.grad(expected.square().sum(), args),
    ):
        torch.testing.assert_close(a, b)



@pytest.mark.parametrize("copies", [2, 3, 4])
def test_v41_mhc_two_sublayer_shift_uses_unequal_coefficients(copies):
    """Independently derive the two shifted HC inputs, not block internals."""

    class FixedMix(torch.nn.Module):
        def __init__(self, pre):
            super().__init__()
            self.pre = pre

        def forward(self, hidden):
            copies = hidden.shape[-2]
            post = torch.full_like(self.pre, 0.25)
            comb = torch.eye(copies).expand(*hidden.shape[:2], -1, -1)
            return self.pre, post, comb

    class RecordInput(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().clone())
            return x + self.increment

    hidden = torch.arange(1, 1 + copies * 2).reshape(1, 1, copies, 2).float()
    pre_mix = torch.arange(1, copies + 1).reshape(1, 1, copies).float()
    pre_mix = pre_mix / pre_mix.sum(-1, keepdim=True)
    expected_hidden, expected_pre = hidden.clone(), pre_mix.clone()
    for layer in range(2):
        attn_pre = pre_mix.roll(layer + 1, -1) * (layer + 2)
        ffn_pre = pre_mix.flip(-1) * (layer + 3)
        attention, ffn = RecordInput(17), RecordInput(-9)
        block = hc.DeepseekV41Block(2, copies, attention, ffn)
        block.attn_norm = torch.nn.Identity()
        block.ffn_norm = torch.nn.Identity()
        block.attn_mixes = FixedMix(attn_pre)
        block.ffn_mixes = FixedMix(ffn_pre)

        hidden, returned_pre = block(hidden, pre_mix)

        # Independent equations across both sublayers AND the next block boundary.
        expected_attn_input = (expected_hidden * expected_pre.unsqueeze(-1)).sum(-2)
        expected_hidden = expected_hidden + 0.25 * (expected_attn_input + 17).unsqueeze(
            -2
        )
        expected_ffn_input = (expected_hidden * attn_pre.unsqueeze(-1)).sum(-2)
        wrong_ffn_input = (expected_hidden * expected_pre.unsqueeze(-1)).sum(-2)
        expected_hidden = expected_hidden + 0.25 * (expected_ffn_input - 9).unsqueeze(
            -2
        )
        torch.testing.assert_close(attention.inputs[0], expected_attn_input)
        torch.testing.assert_close(ffn.inputs[0], expected_ffn_input)
        torch.testing.assert_close(hidden, expected_hidden)
        torch.testing.assert_close(returned_pre, ffn_pre)
        assert not torch.allclose(expected_ffn_input, wrong_ffn_input)
        pre_mix, expected_pre = returned_pre, ffn_pre.clone()



@pytest.mark.parametrize('image', [False, True])
def test_v41_modality_bias_selection_and_vjp(image, moe):
    config = SimpleNamespace(
        hidden_size=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        scoring_func='sqrtsoftplus',
    )
    router = moe.ModalityRouter(
        config, SimpleNamespace(tp_size=1), gate_temperature=0.7
    )
    with torch.no_grad():
        router.router.gate.weight.copy_(torch.eye(3))
        (router.bias_vl if image else router.bias)[2] = 4
    x = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    weights, indices, stats = router(x, torch.tensor([image]))
    assert indices.tolist() == [[0, 2]]
    raw = torch.nn.functional.softplus(x / 0.7).sqrt()[:, [0, 2]]
    expected = raw / raw.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected)
    cotangent = torch.tensor([[1.0, -2.0]])
    torch.testing.assert_close(
        torch.autograd.grad((weights * cotangent).sum(), x)[0],
        torch.autograd.grad((expected * cotangent).sum(), x)[0],
    )
    before = torch.stack([router.bias.clone(), router.bias_vl.clone()])
    router.update_bias(stats)
    delta = torch.zeros(2, 3)
    delta[int(image)] = torch.tensor([-0.001, 0.001, -0.001])
    torch.testing.assert_close(
        torch.stack([router.bias, router.bias_vl]), before + delta
    )
    assert not router.router.compute_aux_loss

