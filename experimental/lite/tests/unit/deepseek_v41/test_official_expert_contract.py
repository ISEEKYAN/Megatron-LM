# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Official DS4.1 expert contract: group-32 activations, weight before FC2 quant."""
import pytest
import torch


def _w():
    from megatron.lite.primitive.quantization import w4a8_experts

    return w4a8_experts


def test_official_quantizer_uses_group_32_and_reference_floor():
    w = _w()
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[0, :32] = 3e-5  # group below the 1e-4 reference floor
    x[1, 32:] = torch.linspace(-5, 5, 32).bfloat16()
    codes, scale = w.quantize_official_act(x)
    assert scale.shape == (2, 2)
    floor_scale = w._ceil_ue8m0(torch.tensor([1e-4 / 448.0]))
    assert torch.equal(scale[0, 0], floor_scale[0])
    # The g128 rule floors at 1e-10, so the tiny group gets a different scale.
    _, legacy = w.quantize_fp8_act(x[:, :32].repeat(1, 4))
    assert not torch.equal(legacy[0, 0], floor_scale[0])
    decoded = w.dequantize_fp8_act(codes, scale)
    assert torch.allclose(decoded[1, 32:], x[1, 32:].float(), rtol=0.07, atol=0)


def _problem(seed=0, experts=2, rows=(3, 5), k=64, n=32):
    torch.manual_seed(seed)
    x = torch.randn(sum(rows), k).bfloat16()
    fc1 = [torch.randn(2 * n, k) / 8 for _ in range(experts)]
    fc2 = [torch.randn(k, n) / 8 for _ in range(experts)]
    weights = torch.rand(sum(rows)) + 0.1
    return x, fc1, fc2, list(rows), weights


def test_official_expert_mlp_matches_reference_semantics():
    w = _w()
    x, fc1, fc2, splits, weights = _problem()
    limit = 1.5
    actual = w.w4a8_expert_mlp(
        x,
        fc1,
        fc2,
        splits,
        swiglu_limit=limit,
        backend="reference",
        row_weights=weights,
    )
    # Reference inference/model.py Expert.forward: FC1 on group-32 input,
    # clamp, silu(gate) * up in FP32, times the routing weight, BF16, FC2 on
    # group-32 input.
    fc1_out = w.w4a8_grouped_gemm(x, fc1, splits, backend="reference", official=True)
    gate, up = fc1_out.float().chunk(2, -1)
    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    hidden = (weights[:, None] * (torch.nn.functional.silu(gate) * up)).bfloat16()
    expected = w.w4a8_grouped_gemm(
        hidden, fc2, splits, backend="reference", official=True
    )
    assert torch.equal(actual, expected)


def test_official_expert_mlp_differs_from_legacy_weighting():
    w = _w()
    x, fc1, fc2, splits, weights = _problem(seed=1, k=128, n=128)
    official = w.w4a8_expert_mlp(
        x, fc1, fc2, splits, backend="reference", row_weights=weights
    )
    legacy = w.w4a8_expert_mlp(x, fc1, fc2, splits, backend="reference")
    weighted_after = (legacy.float() * weights[:, None]).bfloat16()
    # Weighting before FC2 quantization is a different rounding contract.
    assert not torch.equal(official, weighted_after)


def test_official_expert_mlp_routes_gradients_to_weights():
    w = _w()
    x, fc1, fc2, splits, weights = _problem(seed=2)
    x = x.requires_grad_()
    weights = weights.requires_grad_()
    fc1 = [p.requires_grad_() for p in fc1]
    out = w.w4a8_expert_mlp(
        x, fc1, fc2, splits, backend="reference", row_weights=weights
    )
    out.float().square().sum().backward()
    for tensor in (x, weights, *fc1):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert weights.grad.abs().sum() > 0


def test_official_expert_mlp_rejects_misaligned_weights():
    w = _w()
    x, fc1, fc2, splits, weights = _problem(seed=3)
    with pytest.raises(ValueError):
        w.w4a8_expert_mlp(
            x, fc1, fc2, splits, backend="reference", row_weights=weights[:-1]
        )
