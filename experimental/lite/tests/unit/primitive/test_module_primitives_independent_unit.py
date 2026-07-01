# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.lora import (
    GroupedLinearLoRA,
    LinearLoRA,
    SharedGroupedLinearLoRA,
    freeze_non_lora_params,
    lora_scaling,
    normalize_lora_config,
    olora_tail_factors,
    trainable_param_stats,
)

pytestmark = pytest.mark.mlite


def test_lora_config_aliases_and_trainable_param_accounting():
    cfg = normalize_lora_config({"enabled": True, "rank": 2, "alpha": 6, "targets": ["qkv", "fc2"]})

    assert cfg.enabled
    assert cfg.scale == 3.0
    assert cfg.targets() == {"linear_qkv", "linear_fc2"}
    assert cfg.targets_module("qkv")
    assert cfg.targets_module("linear_fc2")
    assert not normalize_lora_config({"enabled": False, "rank": 8}).enabled
    with pytest.raises(TypeError, match="LoRA config"):
        normalize_lora_config(object())

    class TinyAdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 2)
            self.lora_adapter = nn.Linear(3, 2)

    model = TinyAdapterModel()
    stats = freeze_non_lora_params(model)

    assert stats["lora_tensors"] == 2
    assert stats["frozen_tensors"] == 2
    assert not model.base.weight.requires_grad
    assert model.lora_adapter.weight.requires_grad
    assert trainable_param_stats(model) == {
        "trainable_tensors": 2,
        "trainable_numel": model.lora_adapter.weight.numel() + model.lora_adapter.bias.numel(),
    }


def test_rslora_uses_sqrt_rank_scaling():
    # standard LoRA: alpha/rank; rsLoRA: alpha/sqrt(rank). rank=4, alpha=8 -> 2.0 vs 4.0.
    assert lora_scaling(4, 8, use_rslora=False) == 2.0
    assert lora_scaling(4, 8, use_rslora=True) == 4.0
    assert lora_scaling(4, None, use_rslora=True) == 4 / (4**0.5)  # alpha defaults to rank

    cfg = normalize_lora_config({"rank": 4, "alpha": 8, "use_rslora": True})
    assert cfg.use_rslora
    assert cfg.scale == 4.0
    assert normalize_lora_config({"rank": 4, "alpha": 8}).scale == 2.0  # default off

    assert LinearLoRA(2, 1, rank=4, alpha=8).scale == 2.0
    assert LinearLoRA(2, 1, rank=4, alpha=8, use_rslora=True).scale == 4.0
    assert SharedGroupedLinearLoRA(2, 2, 2, rank=4, alpha=8, use_rslora=True).scale == 4.0
    assert GroupedLinearLoRA(2, 2, 2, rank=4, alpha=8, use_rslora=True).scale == 4.0

    # modules record use_rslora so the adapter round-trip can invert scale->alpha correctly
    assert LinearLoRA(2, 1, rank=4, alpha=8, use_rslora=True).use_rslora is True
    assert LinearLoRA(2, 1, rank=4, alpha=8).use_rslora is False


def test_rslora_forward_scales_delta_by_sqrt_rank():
    # identical adapter weights, rsLoRA output = sqrt(rank) x standard output.
    # alpha=8 != rank=4 so the std path is non-trivially scaled (2.0), not 1.0.
    torch.manual_seed(0)
    a = torch.randn(4, 3)
    b = torch.randn(2, 4)
    std = LinearLoRA(3, 2, rank=4, alpha=8, dropout=0.0)
    rs = LinearLoRA(3, 2, rank=4, alpha=8, dropout=0.0, use_rslora=True)
    with torch.no_grad():
        for layer in (std, rs):
            layer.lora_a.copy_(a)
            layer.lora_b.copy_(b)
    x = torch.randn(1, 3)
    torch.testing.assert_close(rs(x), std(x) * (4**0.5))


def test_olora_tail_factors_are_minor_subspace_without_sigma():
    # B0=U_-r, A0=V_-r^T from the SMALLEST r singular values, orthonormal (no Sigma scaling).
    torch.manual_seed(0)
    out_f, in_f, rank = 16, 12, 3
    w = torch.randn(out_f, in_f)
    u, _s, vh = torch.linalg.svd(w, full_matrices=False)  # singular values descending
    b0, a0 = olora_tail_factors(w, rank)

    assert b0.shape == (out_f, rank)
    assert a0.shape == (rank, in_f)
    # orthonormal => NO singular-value scaling injected (unlike MiLoRA's Sigma^1/2)
    torch.testing.assert_close(b0.t() @ b0, torch.eye(rank), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a0 @ a0.t(), torch.eye(rank), atol=1e-4, rtol=1e-4)
    # spans the MINOR subspace (smallest r vectors), up to per-vector sign
    torch.testing.assert_close(b0.abs(), u[:, -rank:].abs(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a0.abs(), vh[-rank:, :].abs(), atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="exceeds"):
        olora_tail_factors(w, rank=13)


def test_linear_lora_olora_tail_preserves_init_output():
    # PiSSA-style residual: effective output unchanged at init, but adapter delta is NON-zero.
    torch.manual_seed(0)
    in_f, out_f, rank = 12, 16, 3
    layer = LinearLoRA(in_f, out_f, rank, alpha=6)  # scale = 6/3 = 2
    w = torch.randn(out_f, in_f)
    w0 = w.clone()
    x = torch.randn(4, in_f)
    base_out = x @ w0.t()

    layer.olora_tail_init_(w)  # sets lora_a/b, subtracts scale*B0@A0 from w in place

    effective = x @ w.t() + layer(x)  # residual base + adapter
    torch.testing.assert_close(effective, base_out, atol=1e-4, rtol=1e-4)
    assert layer.lora_b.abs().sum() > 0  # unlike standard zero-init B


def test_shared_grouped_olora_tail_preserves_every_expert_output():
    # one shared adapter; subtract same delta from each expert -> each expert output preserved.
    torch.manual_seed(0)
    n_exp, in_f, out_f, rank = 3, 12, 16, 2
    layer = SharedGroupedLinearLoRA(n_exp, in_f, out_f, rank, alpha=4)  # scale = 2
    ws = [torch.randn(out_f, in_f) for _ in range(n_exp)]
    w0s = [w.clone() for w in ws]

    layer.olora_tail_init_(ws)

    x = torch.randn(5, in_f)
    shared = (x @ layer.lora_a.t()) @ layer.lora_b.t() * layer.scale
    for w_after, w0 in zip(ws, w0s, strict=True):
        torch.testing.assert_close(x @ w_after.t() + shared, x @ w0.t(), atol=1e-4, rtol=1e-4)


def test_olora_tail_init_rejects_tp_sharded_weight():
    # tp>1 / partitioned surfaces need a distributed SVD; we guard against silent wrong init.
    # rank must be divisible by the partition size (4 % 2 == 0) so construction succeeds and
    # the guard — not the constructor — is what rejects the sharded surface.
    layer = LinearLoRA(12, 16, rank=4, alpha=6, rank_partitioned_a=True, rank_partition_size=2)
    with pytest.raises(NotImplementedError, match="tp=1"):
        layer.olora_tail_init_(torch.randn(16, 12))


def test_linear_lora_forward_backward_matches_low_rank_delta():
    layer = LinearLoRA(3, 2, rank=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        layer.lora_b.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    output = layer(x)

    torch.testing.assert_close(output, torch.tensor([[10.0, 22.0]]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[8.0, 12.0, 0.0]]))


def test_grouped_lora_respects_per_expert_splits():
    layer = GroupedLinearLoRA(2, 2, 2, rank=1, alpha=1, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
        layer.lora_b.copy_(torch.tensor([[[2.0], [3.0]], [[5.0], [7.0]]]))

    x = torch.tensor([[2.0, 9.0], [4.0, 1.0], [6.0, 3.0]])
    output = layer(x, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0], [5.0, 7.0], [15.0, 21.0]]))
    with pytest.raises(ValueError, match="expected 2 splits"):
        layer(x, [3])


def test_shared_grouped_lora_uses_one_adapter_for_all_experts():
    layer = SharedGroupedLinearLoRA(2, 2, 2, rank=1, alpha=2, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, -1.0]]))
        layer.lora_b.copy_(torch.tensor([[2.0], [3.0]]))

    output = layer(torch.tensor([[3.0, 1.0], [4.0, 7.0]]), [1, 1])

    torch.testing.assert_close(output, torch.tensor([[8.0, 12.0], [-12.0, -18.0]]))


def test_mrope_interleaves_text_height_and_width_sections():
    from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding

    base = torch.arange(3 * 2 * 6, dtype=torch.float32).reshape(3, 2, 6)

    interleaved = MultimodalRotaryEmbedding._apply_interleaved_mrope(base, mrope_section=[1, 1, 1])

    expected = base[0].clone()
    expected[..., 1] = base[1, ..., 1]
    expected[..., 2] = base[2, ..., 2]
    torch.testing.assert_close(interleaved, expected)


def test_mtp_aux_loss_scaler_threads_independent_gradient(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    MTPLossAutoScaler.set_loss_scale(torch.tensor(0.125))
    output = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mtp_loss = torch.tensor(4.0, requires_grad=True)

    MTPLossAutoScaler.apply(output * 3.0, mtp_loss).sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 3.0))
    torch.testing.assert_close(mtp_loss.grad, torch.tensor(0.125))
    MTPLossAutoScaler.main_loss_backward_scale = 1.0


def test_gated_delta_static_helpers_are_finite_and_shape_stable(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    alpha = torch.tensor([[[0.0, 1.0], [-1.0, 2.0]]])
    beta = torch.tensor([[[0.0, 2.0], [-2.0, 4.0]]])

    g, beta_sigmoid = GatedDeltaNet._compute_g_and_beta(torch.zeros(2), torch.ones(2), alpha, beta)

    assert g.shape == alpha.shape
    assert beta_sigmoid.shape == beta.shape
    assert torch.isfinite(g).all()
    assert torch.isfinite(beta_sigmoid).all()
    assert torch.all(g < 0)
