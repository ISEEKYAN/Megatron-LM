# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for the integer weight-only QAT primitive (phase 1).

Covers the design's three-state separation: master-weight identity, fake-quant
forward + STE backward numerics, disabled=bit-identical, and the packed
deployment round-trip (K-0150 maxdiff=0).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    WeightFakeQuant,
    _FakeQuantizeSTE,
    apply_qat_to_chunks,
    compute_amax,
    dequantize_weight,
    fake_quantize_weight,
    normalize_qat_spec,
    pack_int4,
    quantize_weight,
    unpack_int4,
)

pytestmark = pytest.mark.mlite


# --------------------------------------------------------------------------- config


def test_spec_defaults_are_inert_and_normalize():
    spec = normalize_qat_spec(None)
    assert spec.enabled is False
    assert normalize_qat_spec({"enabled": True, "format": "int4", "group_size": 8}).num_bits == 4
    assert normalize_qat_spec(QATSpec(enabled=True)).num_bits == 8
    with pytest.raises(TypeError, match="QAT config"):
        normalize_qat_spec(object())


def test_spec_rejects_deferred_and_unsupported_formats():
    for fmt in ("nvfp4_w4a16", "mxfp4", "fp8"):
        with pytest.raises(ValueError, match="deferred"):
            QATSpec(enabled=True, format=fmt)
    with pytest.raises(ValueError, match="Unknown QAT format"):
        QATSpec(enabled=True, format="int3")
    with pytest.raises(ValueError, match="activation quantization"):
        QATSpec(enabled=True, activation_bits=4)
    with pytest.raises(ValueError, match="learnable_scales"):
        QATSpec(enabled=True, learnable_scales=True)
    # disabled spec never validates format -> stays inert
    assert QATSpec(enabled=False, format="nvfp4_w4a16").enabled is False


def test_targets_module_skips_ignore_patterns():
    spec = QATSpec(enabled=True)
    assert spec.targets_module("layers.0.mlp.gate_up")
    assert not spec.targets_module("lm_head")
    assert not spec.targets_module("layers.0.mlp.router.gate")


# --------------------------------------------------------------------------- numerics


@pytest.mark.parametrize("num_bits,fmt", [(8, "int8"), (4, "int4")])
@pytest.mark.parametrize("group_size", [0, -1, 4])
def test_fake_quant_matches_manual_qdq(num_bits, fmt, group_size):
    torch.manual_seed(0)
    w = torch.randn(6, 8, dtype=torch.float32)
    spec = QATSpec(enabled=True, format=fmt, group_size=group_size)
    w_hat = fake_quantize_weight(w, spec)

    # manual reference
    qmax = (1 << (num_bits - 1)) - 1
    if group_size == 0:
        scale = w.abs().amax() / qmax
        ref = torch.round(w / scale).clamp(-qmax, qmax) * scale
    elif group_size == -1:
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        ref = torch.round(w / scale).clamp(-qmax, qmax) * scale
    else:
        v = w.reshape(6, 8 // group_size, group_size)
        scale = v.abs().amax(dim=2, keepdim=True) / qmax
        ref = (torch.round(v / scale).clamp(-qmax, qmax) * scale).reshape(6, 8)
    torch.testing.assert_close(w_hat, ref, rtol=0, atol=0)


def test_fake_quant_error_bounded_by_scale():
    # DQ error must be at most half a quantization step per element.
    w = torch.randn(4, 16, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="int8", group_size=-1)
    w_hat = fake_quantize_weight(w, spec)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127).expand_as(w)
    assert torch.all((w - w_hat).abs() <= scale / 2 + 1e-6)


def test_affine_quant_reconstructs_range():
    w = torch.linspace(-3.0, 5.0, steps=32).reshape(2, 16)
    spec = QATSpec(enabled=True, format="int8", symmetric=False, group_size=0)
    w_hat = fake_quantize_weight(w, spec)
    # affine covers the asymmetric [-3,5] range; error <= one step
    step = (w.max() - w.min()) / 255
    assert torch.all((w - w_hat).abs() <= step + 1e-5)


# --------------------------------------------------------------------------- STE


def test_ste_passes_gradient_through_qdq():
    w = torch.randn(5, 8, dtype=torch.float32, requires_grad=True)
    spec = QATSpec(enabled=True, format="int4", group_size=0)
    fake_quantize_weight(w, spec).sum().backward()
    # dynamic max-calibration: no element saturates, so STE is identity (grad=1).
    assert w.grad is not None
    torch.testing.assert_close(w.grad, torch.ones_like(w))


def test_ste_clip_zeroes_saturated_grad_and_passthrough_toggle():
    # Drive saturation explicitly with a too-small scale so codes exceed [-7,7].
    w = torch.tensor([[-2.0, -0.1, 0.1, 2.0]], requires_grad=True)
    scale = torch.tensor(0.1)  # code = w/scale = [-20,-1,1,20] -> saturates at +-7
    out = _FakeQuantizeSTE.apply(w, scale, None, -7, 7, True)
    out.sum().backward()
    assert w.grad.tolist() == [[0.0, 1.0, 1.0, 0.0]]

    w2 = w.detach().clone().requires_grad_(True)
    out2 = _FakeQuantizeSTE.apply(w2, scale, None, -7, 7, False)  # pure pass-through
    out2.sum().backward()
    assert w2.grad.tolist() == [[1.0, 1.0, 1.0, 1.0]]


# --------------------------------------------------------------------------- master identity


def test_parametrization_preserves_master_weight_identity():
    lin = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    master_before = lin.weight
    spec = QATSpec(enabled=True, format="int4", group_size=-1)
    parametrize.register_parametrization(
        lin, "weight", WeightFakeQuant(spec, lin.weight.shape), unsafe=True
    )
    # master survives untouched as .original, still trainable, still bf16
    original = lin.parametrizations.weight.original
    assert original is master_before
    assert original.requires_grad
    assert original.dtype == torch.bfloat16
    # accessing .weight yields W_hat (quantized), not the master
    assert lin.weight.dtype == torch.bfloat16
    assert not torch.equal(lin.weight, original)
    # gradient flows back to the master through STE
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    lin(x).sum().backward()
    assert original.grad is not None


# --------------------------------------------------------------------------- disabled = bit-identical


def _toy_chunk():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 12, bias=False)
            self.proj = nn.Linear(12, 8, bias=False)
            self.lm_head = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return self.lm_head(self.proj(self.qkv(x)))

    return Toy()


def test_disabled_spec_is_bit_identical():
    torch.manual_seed(1)
    chunk = _toy_chunk()
    x = torch.randn(4, 8)
    ref = chunk(x)
    stats = apply_qat_to_chunks([chunk], QATSpec(enabled=False))
    assert stats == {"quantized_modules": 0, "skipped_ignored": 0, "skipped_no_weight": 0}
    assert not parametrize.is_parametrized(chunk.qkv, "weight")
    torch.testing.assert_close(chunk(x), ref, rtol=0, atol=0)


def test_apply_quantizes_targets_and_skips_ignored():
    torch.manual_seed(2)
    chunk = _toy_chunk()
    spec = QATSpec(enabled=True, format="int8", group_size=-1)
    stats = apply_qat_to_chunks([chunk], spec)
    assert stats["quantized_modules"] == 2  # qkv + proj
    assert parametrize.is_parametrized(chunk.qkv, "weight")
    assert parametrize.is_parametrized(chunk.proj, "weight")
    assert not parametrize.is_parametrized(chunk.lm_head, "weight")  # ignored
    # forward changes (weights are now fake-quantized) and grads flow to masters
    x = torch.randn(4, 8)
    chunk(x).sum().backward()
    assert chunk.qkv.parametrizations.weight.original.grad is not None
    assert chunk.lm_head.weight.grad is not None  # untouched, plain param


# --------------------------------------------------------------------------- deployment round-trip


@pytest.mark.parametrize("fmt", ["int8", "int4"])
@pytest.mark.parametrize("group_size", [0, -1, 4])
def test_export_roundtrip_matches_fake_quant(fmt, group_size):
    torch.manual_seed(3)
    w = torch.randn(6, 8, dtype=torch.float32)
    spec = QATSpec(enabled=True, format=fmt, group_size=group_size)
    packed = quantize_weight(w, spec)
    recon = dequantize_weight(packed, spec)
    # deploy dequant must equal the training fake-quant exactly (K-0150 maxdiff=0)
    torch.testing.assert_close(recon, fake_quantize_weight(w, spec), rtol=0, atol=0)


def test_int4_pack_unpack_roundtrip():
    torch.manual_seed(4)
    w = torch.randn(6, 8, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="int4", group_size=-1)
    codes = quantize_weight(w, spec)["qweight"]
    assert codes.min() >= -7 and codes.max() <= 7
    packed = pack_int4(codes)
    assert packed.dtype == torch.uint8 and packed.shape == (6, 4)
    torch.testing.assert_close(unpack_int4(packed), codes, rtol=0, atol=0)


def test_amax_buffer_shapes():
    w = torch.randn(6, 8)
    assert compute_amax(w, 0).shape == torch.Size([])
    assert compute_amax(w, -1).shape == torch.Size([6, 1])
    assert compute_amax(w, 4).shape == torch.Size([6, 2, 1])
