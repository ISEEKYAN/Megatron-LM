# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the production MXFP8 boundary with byte-offset native operands."""
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.linear import _MXFP8Linear
from megatron.lite.primitive.quantization.frozen import FrozenWeight


@pytest.mark.parametrize('offset_a,offset_b', [(0, 0), (1, 0), (0, 1), (1, 7)])
def test_native_mxfp8_operand_alignment_preserves_bits(monkeypatch, offset_a, offset_b):
    def codes(offset):
        storage = torch.arange(128 * 128 + offset, dtype=torch.int64).to(torch.uint8)
        return storage[offset:].view(torch.float8_e4m3fn).reshape(128, 128)

    a, b = codes(offset_a), codes(offset_b)
    expected_a, expected_b = a.view(torch.uint8).clone(), b.view(torch.uint8).clone()
    sf = torch.full((4, 4), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    weight = FrozenWeight(b, sf, (128, 128), 'fp8_block')
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    module = 'vllm.model_executor.kernels.linear.mxfp8.batch_invariant'
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(_uses_tensor_core=lambda shape: True))
    module = 'vllm.model_executor.layers.quantization.utils.mxfp8_utils'
    monkeypatch.setitem(
        sys.modules,
        module,
        SimpleNamespace(
            mxfp8_e4m3_quantize=lambda *args, **kw: (a, torch.zeros(512, dtype=torch.uint8)),
            swizzle_mxfp8_scale=lambda value, *shape: value,
        ),
    )
    calls = []

    def kernel(left, right, *args):
        assert left.data_ptr() % 16 == right.data_ptr() % 16 == 0
        assert torch.equal(left.view(torch.uint8), expected_a)
        assert torch.equal(right.T.view(torch.uint8), expected_b)
        assert (left.data_ptr() == a.data_ptr()) == (offset_a == 0)
        assert (right.data_ptr() == b.data_ptr()) == (offset_b == 0)
        calls.append(True)
        return torch.zeros((128, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(torch.ops.vllm, 'bi_pinned_mxfp8_mm', kernel, raising=False)
    ctx = SimpleNamespace(save_for_backward=lambda *args: None)
    output = _MXFP8Linear.forward(ctx, torch.ones(128, 128, dtype=torch.bfloat16), weight, None)
    assert calls == [True] and output.shape == (128, 128)
    assert b.data_ptr() % 16 == offset_b
    assert torch.equal(b.view(torch.uint8), expected_b)  # original base remains untouched
