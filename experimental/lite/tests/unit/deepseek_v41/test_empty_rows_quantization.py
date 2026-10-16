# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Zero-row inputs: a CP rank may hold no rows of a document, an expert no tokens."""

import pytest
import torch


def test_mxfp8_rows_quantization_accepts_zero_rows(v41_core_te):
    from megatron.lite.primitive.quantization import mxfp8

    empty = mxfp8._quantize_rows(torch.zeros(1, 0, 64, dtype=torch.bfloat16))
    assert empty.values.shape == (1, 0, 64)
    assert empty.scale.shape == (1, 0, 2)
    full = mxfp8._quantize_rows(torch.randn(1, 3, 64, dtype=torch.bfloat16))
    assert full.scale.shape == (1, 3, 2)


@pytest.mark.parametrize('rows', [0, 3])
def test_packed_ue8m0_unpack_accepts_zero_rows(v41_core_te, rows):
    from megatron.lite.primitive.quantization import w4a8_experts

    groups = 5
    packed = torch.full((rows, 2), 0x7F7F7F7F, dtype=torch.int32)
    scale = w4a8_experts._unpack_ue8m0_scales(packed, groups)
    assert scale.shape == (rows, groups)
    if rows:
        assert torch.equal(scale, torch.ones(rows, groups))
